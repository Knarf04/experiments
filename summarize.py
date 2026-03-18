"""
Summarize layer*.jsonl files written by the distribution_reg recorder in modeling_bamba.py.

Usage:
    python summarize.py <directory> [--out summary.json] [-v]

Each layer*.jsonl file contains one JSON record per forward pass:
    layer_idx     : int
    dt_mean       : [batch, nheads]       softplus(dt+dt_bias).mean(seq)
    dt_var        : [batch, nheads]       softplus(dt+dt_bias).var(seq)
    forget_mean   : [batch, nheads]       exp(A*dt_sp).mean(seq)
    forget_var    : [batch, nheads]       exp(A*dt_sp).var(seq)
    spectrum_std  : [batch]               std of forget_mean over head dim
    erf           : [batch, nheads]       effective receptive field (mmd_ssd_last)
    state_cos_sim : [batch, nheads, npos, npos]  cosine sim between states at intervals

Aggregation rules (samples are i.i.d., averaging over batch is correct):
    dt_mean, forget_mean, erf   : mean over batch  → [nheads]
    dt_var, forget_var          : mean over batch  → [nheads]   (within-seq variance; tells how dynamic the gate is)
    spectrum_std                : mean over batch  → scalar
    state_cos_sim               : mean over batch  → [nheads, npos, npos]
All fields are then averaged across records (forward passes) in the file.
"""

import argparse
import json
import os
import glob
import re

import numpy as np


# Fields that have a batch dimension to average over, and their output shape description
FIELDS = {
    "dt_mean":      "nheads",
    "dt_var":       "nheads",
    "forget_mean":  "nheads",
    "forget_var":   "nheads",
    "spectrum_std": "scalar",
    "erf":          "nheads",
    "state_cos_sim": "nheads x npos x npos",
}


def load_layer(path: str) -> dict:
    """
    Read all records from a layer jsonl.

    For each field, average over the batch dim (axis 0) per record,
    then average those per-record means across all records.

    Returns a dict with:
        layer_idx  : int
        n_records  : int
        <field>    : np.ndarray  (final aggregated array)
    """
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        return {}

    result = {
        "layer_idx": int(records[0]["layer_idx"]),
        "n_records": len(records),
    }

    for key in FIELDS:
        if key not in records[0]:
            continue

        # Pool all records and batch entries as i.i.d. samples, then average.
        # Each record value is (batch, ...) so concatenate along axis 0 first.
        all_samples = np.concatenate(
            [np.array(r[key], dtype=np.float32) for r in records], axis=0
        )  # (n_records * batch, ...)
        result[key] = all_samples.mean(axis=0)  # → (...)

    return result


def print_summary(summaries: list[dict], verbose: bool = False) -> None:
    header = (f"{'layer':>6}  {'recs':>5}  "
              f"{'dt_mean':>9}  {'dt_var':>9}  "
              f"{'fgt_mean':>9}  {'fgt_var':>9}  "
              f"{'erf':>9}  {'spec_std':>9}  {'cos_sim':>9}")
    print(header)
    print("-" * len(header))

    for s in sorted(summaries, key=lambda x: x.get("layer_idx", 0)):
        def sc(key):
            """Scalar display: mean over head dim (or the value itself if already scalar)."""
            v = s.get(key)
            if v is None:
                return float("nan")
            return float(np.mean(v))

        row = (f"{s['layer_idx']:>6}  {s['n_records']:>5}  "
               f"{sc('dt_mean'):>9.4f}  {sc('dt_var'):>9.4f}  "
               f"{sc('forget_mean'):>9.4f}  {sc('forget_var'):>9.4f}  "
               f"{sc('erf'):>9.4f}  {sc('spectrum_std'):>9.4f}  "
               f"{sc('state_cos_sim'):>9.4f}")
        print(row)

        if verbose:
            for key in ("dt_mean", "forget_mean", "erf"):
                v = s.get(key)
                if v is not None and np.asarray(v).ndim >= 1:
                    print(f"    {key} per-head: "
                          f"{np.array2string(np.asarray(v), precision=4, suppress_small=True)}")
            v = s.get("state_cos_sim")
            if v is not None:
                arr = np.asarray(v)
                print(f"    state_cos_sim shape={arr.shape}  "
                      f"mean={arr.mean():.4f}  min={arr.min():.4f}  max={arr.max():.4f}")


def summaries_to_json(summaries: list[dict]) -> list[dict]:
    """Convert numpy arrays to lists for JSON serialization."""
    out = []
    for s in summaries:
        entry = {"layer_idx": s["layer_idx"], "n_records": s["n_records"]}
        for key in FIELDS:
            if key in s:
                entry[key] = np.asarray(s[key]).tolist()
        out.append(entry)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Summarize bamba distribution_reg JSONL files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("directory", help="Directory containing layer*.jsonl files")
    parser.add_argument("--out", default=None, help="Save JSON summary to this path")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-head arrays for dt_mean, forget_mean, erf")
    args = parser.parse_args()

    files = sorted(
        glob.glob(os.path.join(args.directory, "**", "layer*.jsonl"), recursive=True),
        key=lambda p: int(re.search(r"layer(\d+)", os.path.basename(p)).group(1)),
    )
    if not files:
        files = sorted(
            glob.glob(os.path.join(args.directory, "layer*.jsonl")),
            key=lambda p: int(re.search(r"layer(\d+)", os.path.basename(p)).group(1)),
        )
    if not files:
        print(f"No layer*.jsonl files found under {args.directory}")
        return

    print(f"Found {len(files)} layer file(s) in {args.directory}\n")

    summaries = []
    for path in files:
        s = load_layer(path)
        if not s:
            print(f"  [skip] {path} — empty")
            continue
        summaries.append(s)

    print_summary(summaries, verbose=args.verbose)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summaries_to_json(summaries), f, indent=2)
        print(f"\nSummary saved to {args.out}")


if __name__ == "__main__":
    main()
