"""
Summarize layer*.jsonl files written by the distribution_reg recorder in modeling_bamba.py
/ modeling_mamba2.py / modeling_nemotron_h.py.

Usage:
    python summarize.py <directory> [--out summary.json] [-v]
    python summarize.py <directory> --state-mag-out state_mag.npz \
                                    [--seq-len 65536] [--bin-size 1024]

Each layer*.jsonl file contains one JSON record per forward pass:
    layer_idx      : int
    dt_mean        : [batch, nheads]       softplus(dt+dt_bias).mean(seq)
    dt_var         : [batch, nheads]       softplus(dt+dt_bias).var(seq)
    forget_mean    : [batch, nheads]       exp(A*dt_sp).mean(seq)
    forget_var     : [batch, nheads]       exp(A*dt_sp).var(seq)
    spectrum_std   : [batch]               std of forget_mean over head dim
    erf            : [batch, nheads]       effective receptive field (mmd_ssd_last)
    state_cos_sim  : [batch, nheads, npos, npos]  cosine sim between states at intervals
    state_mag_bin  : [batch, nheads, nbins]  exact ||h_s^h||_F at bin boundaries
    state_bin_size : int                     bin width in tokens (state_mag_bin only)
    seq_len        : int                     sequence length used (state_mag_bin only)

Aggregation rules (samples are i.i.d., averaging over batch is correct):
    dt_mean, forget_mean, erf   : mean over batch  → [nheads]
    dt_var, forget_var          : mean over batch  → [nheads]   (within-seq variance; tells how dynamic the gate is)
    spectrum_std                : mean over batch  → scalar
    state_cos_sim               : mean over batch  → [nheads, npos, npos]
    state_mag_bin               : mean over batch  → [nheads, nbins]   (also: std/sem/median/p25/p75 + drift_ratio/log_slope)
All fields are then averaged across records (forward passes) in the file.

For state_mag_bin we additionally save a richer per-layer summary (mean / std /
sem / median / p25 / p75 over samples plus per-head drift_ratio and log_slope)
to an .npz when --state-mag-out is set. Only records sharing a single
(seq_len, state_bin_size) combo are included; if not specified via --seq-len /
--bin-size, the most common combo present in the file is used.
"""

import argparse
import json
import os
import glob
import re

import numpy as np


# Fields that have a batch dimension to average over, and their output shape description
FIELDS = {
    "dt_mean":       "nheads",
    "dt_var":        "nheads",
    "forget_mean":   "nheads",
    "forget_var":    "nheads",
    "spectrum_std":  "scalar",
    "erf":           "nheads",
    "state_cos_sim": "nheads x npos x npos",
    "state_mag_bin": "nheads x nbins",
}


def _resolve_state_mag_filter(records):
    """Pick the (seq_len, state_bin_size) combo with the most records carrying state_mag_bin."""
    combos: dict[tuple, int] = {}
    for r in records:
        if "state_mag_bin" not in r:
            continue
        key = (r.get("seq_len"), r.get("state_bin_size"))
        combos[key] = combos.get(key, 0) + 1
    if not combos:
        return None, None
    return max(combos.items(), key=lambda kv: kv[1])[0]


def _filter_records(records, target_seq_len, target_bin_size):
    """Keep records where seq_len/state_bin_size match the targets (None = no constraint)."""
    out = []
    for r in records:
        if target_seq_len is not None and r.get("seq_len") != target_seq_len:
            continue
        if target_bin_size is not None and r.get("state_bin_size") != target_bin_size:
            continue
        out.append(r)
    return out


def compute_state_mag_stats(records, eps: float = 1e-12) -> dict:
    """Rich per-(head, bin) and per-head summary across all sample sequences in `records`.

    All input records must already share a single (seq_len, state_bin_size); the
    caller is responsible for filtering. Concatenates state_mag_bin across records
    and the within-record batch axis to produce one [N, H, n_bins] sample tensor.
    """
    arrays = [np.asarray(r["state_mag_bin"], dtype=np.float32) for r in records
              if "state_mag_bin" in r]
    if not arrays:
        return {}

    shapes = {a.shape[1:] for a in arrays}
    assert len(shapes) == 1, f"Inconsistent state_mag_bin shapes: {shapes}"

    mag = np.concatenate(arrays, axis=0)  # [N, H, n_bins]
    n_samples, n_heads, n_bins = mag.shape

    bin_size = int(records[0]["state_bin_size"])
    seq_len = int(records[0]["seq_len"])
    positions = np.arange(n_bins, dtype=np.int64) * bin_size + (bin_size - 1)

    mean = mag.mean(axis=0)
    std = mag.std(axis=0)
    sem = std / np.sqrt(max(n_samples, 1))
    median = np.median(mag, axis=0)
    p25 = np.percentile(mag, 25, axis=0)
    p75 = np.percentile(mag, 75, axis=0)

    drift_ratio = mean.max(axis=1) / (mean.min(axis=1) + eps)

    log_mean = np.log(mean + eps)
    pos_f = positions.astype(np.float64)
    log_slope = np.zeros(n_heads, dtype=np.float32)
    for h in range(n_heads):
        coeffs = np.polyfit(pos_f, log_mean[h, :], deg=1)
        log_slope[h] = float(coeffs[0])

    return {
        "n_samples":      int(n_samples),
        "n_heads":        int(n_heads),
        "n_bins":         int(n_bins),
        "seq_len":        int(seq_len),
        "state_bin_size": int(bin_size),
        "positions":      positions,
        "mean":           mean.astype(np.float32),
        "std":            std.astype(np.float32),
        "sem":            sem.astype(np.float32),
        "median":         median.astype(np.float32),
        "p25":            p25.astype(np.float32),
        "p75":            p75.astype(np.float32),
        "drift_ratio":    drift_ratio.astype(np.float32),
        "log_slope":      log_slope.astype(np.float32),
    }


def save_state_mag_npz(summaries, path: str) -> None:
    """Stack per-layer state_mag stats and save to a single .npz.

    Stored arrays (L = number of layers with stats present):
        layer_idx    : [L]
        positions    : [n_bins]               (shared across layers)
        mean / std / sem / median / p25 / p75 : [L, H, n_bins]
        drift_ratio / log_slope               : [L, H]
        seq_len, state_bin_size, n_samples    : scalars (taken from the first present layer)
    """
    by_layer = []
    for s in summaries:
        st = s.get("state_mag_stats")
        if st:
            by_layer.append((s["layer_idx"], st))
    if not by_layer:
        print(f"  [state_mag] no layers with state_mag stats — skipping save to {path}")
        return

    by_layer.sort(key=lambda kv: kv[0])
    layer_idx = np.array([li for li, _ in by_layer], dtype=np.int64)

    ref = by_layer[0][1]
    positions = ref["positions"]
    seq_len = ref["seq_len"]
    bin_size = ref["state_bin_size"]
    n_samples = ref["n_samples"]
    n_heads = ref["n_heads"]
    n_bins = ref["n_bins"]

    for li, st in by_layer:
        if st["n_heads"] != n_heads or st["n_bins"] != n_bins:
            raise ValueError(
                f"layer {li}: shape mismatch (H={st['n_heads']}, nbins={st['n_bins']}) "
                f"vs reference (H={n_heads}, nbins={n_bins})"
            )

    def stack(field):
        return np.stack([st[field] for _, st in by_layer], axis=0)

    np.savez(
        path,
        layer_idx=layer_idx,
        positions=positions,
        seq_len=np.int64(seq_len),
        state_bin_size=np.int64(bin_size),
        n_samples=np.int64(n_samples),
        mean=stack("mean"),
        std=stack("std"),
        sem=stack("sem"),
        median=stack("median"),
        p25=stack("p25"),
        p75=stack("p75"),
        drift_ratio=stack("drift_ratio"),
        log_slope=stack("log_slope"),
    )
    print(f"\nstate_mag stats saved to {path}  (L={len(by_layer)}, H={n_heads}, bins={n_bins})")


def load_layer(path: str, target_seq_len=None, target_bin_size=None) -> dict:
    """
    Read all records from a layer jsonl.

    For each field, average over the batch dim (axis 0) per record,
    then average those per-record means across all records.

    For state_mag_bin specifically, records are filtered to a single
    (seq_len, state_bin_size) combo before aggregation — auto-resolved to the
    most common combo if `target_seq_len` / `target_bin_size` are not given —
    and a richer per-layer stats dict is attached under "state_mag_stats".

    Returns a dict with:
        layer_idx          : int
        n_records          : int
        <field>            : np.ndarray  (final aggregated array)
        state_mag_stats    : dict        (only if state_mag_bin present)
        state_mag_seq_len  : int         (resolved filter)
        state_mag_bin_size : int         (resolved filter)
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

    has_state_mag = any("state_mag_bin" in r for r in records)
    sm_seq_len = target_seq_len
    sm_bin_size = target_bin_size
    if has_state_mag and (sm_seq_len is None or sm_bin_size is None):
        auto_seq, auto_bin = _resolve_state_mag_filter(records)
        if sm_seq_len is None:
            sm_seq_len = auto_seq
        if sm_bin_size is None:
            sm_bin_size = auto_bin
    sm_records = (_filter_records(records, sm_seq_len, sm_bin_size)
                  if has_state_mag else [])

    for key in FIELDS:
        if key not in records[0]:
            continue
        if key == "state_mag_bin":
            # Aggregate only across the filtered, shape-compatible subset.
            if not sm_records:
                continue
            all_samples = np.concatenate(
                [np.array(r[key], dtype=np.float32) for r in sm_records], axis=0
            )
            result[key] = all_samples.mean(axis=0)
            continue

        # Pool all records and batch entries as i.i.d. samples, then average.
        # Each record value is (batch, ...) so concatenate along axis 0 first.
        all_samples = np.concatenate(
            [np.array(r[key], dtype=np.float32) for r in records], axis=0
        )  # (n_records * batch, ...)
        result[key] = all_samples.mean(axis=0)  # → (...)

    if sm_records:
        result["state_mag_stats"] = compute_state_mag_stats(sm_records)
        result["state_mag_seq_len"] = sm_seq_len
        result["state_mag_bin_size"] = sm_bin_size

    return result


def print_summary(summaries: list[dict], verbose: bool = False) -> None:
    header = (f"{'layer':>6}  {'recs':>5}  "
              f"{'dt_mean':>9}  {'dt_var':>9}  "
              f"{'fgt_mean':>9}  {'fgt_var':>9}  "
              f"{'erf':>9}  {'spec_std':>9}  {'cos_sim':>9}  "
              f"{'mag_mean':>9}  {'mag_drift':>9}")
    print(header)
    print("-" * len(header))

    for s in sorted(summaries, key=lambda x: x.get("layer_idx", 0)):
        def sc(key):
            """Scalar display: mean over head dim (or the value itself if already scalar)."""
            v = s.get(key)
            if v is None:
                return float("nan")
            return float(np.mean(v))

        st = s.get("state_mag_stats")
        mag_mean = float(st["mean"].mean()) if st else float("nan")
        mag_drift = float(np.median(st["drift_ratio"])) if st else float("nan")

        row = (f"{s['layer_idx']:>6}  {s['n_records']:>5}  "
               f"{sc('dt_mean'):>9.4f}  {sc('dt_var'):>9.4f}  "
               f"{sc('forget_mean'):>9.4f}  {sc('forget_var'):>9.4f}  "
               f"{sc('erf'):>9.4f}  {sc('spectrum_std'):>9.4f}  "
               f"{sc('state_cos_sim'):>9.4f}  "
               f"{mag_mean:>9.4f}  {mag_drift:>9.4f}")
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
            if st is not None:
                print(f"    state_mag  N={st['n_samples']}  H={st['n_heads']}  "
                      f"bins={st['n_bins']}  bin_size={st['state_bin_size']}  "
                      f"seq_len={st['seq_len']}")
                print(f"      drift_ratio: median={float(np.median(st['drift_ratio'])):.3f}  "
                      f"max={float(st['drift_ratio'].max()):.3f}")
                print(f"      log_slope:   median={float(np.median(st['log_slope'])):.3e}  "
                      f"max={float(np.abs(st['log_slope']).max()):.3e}")


def summaries_to_json(summaries: list[dict]) -> list[dict]:
    """Convert numpy arrays to lists for JSON serialization."""
    out = []
    for s in summaries:
        entry = {"layer_idx": s["layer_idx"], "n_records": s["n_records"]}
        for key in FIELDS:
            if key in s:
                entry[key] = np.asarray(s[key]).tolist()
        st = s.get("state_mag_stats")
        if st is not None:
            entry["state_mag_stats"] = {
                "n_samples":      st["n_samples"],
                "n_heads":        st["n_heads"],
                "n_bins":         st["n_bins"],
                "seq_len":        st["seq_len"],
                "state_bin_size": st["state_bin_size"],
                "positions":      st["positions"].tolist(),
                "mean":           st["mean"].tolist(),
                "std":            st["std"].tolist(),
                "sem":            st["sem"].tolist(),
                "median":         st["median"].tolist(),
                "p25":            st["p25"].tolist(),
                "p75":            st["p75"].tolist(),
                "drift_ratio":    st["drift_ratio"].tolist(),
                "log_slope":      st["log_slope"].tolist(),
            }
        out.append(entry)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Summarize bamba distribution_reg JSONL files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("directory", help="Directory containing layer*.jsonl files")
    parser.add_argument("--out", default=None, help="Save JSON summary to this path")
    parser.add_argument("--state-mag-out", default=None,
                        help="Save state_mag rich per-layer stats to this .npz path")
    parser.add_argument("--seq-len", type=int, default=None,
                        help="Filter records to this seq_len for state_mag aggregation")
    parser.add_argument("--bin-size", type=int, default=None,
                        help="Filter records to this state_bin_size for state_mag aggregation")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-head arrays for dt_mean, forget_mean, erf, state_mag")
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
        s = load_layer(path, target_seq_len=args.seq_len, target_bin_size=args.bin_size)
        if not s:
            print(f"  [skip] {path} — empty")
            continue
        summaries.append(s)

    print_summary(summaries, verbose=args.verbose)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summaries_to_json(summaries), f, indent=2)
        print(f"\nSummary saved to {args.out}")

    if args.state_mag_out:
        save_state_mag_npz(summaries, args.state_mag_out)


if __name__ == "__main__":
    main()
