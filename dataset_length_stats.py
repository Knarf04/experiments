"""
Document-length statistics for an fms-fsdp-style dataset directory.

Uses the same file handlers (ArrowHandler / ParquetHandler / AutoHandler) as the
training pipeline in fms-fsdp, so the token counts here match what the loader
would actually see per document.

Typical usage matching the provided CLI:

    python experiments/dataset_length_stats.py \
        --data_path /datasets \
        --datasets smollm-corpus/cosmopedia-v2 \
        --col_name text,content,contents,tokens \
        --file_type auto \
        --tokenizer_path /gpfs/hshen/tokenizer/llama3 \
        --eos_token 128000
"""

import argparse
import json
import math
import os
import statistics
import sys
import time
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the exact handlers used by the training loader.
_FMS_FSDP = os.path.join(os.path.dirname(__file__), "..", "fms-fsdp")
if os.path.isdir(_FMS_FSDP):
    sys.path.insert(0, os.path.abspath(_FMS_FSDP))
from fms_fsdp.utils.dataset_utils import (  # noqa: E402
    ArrowHandler,
    AutoHandler,
    ParquetHandler,
)


_HANDLER_MAP = {
    "arrow": ArrowHandler,
    "hf_parquet": ParquetHandler,
    "auto": AutoHandler,
}


def build_handler(file_type: str, tokenizer_path: str, col_names, max_doclen: int):
    if file_type == "arrow":
        return ArrowHandler(col_names=col_names)
    if file_type == "hf_parquet":
        return ParquetHandler(tokenizer_path, col_names=col_names, max_doclen=max_doclen)
    if file_type == "auto":
        return AutoHandler(tokenizer_path, col_names=col_names, max_doclen=max_doclen)
    raise ValueError(f"Unknown file_type {file_type!r}; expected one of {list(_HANDLER_MAP)}")


def find_shards(root: str, handler, min_bytes: int = 1_000_000):
    """Mirror StreamingDocDataset's file discovery: walk the tree, keep files the
    handler recognizes and that exceed min_bytes (skip empty / tiny shards)."""
    shards = []
    for dirpath, _, filenames in os.walk(root, followlinks=True):
        for name in filenames:
            path = os.path.join(dirpath, name)
            if handler.is_legal(path) and os.path.getsize(path) > min_bytes:
                shards.append(path)
    shards.sort()
    return shards


def iter_doc_lengths(
    shard_paths,
    handler,
    drop_tokens,
    max_docs: int,
    log_every: int,
):
    """Yield the token length of each document, across all shards."""
    seen = 0
    t0 = time.time()
    for shard_idx, path in enumerate(shard_paths):
        try:
            reader = handler.open(path)
            n_docs = handler.length(path)
        except Exception as e:
            print(f"[warn] failed to open {path}: {e}", file=sys.stderr)
            continue

        for doc_idx in range(n_docs):
            try:
                doc = handler.get(reader, doc_idx, drop_tokens)
            except Exception as e:
                print(
                    f"[warn] failed to read doc {doc_idx} in {path}: {e}",
                    file=sys.stderr,
                )
                continue

            yield len(doc)
            seen += 1
            if log_every and seen % log_every == 0:
                rate = seen / max(time.time() - t0, 1e-9)
                print(
                    f"  ... {seen:,} docs ({rate:,.0f} docs/s)"
                    f"  [shard {shard_idx + 1}/{len(shard_paths)}]",
                    file=sys.stderr,
                )
            if max_docs and seen >= max_docs:
                return


def plot_histogram(lengths, title, out_path, n_bins=80):
    """Log-scale histogram of document lengths."""
    if not lengths:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    arr = [max(L, 1) for L in lengths]  # avoid log(0)
    lo = max(min(arr), 1)
    hi = max(arr)
    if hi <= lo:
        hi = lo + 1
    bins = [10 ** x for x in [
        math.log10(lo) + i * (math.log10(hi) - math.log10(lo)) / n_bins
        for i in range(n_bins + 1)
    ]]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(arr, bins=bins, edgecolor="black", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_xlabel("document length (tokens)")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.grid(True, which="both", axis="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"    histogram saved to {out_path}")


def summarize(lengths, buckets):
    if not lengths:
        print("No documents found.")
        return {}

    n = len(lengths)
    total_tokens = sum(lengths)
    mean = total_tokens / n
    try:
        stdev = statistics.stdev(lengths) if n > 1 else 0.0
    except statistics.StatisticsError:
        stdev = 0.0

    sorted_lens = sorted(lengths)

    def pct(p):
        k = min(int(p / 100 * n), n - 1)
        return sorted_lens[k]

    summary = {
        "n_docs": n,
        "total_tokens": total_tokens,
        "min": sorted_lens[0],
        "max": sorted_lens[-1],
        "mean": mean,
        "stdev": stdev,
        "p50": pct(50),
        "p90": pct(90),
        "p95": pct(95),
        "p99": pct(99),
        "p99.9": pct(99.9),
    }

    print("\n=== Document length (tokens) ===")
    print(f"  n_docs       : {n:,}")
    print(f"  total_tokens : {total_tokens:,}")
    print(f"  min / max    : {summary['min']:,} / {summary['max']:,}")
    print(f"  mean ± stdev : {mean:,.1f} ± {stdev:,.1f}")
    print(
        f"  p50 / p90 / p95 / p99 / p99.9 :"
        f" {summary['p50']:,} / {summary['p90']:,} /"
        f" {summary['p95']:,} / {summary['p99']:,} /"
        f" {summary['p99.9']:,}"
    )

    # Bucket counts (length thresholds)
    if buckets:
        counts = Counter()
        for L in lengths:
            for b in buckets:
                if L >= b:
                    counts[b] += 1
        print("\n  docs with length >= threshold:")
        for b in buckets:
            c = counts[b]
            print(f"    >= {b:>8,} : {c:>10,} ({100 * c / n:5.2f}%)")
        summary["buckets_ge"] = {str(b): counts[b] for b in buckets}

    return summary


def parse_tokens_csv(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_cols_csv(s: str):
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_buckets_csv(s: str):
    return sorted(int(x.strip()) for x in s.split(",") if x.strip())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_path", default="/datasets",
                    help="Root directory containing the sub-dataset folders.")
    ap.add_argument("--datasets", default="smollm-corpus/cosmopedia-v2",
                    help="Comma-separated sub-dataset directories under --data_path "
                         "(matches the --datasets arg in the training CLI). "
                         "Example: "
                         "nemotron_cc/kind2=distill,nemotron_cc/kind2=diverse_qa_pairs,"
                         "dolmino/dolmino_mix_1124_dclm,smollm-corpus/cosmopedia-v2")
    ap.add_argument("--col_name", default="text,content,contents,tokens",
                    help="Comma-separated candidate column names; first match wins.")
    ap.add_argument("--file_type", default="auto", choices=list(_HANDLER_MAP.keys()))
    ap.add_argument("--tokenizer_path", default="/gpfs/hshen/tokenizer/llama3",
                    help="HF-compatible tokenizer dir. Only used for parquet files.")
    ap.add_argument("--eos_token", type=int, default=128000,
                    help="Token id stripped from doc boundaries (if present).")
    ap.add_argument("--bos_token", type=int, default=None,
                    help="Optional BOS token id to strip from doc boundaries.")
    ap.add_argument("--extra_drop", default="",
                    help="Additional comma-separated token ids to strip.")
    ap.add_argument("--doc_cutoff", type=int, default=1_000_000,
                    help="max_doclen for parquet handler (characters tokenized).")
    ap.add_argument("--min_bytes", type=int, default=1_000_000,
                    help="Skip shard files smaller than this (matches training).")
    ap.add_argument("--max_docs", type=int, default=0,
                    help="Stop after this many docs per sub-dataset (0 = all).")
    ap.add_argument("--log_every", type=int, default=10_000,
                    help="Progress report cadence (docs). 0 to disable.")
    ap.add_argument("--buckets", default="2048,8192,32768,65536,131072,262144,524288",
                    help="Comma-separated length thresholds for histogram buckets.")
    ap.add_argument("--out_json", default="",
                    help="If set, write per-dataset summaries here.")
    ap.add_argument("--dump_lengths", default="",
                    help="If set, write raw per-doc lengths to this text file "
                         "(one int per line per dataset; files suffixed by index).")
    ap.add_argument("--plot_dir", default="/gpfs/hshen/plots/dataset_stats",
                    help="Directory to save per-dataset histogram plots.")
    ap.add_argument("--hist_bins", type=int, default=80,
                    help="Number of histogram bins (log-spaced).")
    ap.add_argument("--no_plot", action="store_true",
                    help="Disable histogram plotting.")
    args = ap.parse_args()

    col_names = parse_cols_csv(args.col_name)
    buckets = parse_buckets_csv(args.buckets)
    drop = set()
    if args.bos_token is not None:
        drop.add(args.bos_token)
    if args.eos_token is not None:
        drop.add(args.eos_token)
    drop.update(parse_tokens_csv(args.extra_drop))

    datasets = parse_cols_csv(args.datasets)
    all_summaries = {}

    for i, ds in enumerate(datasets):
        root = os.path.join(args.data_path, ds)
        if not os.path.isdir(root):
            print(f"[error] dataset directory not found: {root}", file=sys.stderr)
            continue

        print(f"\n### Dataset {i + 1}/{len(datasets)}: {ds}")
        print(f"    root={root}")

        handler = build_handler(args.file_type, args.tokenizer_path, col_names, args.doc_cutoff)
        shards = find_shards(root, handler, min_bytes=args.min_bytes)
        print(f"    found {len(shards)} shard files")
        if not shards:
            continue

        dump_f = None
        if args.dump_lengths:
            dump_path = f"{args.dump_lengths}.{i}.{ds.replace('/', '_')}.txt"
            dump_f = open(dump_path, "w")
            print(f"    dumping per-doc lengths to {dump_path}")

        lengths = []
        for L in iter_doc_lengths(
            shards, handler, drop, args.max_docs, args.log_every,
        ):
            lengths.append(L)
            if dump_f is not None:
                dump_f.write(f"{L}\n")

        if dump_f is not None:
            dump_f.close()

        summary = summarize(lengths, buckets)
        all_summaries[ds] = summary

        if not args.no_plot and lengths:
            safe_name = ds.replace("/", "_")
            out_path = os.path.join(args.plot_dir, f"{safe_name}.png")
            title = (
                f"{ds}  —  n_docs={len(lengths):,}  "
                f"(mean={summary['mean']:,.0f}, "
                f"p50={summary['p50']:,}, p99={summary['p99']:,}, "
                f"max={summary['max']:,})"
            )
            plot_histogram(lengths, title, out_path, n_bins=args.hist_bins)

        print(f"### Done: {ds}  ({len(lengths):,} docs)\n", flush=True)

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(all_summaries, f, indent=2)
        print(f"\nSummary JSON written to {args.out_json}")


if __name__ == "__main__":
    main()
