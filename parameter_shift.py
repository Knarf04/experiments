import argparse
import collections
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def compare_checkpoints(ckpt_a: str, ckpt_b: str) -> dict[str, torch.Tensor]:
    """Load two HF checkpoints, compute weight differences (a - b), print statistics, return the diff."""

    model_a = AutoModelForCausalLM.from_pretrained(ckpt_a, torch_dtype=torch.float32)
    model_b = AutoModelForCausalLM.from_pretrained(ckpt_b, torch_dtype=torch.float32)

    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()

    common_keys = sorted(set(sd_a.keys()) & set(sd_b.keys()))

    diff = {}
    # accumulate per-layer stats (layer = prefix up to numeric layer id, e.g. "model.layers.0")
    layer_stats: dict[str, list[tuple[int, float, float, float, float]]] = collections.defaultdict(list)

    # ── Entrywise statistics ──
    print("=" * 100)
    print("ENTRYWISE  (per parameter tensor, Δ = A - B)")
    print("=" * 100)
    fmt = "{:<55s} {:>12s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s}"
    print(fmt.format("Parameter", "Shape", "MeanΔ", "StdΔ", "Mean|Δ|", "MaxΔ", "MinΔ"))
    print("-" * 100)

    for key in common_keys:
        delta = sd_a[key] - sd_b[key]
        diff[key] = delta

        numel = delta.numel()
        mean = delta.mean().item()
        std = delta.std().item() if numel > 1 else 0.0
        mean_abs = delta.abs().mean().item()
        max_val = delta.max().item()
        min_val = delta.min().item()

        shape_str = str(list(sd_a[key].shape))
        print(fmt.format(
            key if len(key) <= 54 else "…" + key[-53:],
            shape_str if len(shape_str) <= 12 else shape_str[:11] + "…",
            f"{mean:.4g}",
            f"{std:.4g}",
            f"{mean_abs:.4g}",
            f"{max_val:.4g}",
            f"{min_val:.4g}",
        ))

        parts = key.split(".")
        layer_key = _layer_prefix(parts)
        layer_stats[layer_key].append((numel, mean * numel, float(delta.pow(2).sum().item()),
                                       max_val, min_val))

    # ── Layerwise statistics ──
    print()
    print("=" * 100)
    print("LAYERWISE  (aggregated by layer prefix, Δ = A - B)")
    print("=" * 100)
    layer_fmt = "{:<40s} {:>12s} {:>12s} {:>12s} {:>12s} {:>12s}"
    print(layer_fmt.format("Layer", "#Params", "MeanΔ", "RMSE", "MaxΔ", "MinΔ"))
    print("-" * 100)

    for layer_key in sorted(layer_stats.keys()):
        entries = layer_stats[layer_key]
        n = sum(e[0] for e in entries)
        sum_vals = sum(e[1] for e in entries)
        sum_sq = sum(e[2] for e in entries)
        mx = max(e[3] for e in entries)
        mn = min(e[4] for e in entries)
        print(layer_fmt.format(
            layer_key if len(layer_key) <= 39 else "…" + layer_key[-38:],
            str(n),
            f"{sum_vals / n:.4g}",
            f"{(sum_sq / n) ** 0.5:.4g}",
            f"{mx:.4g}",
            f"{mn:.4g}",
        ))

    # ── Global summary ──
    print()
    all_deltas = torch.cat([diff[k].flatten() for k in common_keys])
    print("=" * 100)
    print("GLOBAL  (Δ = A - B)")
    print("=" * 100)
    print(f"  Total parameters : {all_deltas.numel():,}")
    print(f"  Mean Δ           : {all_deltas.mean().item():.6g}")
    print(f"  Std Δ            : {all_deltas.std().item():.6g}")
    print(f"  Mean |Δ|         : {all_deltas.abs().mean().item():.6g}")
    print(f"  Max Δ            : {all_deltas.max().item():.6g}")
    print(f"  Min Δ            : {all_deltas.min().item():.6g}")
    print(f"  RMSE             : {all_deltas.pow(2).mean().sqrt().item():.6g}")
    print("=" * 100)

    return diff


def compare_diffs(
    diff_a: dict[str, torch.Tensor],
    diff_b: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Compare two diffs via cosine similarity to measure whether parameter shifts are aligned.

    Returns a dict mapping each parameter name to its cosine similarity.
    """
    common_keys = sorted(set(diff_a.keys()) & set(diff_b.keys()))

    cos = torch.nn.CosineSimilarity(dim=0)

    results: dict[str, float] = {}
    # accumulate flattened vectors per layer for layerwise cosine sim
    layer_vecs_a: dict[str, list[torch.Tensor]] = collections.defaultdict(list)
    layer_vecs_b: dict[str, list[torch.Tensor]] = collections.defaultdict(list)

    # ── Entrywise cosine similarity ──
    print("=" * 90)
    print("ENTRYWISE COSINE SIMILARITY  (per parameter tensor)")
    print("=" * 90)
    fmt = "{:<55s} {:>12s} {:>12s} {:>12s}"
    print(fmt.format("Parameter", "CosSim", "‖Δ_a‖", "‖Δ_b‖"))
    print("-" * 90)

    for key in common_keys:
        a = diff_a[key].flatten().float()
        b = diff_b[key].flatten().float()

        norm_a = a.norm().item()
        norm_b = b.norm().item()

        if norm_a == 0 and norm_b == 0:
            sim = 1.0  # both zero → no change in either, trivially aligned
        elif norm_a == 0 or norm_b == 0:
            sim = 0.0  # one changed, the other didn't
        else:
            sim = cos(a, b).item()

        results[key] = sim

        print(fmt.format(
            key if len(key) <= 54 else "…" + key[-53:],
            f"{sim:+.6f}",
            f"{norm_a:.4g}",
            f"{norm_b:.4g}",
        ))

        layer_key = _layer_prefix(key.split("."))
        layer_vecs_a[layer_key].append(a)
        layer_vecs_b[layer_key].append(b)

    # ── Layerwise cosine similarity ──
    print()
    print("=" * 90)
    print("LAYERWISE COSINE SIMILARITY  (params concatenated per layer)")
    print("=" * 90)
    layer_fmt = "{:<40s} {:>12s} {:>12s} {:>12s}"
    print(layer_fmt.format("Layer", "CosSim", "‖Δ_a‖", "‖Δ_b‖"))
    print("-" * 90)

    for layer_key in sorted(layer_vecs_a.keys()):
        a = torch.cat(layer_vecs_a[layer_key])
        b = torch.cat(layer_vecs_b[layer_key])
        norm_a = a.norm().item()
        norm_b = b.norm().item()
        if norm_a == 0 and norm_b == 0:
            sim = 1.0
        elif norm_a == 0 or norm_b == 0:
            sim = 0.0
        else:
            sim = cos(a, b).item()
        print(layer_fmt.format(
            layer_key if len(layer_key) <= 39 else "…" + layer_key[-38:],
            f"{sim:+.6f}",
            f"{norm_a:.4g}",
            f"{norm_b:.4g}",
        ))

    # ── Global cosine similarity ──
    print()
    all_a = torch.cat([diff_a[k].flatten().float() for k in common_keys])
    all_b = torch.cat([diff_b[k].flatten().float() for k in common_keys])
    global_sim = cos(all_a, all_b).item()
    print("=" * 90)
    print("GLOBAL COSINE SIMILARITY")
    print("=" * 90)
    print(f"  cos(Δ_a, Δ_b) : {global_sim:+.6f}")
    print(f"  ‖Δ_a‖         : {all_a.norm().item():.6g}")
    print(f"  ‖Δ_b‖         : {all_b.norm().item():.6g}")
    print("=" * 90)

    return results


def save_diff(diff: dict[str, torch.Tensor], path: str | Path) -> None:
    """Save a diff dict to disk as a .pt file."""
    torch.save(diff, path)
    print(f"Saved diff ({len(diff)} tensors) → {path}")


def load_diff(path: str | Path) -> dict[str, torch.Tensor]:
    """Load a diff dict from a .pt file."""
    diff = torch.load(path, weights_only=True)
    print(f"Loaded diff ({len(diff)} tensors) ← {path}")
    return diff


def save_compare_results(results: dict[str, float], path: str | Path) -> None:
    """Save compare_diffs results as plain text (one parameter per line)."""
    path = Path(path)
    lines = [f"{key}\t{sim:+.6f}" for key, sim in sorted(results.items())]
    path.write_text("\n".join(lines) + "\n")
    print(f"Saved comparison results ({len(results)} entries) → {path}")


def _layer_prefix(parts: list[str]) -> str:
    """Heuristic: walk dotted name parts, stop after the first numeric segment."""
    prefix = []
    for p in parts:
        prefix.append(p)
        if p.isdigit():
            return ".".join(prefix)
    # no numeric layer id → use first two components (e.g. "model.embed_tokens")
    return ".".join(parts[:min(2, len(parts))])


def main():
    parser = argparse.ArgumentParser(description="Compare HF model checkpoints weight-by-weight.")
    sub = parser.add_subparsers(dest="mode", required=True)

    # ── two_ckpt ──
    p_ckpt = sub.add_parser("ckpts", help="Compute diff (A - B) between two checkpoints")
    p_ckpt.add_argument("ckpt_a", help="Path or HF hub ID for checkpoint A")
    p_ckpt.add_argument("name_a", help="Name for checkpoint A")
    p_ckpt.add_argument("ckpt_b", help="Path or HF hub ID for checkpoint B")
    p_ckpt.add_argument("name_b", help="Name for checkpoint B")
    p_ckpt.add_argument("--save", metavar="DIR", default=None,
                       help="Directory to save the diff .pt file")

    # ── two_diffs ──
    p_diff = sub.add_parser("diffs", help="Compare alignment of two precomputed diffs")
    p_diff.add_argument("diff_a", help="Path to first diff .pt file")
    p_diff.add_argument("diff_b", help="Path to second diff .pt file")
    p_diff.add_argument("--save", metavar="PATH", default=None,
                         help="Path to save comparison results as plain text")

    args = parser.parse_args()

    if args.mode == "ckpts":
        diff = compare_checkpoints(args.ckpt_a, args.ckpt_b)
        if args.save:
            out_dir = Path(args.save)
            out_dir.mkdir(parents=True, exist_ok=True)
            save_diff(diff, out_dir / f"diff_{args.name_a}_{args.name_b}.pt")

    elif args.mode == "diffs":
        diff_a = load_diff(args.diff_a)
        diff_b = load_diff(args.diff_b)
        results = compare_diffs(diff_a, diff_b)
        if args.save:
            save_compare_results(results, args.save)


if __name__ == "__main__":
    main()
