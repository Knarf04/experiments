import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt1', type=str, required=True, help="Path to base checkpoint (.pth)")
parser.add_argument('--ckpt2', type=str, required=True, help="Path to fine-tuned checkpoint (.pth)")
parser.add_argument('--rank-thresholds', type=float, nargs='+', default=[0.90, 0.95, 0.99],
                    help="Cumulative energy thresholds for effective rank (default: 0.90 0.95 0.99)")
args = parser.parse_args()

ckpt1 = torch.load(args.ckpt1, map_location='cpu')['model_state']
ckpt2 = torch.load(args.ckpt2, map_location='cpu')['model_state']

METRICS = ["mean", "std", "mean_abs", "max", "min", "l2", "rms"]

print(f"============= {args.ckpt1} vs. {args.ckpt2} =============")
for key in ckpt1:
    diff = ckpt1[key] - ckpt2[key]

    numel = diff.numel()
    mean = diff.mean().item()
    std = diff.std().item() if numel > 1 else 0.0
    mean_abs = diff.abs().mean().item()
    max_val = diff.max().item()
    min_val = diff.min().item()
    l2 = diff.float().norm(2).item()
    rms = (diff.float().pow(2).mean().sqrt()).item()
    print(f"  {key}: MeanΔ={mean:.4g}  StdΔ={std:.4g}  Mean|Δ|={mean_abs:.4g}  MaxΔ={max_val:.4g}  MinΔ={min_val:.4g}  L2={l2:.4g}  RMS={rms:.4g}")

# Global summary
all_deltas = torch.cat([ckpt1[k].flatten() - ckpt2[k].flatten() for k in ckpt1])
print(f"  --- GLOBAL: MeanΔ={all_deltas.mean().item():.4g}  StdΔ={all_deltas.std().item():.4g}  Mean|Δ|={all_deltas.abs().mean().item():.4g}  MaxΔ={all_deltas.max().item():.4g}  MinΔ={all_deltas.min().item():.4g}")

# ---- SVD rank analysis on 2D weight diffs ----
print(f"\n============= SVD Rank Analysis (thresholds: {args.rank_thresholds}) =============")
print(f"  {'param':<60s} {'shape':>14s}  {'rank':>4s}  " + "  ".join(f"r@{t:.0%}" for t in args.rank_thresholds) + "  top-1%  top-5%")

for key in ckpt1:
    diff = (ckpt1[key] - ckpt2[key]).float()

    # Only analyze 2D (matrix) parameters
    if diff.ndim != 2:
        continue

    # Skip if diff is all zeros (unchanged parameter)
    if diff.norm() == 0:
        print(f"  {key:<60s} {str(list(diff.shape)):>14s}  unchanged")
        continue

    S = torch.linalg.svdvals(diff)
    full_rank = min(diff.shape)

    # Cumulative energy (fraction of total squared Frobenius norm)
    energy = (S ** 2).cumsum(0) / (S ** 2).sum()

    # Effective rank at each threshold: smallest r such that energy[r-1] >= threshold
    ranks_at_thresh = []
    for t in args.rank_thresholds:
        r = int((energy >= t).nonzero(as_tuple=True)[0][0].item()) + 1
        ranks_at_thresh.append(r)

    # Energy captured by top 1% and top 5% of singular values
    r1 = max(1, full_rank // 100)
    r5 = max(1, full_rank // 20)
    energy_top1 = (S[:r1] ** 2).sum() / (S ** 2).sum()
    energy_top5 = (S[:r5] ** 2).sum() / (S ** 2).sum()

    rank_strs = "  ".join(f"{r:>5d}" for r in ranks_at_thresh)
    print(f"  {key:<60s} {str(list(diff.shape)):>14s}  {full_rank:>4d}  {rank_strs}  {energy_top1:.3f}   {energy_top5:.3f}")
