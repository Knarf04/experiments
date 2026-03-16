import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt1', type=str, required=True, help="Path to first checkpoint (.pth)")
parser.add_argument('--ckpt2', type=str, required=True, help="Path to second checkpoint (.pth)")
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
