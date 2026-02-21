import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--model-type', type=str, required=True, choices=['bamba', 'nemoh', 'g4l'])
parser.add_argument('--disp-name', type=str, required=True)
args = parser.parse_args()

model_type = args.model_type
disp_name = args.disp_name

# Base checkpoint
if model_type == "bamba":
    # Bamba
    ckpt1 = torch.load("/gpfs/hshen/fms-ckpt/bamba_v2_9b/consolidated.00.pth", map_location='cpu')['model_state']
elif model_type == "nemoh":
    # Nemotron-H
    ckpt1 = torch.load("/gpfs/hshen/fms-ckpt/nemotron_h_8b/consolidated.00.pth", map_location='cpu')['model_state']
else:
    # Granite 4 lite
    ckpt1 = torch.load("/gpfs/hshen/fms-ckpt/granite_4_lite/consolidated.00.pth", map_location='cpu')['model_state']

diff_dict = {}
metrics_dict = {}  # metrics_dict[metric][key] = [val_per_step_pair]
METRICS = ["mean", "std", "mean_abs", "max", "min", "l2", "rms"]
for step in range(500, 6500, 500):
    diff_dict[f"{step-500}-{step}"] = {}
    print(f"============= Step {step-500} vs. {step} =============")
    ckpt2 = torch.load(f"/gpfs/hshen/sft_freeze/{disp_name}/pth/step_{step}/consolidated.00.pth", map_location='cpu')['model_state']
    for key in ckpt1:
        diff = ckpt1[key] - ckpt2[key]
        diff_dict[f"{step-500}-{step}"][key] = diff

        # Show statistics
        numel = diff.numel()
        mean = diff.mean().item()
        std = diff.std().item() if numel > 1 else 0.0
        mean_abs = diff.abs().mean().item()
        max_val = diff.max().item()
        min_val = diff.min().item()
        l2 = diff.float().norm(2).item()
        rms = (diff.float().pow(2).mean().sqrt()).item()
        print(f"  {key}: MeanΔ={mean:.4g}  StdΔ={std:.4g}  Mean|Δ|={mean_abs:.4g}  MaxΔ={max_val:.4g}  MinΔ={min_val:.4g}  L2={l2:.4g}  RMS={rms:.4g}")

        vals = {"mean": mean, "std": std, "mean_abs": mean_abs, "max": max_val, "min": min_val, "l2": l2, "rms": rms}
        for m in METRICS:
            metrics_dict.setdefault(m, {}).setdefault(key, []).append(vals[m])

    # Global summary for this step pair
    all_deltas = torch.cat([diff_dict[f"{step-500}-{step}"][k].flatten() for k in ckpt1])
    print(f"  --- GLOBAL: MeanΔ={all_deltas.mean().item():.4g}  StdΔ={all_deltas.std().item():.4g}  Mean|Δ|={all_deltas.abs().mean().item():.4g}  MaxΔ={all_deltas.max().item():.4g}  MinΔ={all_deltas.min().item():.4g}")

    ckpt1 = ckpt2

# Compare neighboring diffs: cosine similarity between diff vectors at each parameter
step_pairs = list(diff_dict.keys())
step_dict = {}
for i in range(len(step_pairs) - 1):
    sp1, sp2 = step_pairs[i], step_pairs[i + 1]
    print(f"\n============= Cosine Similarity: {sp1} vs. {sp2} =============")
    for key in diff_dict[sp1]:
        d1 = diff_dict[sp1][key].flatten().float()
        d2 = diff_dict[sp2][key].flatten().float()
        cos_sim = torch.nn.functional.cosine_similarity(d1.unsqueeze(0), d2.unsqueeze(0)).item()
        print(f"  {key}: cos_sim={cos_sim:.6f}")
        if key not in step_dict:
            step_dict[key] = []
        step_dict[key] += [cos_sim]

import csv
csv_base = f"/gpfs/hshen/csv/{disp_name}"

# Write per-metric CSVs
all_step_labels = [f"{s-500}-{s}" for s in range(1000, 6500, 500)]
for m in METRICS:
    csv_path_m = f"{csv_base}/diff_{m}.csv"
    with open(csv_path_m, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["param"] + all_step_labels)
        for key, vals in metrics_dict[m].items():
            writer.writerow([key] + [f"{v:.6g}" for v in vals])
    print(f"Wrote {csv_path_m}")

csv_path = f"{csv_base}/cos_sim.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    header = ["param"] + [f"{step_pairs[i]}_vs_{step_pairs[i+1]}" for i in range(len(step_pairs) - 1)]
    writer.writerow(header)
    for key, sims in step_dict.items():
        writer.writerow([key] + [f"{s:.6f}" for s in sims])
print(f"Cosine similarity CSV written to {csv_path}")

