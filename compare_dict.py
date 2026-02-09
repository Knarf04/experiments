import torch

ckpt1 = torch.load(f"/gpfs/hshen/bamba_freeze/bamba-32k-rerun-ckpt-2/pth/step_500/consolidated.00.pth", map_location='cpu')['model_state']
diff_dict = {}
for step in range(1000, 6500, 500):
    diff_dict[f"{step-500}-{step}"] = {}
    print(f"============= Step {step-500} vs. {step} =============")
    ckpt2 = torch.load(f"/gpfs/hshen/bamba_freeze/bamba-32k-rerun-ckpt-2/pth/step_{step}/consolidated.00.pth", map_location='cpu')['model_state']
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
        print(f"  {key}: MeanΔ={mean:.4g}  StdΔ={std:.4g}  Mean|Δ|={mean_abs:.4g}  MaxΔ={max_val:.4g}  MinΔ={min_val:.4g}  L2={l2:.4g}")

    # Global summary for this step pair
    all_deltas = torch.cat([diff_dict[f"{step-500}-{step}"][k].flatten() for k in ckpt1])
    print(f"  --- GLOBAL: MeanΔ={all_deltas.mean().item():.4g}  StdΔ={all_deltas.std().item():.4g}  Mean|Δ|={all_deltas.abs().mean().item():.4g}  MaxΔ={all_deltas.max().item():.4g}  MinΔ={all_deltas.min().item():.4g}")

    ckpt1 = ckpt2

# Compare neighboring diffs: cosine similarity between diff vectors at each parameter
step_pairs = list(diff_dict.keys())
for i in range(len(step_pairs) - 1):
    sp1, sp2 = step_pairs[i], step_pairs[i + 1]
    print(f"\n============= Cosine Similarity: {sp1} vs. {sp2} =============")
    for key in diff_dict[sp1]:
        d1 = diff_dict[sp1][key].flatten().float()
        d2 = diff_dict[sp2][key].flatten().float()
        cos_sim = torch.nn.functional.cosine_similarity(d1.unsqueeze(0), d2.unsqueeze(0)).item()
        print(f"  {key}: cos_sim={cos_sim:.6f}")
