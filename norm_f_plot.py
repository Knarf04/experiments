import torch
import matplotlib.pyplot as plt

ckpt = torch.load("/gpfs/davis/granites/bamba-merged/consolidated_ckpt.pth", map_location='cpu')['model_state']

steps = [0]
key = 'backbone.norm_f.weight'
L2_norm = torch.linalg.norm

norm_f = [L2_norm(ckpt[key]).item()]

for step in range(500, 6500, 500):
    ckpt = torch.load(f"/gpfs/hshen/bamba_freeze/bamba-32k-rerun-ckpt-2/pth/step_{step}/consolidated.00.pth", map_location='cpu')['model_state']
    steps += [step]
    norm_f += [L2_norm(ckpt[key]).item()]
    
plt.plot(steps, norm_f)
plt.savefig("/gpfs/hshen/figs/norm_f_bamba.png", dpi=600)
plt.show()