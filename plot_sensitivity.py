import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_summary(path: str) -> dict[int, dict]:
    """Load summary JSON (from summarize.py --out) keyed by layer_idx."""
    with open(path, encoding="utf-8") as f:
        entries = json.load(f)
    return {
        int(e["layer_idx"]): {k: np.array(v) if isinstance(v, list) else v
                               for k, v in e.items()}
        for e in entries
    }


REQUIRED_FIELDS = {"dt_mean", "forget_mean"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=str, default="/gpfs/hshen/mmd",
                        help="Root directory containing {disp_name}/step_{n}/summary.json")
    parser.add_argument("--output-dir", type=str, default="/gpfs/hshen/plots")
    parser.add_argument("--disp-name", type=str, required=True,
                        help="Subdirectory name under root-dir (also used for output)")
    args = parser.parse_args()

    out_dir = os.path.join(args.output_dir, args.disp_name)
    os.makedirs(out_dir, exist_ok=True)

    # Load step 0 and step 500
    paths = {}
    for step in (0, 500):
        p = os.path.join(args.root_dir, args.disp_name, f"step_{step}", "summary.json")
        if not os.path.isfile(p):
            print(f"Missing {p}, cannot produce sensitivity plot.")
            return
        paths[step] = p

    s0 = load_summary(paths[0])
    s500 = load_summary(paths[500])

    # Layer indices present in both steps with required fields
    layers_0 = set(k for k in s0 if REQUIRED_FIELDS.issubset(s0[k].keys()))
    layers_500 = set(k for k in s500 if REQUIRED_FIELDS.issubset(s500[k].keys()))
    layer_indices = sorted(layers_0 & layers_500)
    if not layer_indices:
        print("No layers with dt_mean/forget_mean in both steps.")
        return

    # Pool all heads across all layers
    init_fg_all, d_dt_all, d_fg_all = [], [], []
    for layer_idx in layer_indices:
        fg0 = s0[layer_idx]["forget_mean"]      # [nheads]
        fg1 = s500[layer_idx]["forget_mean"]
        dt0 = s0[layer_idx]["dt_mean"]
        dt1 = s500[layer_idx]["dt_mean"]
        init_fg_all.append(fg0)
        d_dt_all.append(np.abs(dt1 - dt0))
        d_fg_all.append(np.abs(fg1 - fg0))

    init_fg = np.concatenate(init_fg_all)
    d_dt = np.concatenate(d_dt_all)
    d_fg = np.concatenate(d_fg_all)

    # Bin by initial forget gate value
    bin_edges = np.arange(0, 1.05, 0.1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    sensitivity = np.full(len(bin_centers), np.nan)
    mean_d_fg = np.full(len(bin_centers), np.nan)
    mean_d_dt = np.full(len(bin_centers), np.nan)
    counts = np.zeros(len(bin_centers), dtype=int)

    for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        mask = (init_fg >= lo) & (init_fg < hi)
        n = mask.sum()
        counts[i] = n
        if n == 0:
            continue
        mdt = d_dt[mask].mean()
        mfg = d_fg[mask].mean()
        mean_d_dt[i] = mdt
        mean_d_fg[i] = mfg
        if mdt > 1e-8:
            sensitivity[i] = mfg / mdt

    # Plot: 2 subplots — sensitivity curve + raw deltas
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    valid = ~np.isnan(sensitivity)
    ax1.bar(bin_centers[valid], sensitivity[valid], width=0.08, alpha=0.8,
            color='tab:purple', edgecolor='black', linewidth=0.5)
    for i in np.where(valid)[0]:
        ax1.text(bin_centers[i], sensitivity[i], f"n={counts[i]}",
                 ha='center', va='bottom', fontsize=7)
    ax1.set_xlabel("Initial forget gate (step 0)", fontsize=10, fontweight='bold')
    ax1.set_ylabel("|Δforget| / |Δdt|", fontsize=10, fontweight='bold')
    ax1.set_title("Forget gate sensitivity by initial gate value (step 0 → 500)",
                   fontsize=11)
    ax1.set_xlim(0, 1)

    # Raw |Δforget| and |Δdt| per bin
    w = 0.02
    ax2.bar(bin_centers[valid] - w, mean_d_fg[valid], width=2 * w, alpha=0.7,
            color='tab:orange', label='|Δforget|')
    ax2.bar(bin_centers[valid] + w, mean_d_dt[valid], width=2 * w, alpha=0.7,
            color='tab:blue', label='|Δdt|')
    ax2.set_xlabel("Initial forget gate (step 0)", fontsize=10, fontweight='bold')
    ax2.set_ylabel("Mean magnitude of change", fontsize=10, fontweight='bold')
    ax2.set_title("Raw parameter changes by initial gate bin", fontsize=11)
    ax2.set_xlim(0, 1)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "[model]sensitivity.png"), dpi=600)
    plt.close(fig)

    print(f"Saved sensitivity plot to {out_dir}/")


if __name__ == "__main__":
    main()
