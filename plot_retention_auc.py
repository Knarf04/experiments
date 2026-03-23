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


def lag_profile(mat_h, npos):
    """Compute mean cosine-sim by lag for a single (npos, npos) matrix."""
    lag_sum = np.zeros(npos, dtype=np.float64)
    lag_count = np.zeros(npos, dtype=np.int64)
    for lag in range(1, npos):
        diag_vals = np.diag(mat_h, k=-lag)
        lag_sum[lag] += diag_vals.sum()
        lag_count[lag] += len(diag_vals)
    valid = lag_count[1:] > 0
    lags = np.arange(1, npos)[valid]
    means = lag_sum[1:][valid] / lag_count[1:][valid]
    return lags, means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=str, required=True,
                        help="Path to summary JSON produced by summarize.py --out")
    parser.add_argument("--output-dir", type=str, default="/gpfs/hshen/plots")
    parser.add_argument("--disp-name", type=str, default=None,
                        help="Output subdirectory name (defaults to summary filename stem)")
    args = parser.parse_args()

    disp_name = args.disp_name or os.path.splitext(os.path.basename(args.summary))[0]
    out_dir = os.path.join(args.output_dir, disp_name)
    os.makedirs(out_dir, exist_ok=True)

    data = load_summary(args.summary)
    layer_indices = sorted(k for k in data if "state_cos_sim" in data[k])
    if not layer_indices:
        print("No state_cos_sim data found in summary.")
        return

    # Compute AUC per head across all layers
    # AUC = trapezoidal integral of the retention lag curve, normalized by max lag
    auc_all = []          # flat list of all head AUCs
    auc_by_layer = {}     # layer_idx -> [nheads] AUCs

    for layer_idx in layer_indices:
        mat = data[layer_idx]["state_cos_sim"]  # (H, npos, npos)
        nheads, npos, _ = mat.shape
        layer_aucs = []
        for h in range(nheads):
            lags, means = lag_profile(mat[h], npos)
            if len(lags) >= 2:
                auc = np.trapz(means, lags) / (lags[-1] - lags[0])
            else:
                auc = 0.0
            layer_aucs.append(auc)
        auc_by_layer[layer_idx] = np.array(layer_aucs)
        auc_all.extend(layer_aucs)

    auc_all = np.array(auc_all)

    # ---- 1) Global histogram: AUC across all heads in all layers ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(auc_all, bins=40, alpha=0.8, color='tab:blue', edgecolor='black',
            linewidth=0.5)
    ax.axvline(np.median(auc_all), color='tab:red', linestyle='--', linewidth=1.2,
               label=f"median = {np.median(auc_all):.3f}")
    ax.set_xlabel("Retention AUC (normalized)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Head count", fontsize=11, fontweight='bold')
    ax.set_title(f"{disp_name} — retention AUC distribution "
                 f"({len(auc_all)} heads, {len(layer_indices)} layers)",
                 fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "[model]retention_auc_hist.png"), dpi=600)
    plt.close(fig)

    # ---- 2) Per-layer histogram (small multiples) ----
    ncols = 4
    nrows = (len(layer_indices) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows),
                             squeeze=False)
    global_min = auc_all.min()
    global_max = auc_all.max()
    bins = np.linspace(global_min, global_max, 25)

    for i, layer_idx in enumerate(layer_indices):
        ax = axes[i // ncols][i % ncols]
        aucs = auc_by_layer[layer_idx]
        ax.hist(aucs, bins=bins, alpha=0.8, color='tab:blue', edgecolor='black',
                linewidth=0.5)
        ax.axvline(np.median(aucs), color='tab:red', linestyle='--', linewidth=1.0)
        ax.set_title(f"L{layer_idx} (med={np.median(aucs):.3f})", fontsize=9)
        ax.set_xlim(global_min, global_max)

    # hide unused subplots
    for j in range(len(layer_indices), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.supxlabel("Retention AUC", fontsize=11, fontweight='bold')
    fig.supylabel("Head count", fontsize=11, fontweight='bold')
    fig.suptitle(f"{disp_name} — per-layer retention AUC", fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.96])
    fig.savefig(os.path.join(out_dir, "[model]retention_auc_per_layer.png"), dpi=600)
    plt.close(fig)

    # ---- 3) AUC vs layer index (scatter, one dot per head) ----
    fig, ax = plt.subplots(figsize=(max(8, len(layer_indices) * 0.3 + 2), 5))
    for layer_idx in layer_indices:
        aucs = auc_by_layer[layer_idx]
        ax.scatter([layer_idx] * len(aucs), aucs, s=12, alpha=0.6, color='tab:blue')
        ax.plot(layer_idx, np.median(aucs), 'D', color='tab:red', markersize=5, zorder=5)
    ax.set_xlabel("Layer index", fontsize=11, fontweight='bold')
    ax.set_ylabel("Retention AUC (normalized)", fontsize=11, fontweight='bold')
    ax.set_title(f"{disp_name} — retention AUC per head by layer (red = median)",
                 fontsize=11)
    ax.set_ylim(bottom=0.0)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "[model]retention_auc_scatter.png"), dpi=600)
    plt.close(fig)

    print(f"Saved retention AUC plots to {out_dir}/")


if __name__ == "__main__":
    main()
