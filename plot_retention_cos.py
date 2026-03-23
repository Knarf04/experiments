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

    # cos_mean[layer_idx]: (nheads, npos, npos) — already batch-averaged by summarize.py
    cos_mean = {k: data[k]["state_cos_sim"] for k in layer_indices}

    # ---- Per-layer lag profiles (per head) ----
    for layer_idx in layer_indices:
        mat = cos_mean[layer_idx]       # (H, npos, npos)
        nheads = mat.shape[0]
        npos = mat.shape[1]
        head_colors = plt.cm.tab20(np.linspace(0, 1, nheads))

        fig, ax = plt.subplots(figsize=(8, 5))
        for h in range(nheads):
            lags, means = lag_profile(mat[h], npos)
            ax.plot(lags, means, linewidth=0.9, color=head_colors[h],
                    alpha=0.7, label=f"H{h}")
        ax.set_xlabel("Lag (i − j) in sampled positions", fontsize=11, fontweight='bold')
        ax.set_ylabel("Mean cosine similarity", fontsize=11, fontweight='bold')
        ax.set_title(f"Layer {layer_idx} — retention lag profile (per head)", fontsize=11)
        ax.set_ylim(bottom=0.0)
        ax.legend(fontsize=6, ncol=4, loc='upper right')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"[layer{layer_idx}]retention_cos.png"), dpi=600)
        plt.close(fig)

    # ---- Combined all-layers lag profile ----
    fig, ax = plt.subplots(figsize=(8, 5))
    layer_colors = plt.cm.viridis(np.linspace(0, 1, len(layer_indices)))

    for idx_i, layer_idx in enumerate(layer_indices):
        mat = cos_mean[layer_idx]       # (H, npos, npos)
        mat_avg = mat.mean(axis=0)      # (npos, npos)
        npos = mat_avg.shape[0]
        lags, means = lag_profile(mat_avg, npos)
        ax.plot(lags, means, color=layer_colors[idx_i], linewidth=1.2, label=f"L{layer_idx}")

    ax.set_xlabel("Lag (i − j) in sampled positions", fontsize=11, fontweight='bold')
    ax.set_ylabel("Mean cosine similarity (head-averaged)", fontsize=11, fontweight='bold')
    ax.set_title(f"{disp_name} — all layers, retention lag profile", fontsize=11)
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=7, ncol=2, loc='upper right')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "[model]retention_cos.png"), dpi=600)
    plt.close(fig)

    print(f"Saved retention cos plots to {out_dir}/")


if __name__ == "__main__":
    main()
