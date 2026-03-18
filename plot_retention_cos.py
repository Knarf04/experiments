import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages


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
    parser.add_argument("--model-type", type=str, default="bamba2")
    parser.add_argument("--output-dir", type=str, default="/gpfs/hshen/plots")
    parser.add_argument("--disp-name", type=str, default=None,
                        help="Output subdirectory name (defaults to summary filename stem)")
    args = parser.parse_args()

    disp_name = args.disp_name or os.path.splitext(os.path.basename(args.summary))[0]
    out_dir = os.path.join(args.output_dir, disp_name)
    os.makedirs(out_dir, exist_ok=True)

    attn_layers = {
        "bamba2": [9, 18, 27],
        "nemotronh": [7, 18, 29, 40],
        "mamba2": [],
    }
    skip_layers = attn_layers.get(args.model_type, [])

    data = load_summary(args.summary)
    layer_indices = sorted(
        k for k in data
        if k not in skip_layers and "state_cos_sim" in data[k]
    )
    if not layer_indices:
        print("No state_cos_sim data found in summary.")
        return

    # cos_mean[layer_idx]: (nheads, npos, npos) — already batch-averaged by summarize.py
    cos_mean = {k: data[k]["state_cos_sim"] for k in layer_indices}

    pdf_path = os.path.join(out_dir, f"{disp_name}_retention_cos.pdf")
    with PdfPages(pdf_path) as pdf:

        # ---- Per-layer heatmap grids: one page per layer, one subplot per head ----
        for layer_idx in layer_indices:
            mat = cos_mean[layer_idx]  # (H, npos, npos)
            H, npos, _ = mat.shape

            ncols = min(8, H)
            nrows = (H + ncols - 1) // ncols
            cell = min(1.8, 12.0 / ncols)
            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(cell * ncols, cell * nrows + 0.6),
                                     squeeze=False)

            im_last = None
            for h in range(H):
                r, c = divmod(h, ncols)
                ax = axes[r][c]
                im_last = ax.imshow(mat[h], origin='lower', aspect='auto',
                                    vmin=0.0, vmax=1.0, cmap='viridis',
                                    interpolation='nearest')
                ax.set_title(f"H{h}", fontsize=7)
                ax.set_xlabel("j", fontsize=6)
                ax.set_ylabel("i", fontsize=6)
                ax.tick_params(labelsize=5)
                ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4, integer=True))
                ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, integer=True))

            for h in range(H, nrows * ncols):
                r, c = divmod(h, ncols)
                axes[r][c].set_visible(False)

            if im_last is not None:
                fig.colorbar(im_last, ax=axes.ravel().tolist(),
                             label='cosine similarity', shrink=0.6, pad=0.02)
            fig.suptitle(f"Layer {layer_idx} — per-head state cosine similarity (batch-averaged)",
                         fontsize=10)
            fig.tight_layout(rect=[0, 0, 0.93, 0.96])
            pdf.savefig(fig)
            plt.close(fig)

        # ---- Per-layer lag profiles: one page per layer, one curve per head ----
        for layer_idx in layer_indices:
            mat = cos_mean[layer_idx]  # (H, npos, npos)
            H, npos, _ = mat.shape

            head_cmap = plt.cm.tab20 if H <= 20 else plt.cm.viridis
            colors = head_cmap(np.linspace(0, 1, H))

            fig, ax = plt.subplots(figsize=(8, 5))
            for h in range(H):
                lags, means = lag_profile(mat[h], npos)
                ax.plot(lags, means, color=colors[h], linewidth=0.9,
                        alpha=0.8, label=f"H{h}")

            ax.set_xlabel("Lag (i − j) in sampled positions", fontsize=11, fontweight='bold')
            ax.set_ylabel("Mean cosine similarity", fontsize=11, fontweight='bold')
            ax.set_title(f"Layer {layer_idx} — per-head retention lag profile", fontsize=11)
            ax.set_ylim(bottom=0.0)
            ax.legend(fontsize=5, ncol=max(1, H // 10), loc='upper right',
                      framealpha=0.6, handlelength=1.0)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # ---- Combined summary: head-averaged lag profile, all layers on one page ----
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
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Saved PDF to {pdf_path}")


if __name__ == "__main__":
    main()
