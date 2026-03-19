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


def cos_sim_summary(mat: np.ndarray, interval: int = 2048) -> float:
    """Distance-weighted mean cosine similarity over the lower triangle (lag > 0).

    positions[k] = k * interval, so distance(i, j) = (i - j) * interval.
    The interval factor cancels in weighted_sum / total_weight, so we use
    lag = i - j directly as the weight.
    """
    npos = mat.shape[0]
    i_idx, j_idx = np.tril_indices(npos, k=-1)   # all pairs with i > j
    lags = (i_idx - j_idx).astype(np.float64)     # distance / interval
    sims = mat[i_idx, j_idx].astype(np.float64)
    total_weight = lags.sum()
    if total_weight == 0:
        return float(mat.mean())
    return float((sims * lags).sum() / total_weight)


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
    layer_indices = sorted(
        k for k in data if "erf" in data[k] and "state_cos_sim" in data[k]
    )
    if not layer_indices:
        print("No layers with both 'erf' and 'state_cos_sim' found in summary.")
        return

    # erf[layer_idx]: (nheads,)
    # cos_mean[layer_idx]: (nheads,) — mean over lower triangle per head
    erf_mean = {k: data[k]["erf"] for k in layer_indices}
    cos_mean = {
        k: np.array([cos_sim_summary(data[k]["state_cos_sim"][h])
                     for h in range(data[k]["state_cos_sim"].shape[0])])
        for k in layer_indices
    }

    all_erf = np.concatenate([erf_mean[k] for k in layer_indices])
    pad = (all_erf.max() - all_erf.min()) * 0.05
    erf_xlim = (all_erf.min() - pad, all_erf.max() + pad)

    cmap = plt.cm.viridis(np.linspace(0, 1, len(layer_indices)))

    # ---- Per-layer scatter plots ----
    for layer_idx in layer_indices:
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.scatter(erf_mean[layer_idx], cos_mean[layer_idx], marker='.', s=10)
        ax.set_xlim(erf_xlim)
        ax.set_title(f"Layer {layer_idx}", fontsize=11)
        ax.set_xlabel("ERF", fontsize=11, fontweight='bold')
        ax.set_ylabel("Mean state cosine similarity", fontsize=11, fontweight='bold')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"[layer{layer_idx}]erf_vs_cos_sim.png"), dpi=600)
        plt.close(fig)

    # ---- Combined all-layers scatter ----
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, layer_idx in enumerate(layer_indices):
        ax.scatter(erf_mean[layer_idx], cos_mean[layer_idx],
                   c=[cmap[i]], marker='.', s=10, label=f"L{layer_idx}")
    ax.set_xlim(erf_xlim)
    ax.set_title("All layers", fontsize=11)
    ax.set_xlabel("ERF", fontsize=11, fontweight='bold')
    ax.set_ylabel("Mean state cosine similarity", fontsize=11, fontweight='bold')
    ax.legend(fontsize=6, ncol=2, loc='best')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "[model]erf_vs_cos_sim.png"), dpi=600)
    plt.close(fig)
    print(f"Saved ERF vs cos_sim plots to {out_dir}/")


if __name__ == "__main__":
    main()
