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
        k for k in data if "erf" in data[k] and "dt_mean" in data[k]
    )
    if not layer_indices:
        print("No usable layers found in summary.")
        return

    erf_mean    = {k: data[k]["erf"]         for k in layer_indices}  # [nheads]
    dt_mean     = {k: data[k]["dt_mean"]     for k in layer_indices}  # [nheads]
    forget_mean = {k: data[k]["forget_mean"] for k in layer_indices}  # [nheads]

    log_erf = {k: np.log(erf_mean[k] + 1e-12) for k in layer_indices}
    all_log_erf = np.concatenate([log_erf[k] for k in layer_indices])
    pad = (all_log_erf.max() - all_log_erf.min()) * 0.05
    erf_xlim = (all_log_erf.min() - pad, all_log_erf.max() + pad)

    cmap = plt.cm.viridis(np.linspace(0, 1, len(layer_indices)))

    # ---- Per-layer plots: ERF vs dt ----
    for layer_idx in layer_indices:
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.scatter(log_erf[layer_idx], dt_mean[layer_idx], marker='.', s=10)
        ax.set_xlim(erf_xlim)
        ax.set_title(f"Layer {layer_idx}", fontsize=11)
        ax.set_xlabel("log-ERF", fontsize=11, fontweight='bold')
        ax.set_ylabel(r"$\Delta_t$", fontsize=11, fontweight='bold')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"[layer{layer_idx}]erf_vs_dt.png"), dpi=600)
        plt.close(fig)

    # combined
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, layer_idx in enumerate(layer_indices):
        ax.scatter(log_erf[layer_idx], dt_mean[layer_idx],
                   c=[cmap[i]], marker='.', s=10, label=f"L{layer_idx}")
    ax.set_xlim(erf_xlim)
    ax.set_title("All layers", fontsize=11)
    ax.set_xlabel("log-ERF", fontsize=11, fontweight='bold')
    ax.set_ylabel(r"$\Delta_t$", fontsize=11, fontweight='bold')
    ax.legend(fontsize=6, ncol=2, loc='best')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "[model]erf_vs_dt.png"), dpi=600)
    plt.close(fig)
    print(f"Saved ERF vs dt plots to {out_dir}/")

    # ---- Per-layer plots: ERF vs forget ----
    for layer_idx in layer_indices:
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.scatter(log_erf[layer_idx], forget_mean[layer_idx], marker='.', s=10)
        ax.set_xlim(erf_xlim)
        ax.set_title(f"Layer {layer_idx}", fontsize=11)
        ax.set_xlabel("log-ERF", fontsize=11, fontweight='bold')
        ax.set_ylabel("Average forget gate", fontsize=11, fontweight='bold')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"[layer{layer_idx}]erf_vs_forget.png"), dpi=600)
        plt.close(fig)

    # combined
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, layer_idx in enumerate(layer_indices):
        ax.scatter(log_erf[layer_idx], forget_mean[layer_idx],
                   c=[cmap[i]], marker='.', s=10, label=f"L{layer_idx}")
    ax.set_xlim(erf_xlim)
    ax.set_title("All layers", fontsize=11)
    ax.set_xlabel("log-ERF", fontsize=11, fontweight='bold')
    ax.set_ylabel("Average forget gate", fontsize=11, fontweight='bold')
    ax.legend(fontsize=6, ncol=2, loc='best')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "[model]erf_vs_forget.png"), dpi=600)
    plt.close(fig)
    print(f"Saved ERF vs forget plots to {out_dir}/")


if __name__ == "__main__":
    main()
