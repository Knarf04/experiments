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
    parser.add_argument("--summary1", type=str, required=True,
                        help="First summary JSON (produced by summarize.py --out)")
    parser.add_argument("--summary2", type=str, required=True,
                        help="Second summary JSON")
    parser.add_argument("--model-type", type=str, default="nemotronh")
    parser.add_argument("--output-dir", type=str, default="/gpfs/hshen/plots")
    parser.add_argument("--name1", type=str, default=None,
                        help="Label for summary1 (defaults to filename stem)")
    parser.add_argument("--name2", type=str, default=None,
                        help="Label for summary2 (defaults to filename stem)")
    args = parser.parse_args()

    name1 = args.name1 or os.path.splitext(os.path.basename(args.summary1))[0]
    name2 = args.name2 or os.path.splitext(os.path.basename(args.summary2))[0]

    attn_layers = {
        "bamba2": [9, 18, 27],
        "nemotronh": [7, 18, 29, 40],
        "mamba2": [],
    }
    skip_layers = attn_layers.get(args.model_type, [])

    data1 = load_summary(args.summary1)
    data2 = load_summary(args.summary2)

    layer_indices = sorted(set(data1) & set(data2) - set(skip_layers))
    assert layer_indices, "No common layers between the two summaries"

    erf1    = {k: data1[k]["erf"]         for k in layer_indices}
    erf2    = {k: data2[k]["erf"]         for k in layer_indices}
    dt1     = {k: data1[k]["dt_mean"]     for k in layer_indices}
    dt2     = {k: data2[k]["dt_mean"]     for k in layer_indices}
    forget1 = {k: data1[k]["forget_mean"] for k in layer_indices}
    forget2 = {k: data2[k]["forget_mean"] for k in layer_indices}

    diff_dt      = {k: dt2[k] - dt1[k]     for k in layer_indices}
    diff_forget  = {k: forget2[k] - forget1[k] for k in layer_indices}
    log_erf_diff = {k: np.log(erf2[k] + 1e-12) - np.log(erf1[k] + 1e-12)
                    for k in layer_indices}

    out_dir = os.path.join(args.output_dir, f"{name1}_vs_{name2}")
    os.makedirs(out_dir, exist_ok=True)
    title_prefix = f"{name2} − {name1}"

    cmap = plt.cm.viridis(np.linspace(0, 1, len(layer_indices)))

    # ---- log(ERF_2) - log(ERF_1) vs Δdt ----
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, layer_idx in enumerate(layer_indices):
        ax.scatter(log_erf_diff[layer_idx], diff_dt[layer_idx],
                   c=[cmap[i]], marker='.', s=10, label=f"L{layer_idx}")
    # ax.set_xlim((-0.8, 1.8))
    # ax.set_ylim((-0.24, 0.38))
    ax.axhline(0, color='grey', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='grey', linewidth=0.5, linestyle='--')
    ax.set_title(f"{title_prefix} — all layers", fontsize=11)
    ax.set_xlabel(r"$\log \mathrm{ERF}_2 - \log \mathrm{ERF}_1$", fontsize=11, fontweight='bold')
    ax.set_ylabel(r"$\Delta_t^{(2)} - \Delta_t^{(1)}$", fontsize=11, fontweight='bold')
    ax.legend(fontsize=6, ncol=2, loc='best')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "[model]diff_erf_vs_dt.png"), dpi=600)
    plt.close(fig)
    print(f"Saved diff ERF vs dt plot to {out_dir}/")

    # ---- log(ERF_2) - log(ERF_1) vs Δforget ----
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, layer_idx in enumerate(layer_indices):
        ax.scatter(log_erf_diff[layer_idx], diff_forget[layer_idx],
                   c=[cmap[i]], marker='.', s=10, label=f"L{layer_idx}")
    # ax.set_xlim((-0.8, 1.8))
    # ax.set_ylim((-0.22, 0.13))
    ax.axhline(0, color='grey', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='grey', linewidth=0.5, linestyle='--')
    ax.set_title(f"{title_prefix} — all layers", fontsize=11)
    ax.set_xlabel(r"$\log \mathrm{ERF}_2 - \log \mathrm{ERF}_1$", fontsize=11, fontweight='bold')
    ax.set_ylabel(r"$f^{(2)} - f^{(1)}$", fontsize=11, fontweight='bold')
    ax.legend(fontsize=6, ncol=2, loc='best')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "[model]diff_erf_vs_forget.png"), dpi=600)
    plt.close(fig)
    print(f"Saved diff ERF vs forget plot to {out_dir}/")


if __name__ == "__main__":
    main()
