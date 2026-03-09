import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def read_jsonl(filepath: str, max_records: int = 10000) -> list:
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if len(records) >= max_records:
                break
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, default="nemotronh")
    parser.add_argument("--jsonl-path", type=str, default=None,
                        help="Path to the JSONL file. Defaults to /gpfs/hshen/mmd/{model_type}.jsonl")
    parser.add_argument("--output-dir", type=str, default="/gpfs/hshen/plots")
    parser.add_argument("--max-records", type=int, default=10000)
    args = parser.parse_args()

    model_type = args.model_type
    jsonl_path = args.jsonl_path or f"/gpfs/hshen/mmd/{model_type}.jsonl"

    attn_layers = {
        "bamba2": [9, 18, 27],
        "nemotronh": [7, 18, 29, 40],
        "mamba2": [],
    }
    skip_layers = attn_layers.get(model_type, [])

    records = read_jsonl(jsonl_path, args.max_records)
    print(f"Loaded {len(records)} records from {jsonl_path}")

    # ---- accumulate per-layer, per-head means for dt, forget, erf ----
    # erf shape in record: (batch, nheads)
    # dt / forget shape in record: (batch, seqlen, nheads)

    erf_sum = {}   # layer_idx -> (H,) running sum
    erf_n = {}     # layer_idx -> int count

    dt_sum = {}
    dt_n = {}

    forget_sum = {}
    forget_n = {}

    for rec in records:
        layer_idx = rec['layer_idx']
        if layer_idx in skip_layers:
            continue

        dt = np.array(rec['dt'])          # (B, T, H)
        forget = np.array(rec['forget'])  # (B, T, H)
        erf = np.array(rec['erf'])        # (B, H)

        H = dt.shape[-1]

        # dt: average over batch and seq -> per-head
        dt_flat = dt.reshape(-1, H)  # (B*T, H)
        if layer_idx not in dt_sum:
            dt_sum[layer_idx] = np.zeros(H, dtype=np.float64)
            dt_n[layer_idx] = 0
        dt_sum[layer_idx] += dt_flat.sum(axis=0)
        dt_n[layer_idx] += dt_flat.shape[0]

        # forget: average over batch and seq -> per-head
        f_flat = forget.reshape(-1, H)
        if layer_idx not in forget_sum:
            forget_sum[layer_idx] = np.zeros(H, dtype=np.float64)
            forget_n[layer_idx] = 0
        forget_sum[layer_idx] += f_flat.sum(axis=0)
        forget_n[layer_idx] += f_flat.shape[0]

        # erf: average over batch -> per-head
        erf_flat = erf.reshape(-1, H)  # (B, H)
        if layer_idx not in erf_sum:
            erf_sum[layer_idx] = np.zeros(H, dtype=np.float64)
            erf_n[layer_idx] = 0
        erf_sum[layer_idx] += erf_flat.sum(axis=0)
        erf_n[layer_idx] += erf_flat.shape[0]

    # compute per-head means
    layer_indices = sorted(erf_sum.keys())
    erf_mean = {k: erf_sum[k] / erf_n[k] for k in layer_indices}
    dt_mean = {k: dt_sum[k] / dt_n[k] for k in layer_indices}
    forget_mean = {k: forget_sum[k] / forget_n[k] for k in layer_indices}

    n_layers = len(layer_indices)
    cmap = plt.cm.viridis(np.linspace(0, 1, n_layers))

    # ---- Plot 1: ERF vs dt, per layer + combined ----
    fig, axes = plt.subplots(1, n_layers + 1, figsize=(5 * (n_layers + 1), 4.5), squeeze=False)
    axes = axes[0]

    all_erf_vals = []
    all_dt_vals = []

    for i, layer_idx in enumerate(layer_indices):
        e = np.log(erf_mean[layer_idx] + 1e-12)
        d = dt_mean[layer_idx]
        all_erf_vals.append(e)
        all_dt_vals.append(d)

        axes[i].scatter(e, d, c=[cmap[i]], marker='.', s=10)
        axes[i].set_title(f"Layer {layer_idx}", fontsize=11)
        axes[i].set_xlabel("log-ERF", fontsize=11, fontweight='bold')
        axes[i].set_ylabel(r"$\Delta_t$", fontsize=11, fontweight='bold')

    # combined
    ax_all = axes[-1]
    for i, layer_idx in enumerate(layer_indices):
        ax_all.scatter(all_erf_vals[i], all_dt_vals[i], c=[cmap[i]], marker='.', s=10, label=f"L{layer_idx}")
    ax_all.set_title("All layers", fontsize=11)
    ax_all.set_xlabel("log-ERF", fontsize=11, fontweight='bold')
    ax_all.set_ylabel(r"$\Delta_t$", fontsize=11, fontweight='bold')
    ax_all.legend(fontsize=6, ncol=2, loc='best')

    fig.tight_layout()
    out1 = f"{args.output_dir}/{model_type}_erf_vs_dt.png"
    fig.savefig(out1, dpi=600)
    print(f"Saved {out1}")

    # ---- Plot 2: ERF vs forget, per layer + combined ----
    fig2, axes2 = plt.subplots(1, n_layers + 1, figsize=(5 * (n_layers + 1), 4.5), squeeze=False)
    axes2 = axes2[0]

    all_forget_vals = []

    for i, layer_idx in enumerate(layer_indices):
        e = np.log(erf_mean[layer_idx] + 1e-12)
        fg = forget_mean[layer_idx]
        all_forget_vals.append(fg)

        axes2[i].scatter(e, fg, c=[cmap[i]], marker='.', s=10)
        axes2[i].set_title(f"Layer {layer_idx}", fontsize=11)
        axes2[i].set_xlabel("log-ERF", fontsize=11, fontweight='bold')
        axes2[i].set_ylabel("Average forget gate", fontsize=11, fontweight='bold')

    # combined
    ax_all2 = axes2[-1]
    for i, layer_idx in enumerate(layer_indices):
        ax_all2.scatter(all_erf_vals[i], all_forget_vals[i], c=[cmap[i]], marker='.', s=10, label=f"L{layer_idx}")
    ax_all2.set_title("All layers", fontsize=11)
    ax_all2.set_xlabel("log-ERF", fontsize=11, fontweight='bold')
    ax_all2.set_ylabel("Average forget gate", fontsize=11, fontweight='bold')
    ax_all2.legend(fontsize=6, ncol=2, loc='best')

    fig2.tight_layout()
    out2 = f"{args.output_dir}/{model_type}_erf_vs_forget.png"
    fig2.savefig(out2, dpi=600)
    print(f"Saved {out2}")

    plt.show()


if __name__ == "__main__":
    main()
