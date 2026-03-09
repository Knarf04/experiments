import json
import os
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
    parser.add_argument("--disp-name", type=str, required=True,
                        help="Display name / subdirectory used for JSONL path and output dir")
    parser.add_argument("--model-type", type=str, default="nemotronh")
    parser.add_argument("--output-dir", type=str, default="/gpfs/hshen/plots")
    parser.add_argument("--max-records", type=int, default=10000)
    args = parser.parse_args()

    model_type = args.model_type
    disp_name = args.disp_name
    jsonl_path = f"/gpfs/hshen/mmd/{disp_name}.jsonl"
    out_dir = os.path.join(args.output_dir, disp_name)
    os.makedirs(out_dir, exist_ok=True)

    attn_layers = {
        "bamba2": [9, 18, 27],
        "nemotronh": [7, 18, 29, 40],
        "mamba2": [],
    }
    skip_layers = attn_layers.get(model_type, [])

    records = read_jsonl(jsonl_path, args.max_records)
    print(f"Loaded {len(records)} records from {jsonl_path}")

    # ---- accumulate per-layer, per-head means for dt, forget, erf ----
    erf_sum = {}
    erf_n = {}
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

        dt_flat = dt.reshape(-1, H)
        if layer_idx not in dt_sum:
            dt_sum[layer_idx] = np.zeros(H, dtype=np.float64)
            dt_n[layer_idx] = 0
        dt_sum[layer_idx] += dt_flat.sum(axis=0)
        dt_n[layer_idx] += dt_flat.shape[0]

        f_flat = forget.reshape(-1, H)
        if layer_idx not in forget_sum:
            forget_sum[layer_idx] = np.zeros(H, dtype=np.float64)
            forget_n[layer_idx] = 0
        forget_sum[layer_idx] += f_flat.sum(axis=0)
        forget_n[layer_idx] += f_flat.shape[0]

        erf_flat = erf.reshape(-1, H)
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

    # compute log-ERF per layer and find global range for unified x-axis
    log_erf = {k: np.log(erf_mean[k] + 1e-12) for k in layer_indices}
    all_log_erf = np.concatenate([log_erf[k] for k in layer_indices])
    erf_xlim = (all_log_erf.min(), all_log_erf.max())
    erf_pad = (erf_xlim[1] - erf_xlim[0]) * 0.05
    erf_xlim = (erf_xlim[0] - erf_pad, erf_xlim[1] + erf_pad)

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
    cmap = plt.cm.viridis(np.linspace(0, 1, len(layer_indices)))
    for i, layer_idx in enumerate(layer_indices):
        ax.scatter(log_erf[layer_idx], dt_mean[layer_idx], c=[cmap[i]], marker='.', s=10, label=f"L{layer_idx}")
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
        ax.scatter(log_erf[layer_idx], forget_mean[layer_idx], c=[cmap[i]], marker='.', s=10, label=f"L{layer_idx}")
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
