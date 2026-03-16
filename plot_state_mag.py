import json
import glob
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
    out_dir = os.path.join(args.output_dir, disp_name)
    os.makedirs(out_dir, exist_ok=True)

    attn_layers = {
        "bamba2": [9, 18, 27],
        "nemotronh": [7, 18, 29, 40],
        "mamba2": [],
    }
    skip_layers = attn_layers.get(model_type, [])

    # Load per-layer JSONL files
    pattern = f"/gpfs/hshen/mmd/{disp_name}_layer*.jsonl"
    layer_files = sorted(glob.glob(pattern))
    if not layer_files:
        print(f"No files found matching {pattern}")
        return

    # ---- accumulate per-layer, per-head means ----
    # h_mag is (B, nheads, S) — take last token: h_mag[:, :, -1] -> (B, nheads)
    # dt is (B, T, H), forget is (B, T, H) — already per-token per-head means
    hmag_last_sum = {}
    hmag_last_n = {}
    dt_sum = {}
    dt_n = {}
    forget_sum = {}
    forget_n = {}

    for fpath in layer_files:
        records = read_jsonl(fpath, args.max_records)
        if not records:
            continue
        layer_idx = records[0]['layer_idx']
        if layer_idx in skip_layers:
            continue
        print(f"Loaded {len(records)} records from {fpath}")

        for rec in records:
            dt = np.array(rec['dt'])            # (B, T, H)
            forget = np.array(rec['forget'])    # (B, T, H)
            h_mag = np.array(rec['h_mag'])      # (B, nheads, S)

            H = dt.shape[-1]

            # Last-token state magnitude per head: (B, nheads) -> flatten to (*, H)
            hmag_last = h_mag[:, :, -1]  # (B, H)
            hmag_flat = hmag_last.reshape(-1, H)
            if layer_idx not in hmag_last_sum:
                hmag_last_sum[layer_idx] = np.zeros(H, dtype=np.float64)
                hmag_last_n[layer_idx] = 0
            hmag_last_sum[layer_idx] += hmag_flat.sum(axis=0)
            hmag_last_n[layer_idx] += hmag_flat.shape[0]

            # dt mean per head
            dt_flat = dt.reshape(-1, H)
            if layer_idx not in dt_sum:
                dt_sum[layer_idx] = np.zeros(H, dtype=np.float64)
                dt_n[layer_idx] = 0
            dt_sum[layer_idx] += dt_flat.sum(axis=0)
            dt_n[layer_idx] += dt_flat.shape[0]

            # forget mean per head
            f_flat = forget.reshape(-1, H)
            if layer_idx not in forget_sum:
                forget_sum[layer_idx] = np.zeros(H, dtype=np.float64)
                forget_n[layer_idx] = 0
            forget_sum[layer_idx] += f_flat.sum(axis=0)
            forget_n[layer_idx] += f_flat.shape[0]

    layer_indices = sorted(hmag_last_sum.keys())
    hmag_mean = {k: hmag_last_sum[k] / hmag_last_n[k] for k in layer_indices}
    dt_mean = {k: dt_sum[k] / dt_n[k] for k in layer_indices}
    forget_mean = {k: forget_sum[k] / forget_n[k] for k in layer_indices}

    # log h_mag for x-axis
    log_hmag = {k: np.log(hmag_mean[k] + 1e-12) for k in layer_indices}
    all_log_hmag = np.concatenate([log_hmag[k] for k in layer_indices])
    hmag_xlim = (all_log_hmag.min(), all_log_hmag.max())
    hmag_pad = (hmag_xlim[1] - hmag_xlim[0]) * 0.05
    hmag_xlim = (hmag_xlim[0] - hmag_pad, hmag_xlim[1] + hmag_pad)

    cmap = plt.cm.viridis(np.linspace(0, 1, len(layer_indices)))

    # ---- Combined plot: h_mag_last vs dt ----
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, layer_idx in enumerate(layer_indices):
        ax.scatter(log_hmag[layer_idx], dt_mean[layer_idx],
                   c=[cmap[i]], marker='.', s=10, label=f"L{layer_idx}")
    ax.set_xlim(hmag_xlim)
    ax.set_title("All layers", fontsize=11)
    ax.set_xlabel(r"log $\|h_T\|$", fontsize=11, fontweight='bold')
    ax.set_ylabel(r"$\Delta_t$", fontsize=11, fontweight='bold')
    ax.legend(fontsize=6, ncol=2, loc='best')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "[model]hmag_vs_dt.png"), dpi=600)
    plt.close(fig)
    print(f"Saved h_mag vs dt plot to {out_dir}/")

    # ---- Combined plot: h_mag_last vs forget ----
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, layer_idx in enumerate(layer_indices):
        ax.scatter(log_hmag[layer_idx], forget_mean[layer_idx],
                   c=[cmap[i]], marker='.', s=10, label=f"L{layer_idx}")
    ax.set_xlim(hmag_xlim)
    ax.set_title("All layers", fontsize=11)
    ax.set_xlabel(r"log $\|h_T\|$", fontsize=11, fontweight='bold')
    ax.set_ylabel("Average forget gate", fontsize=11, fontweight='bold')
    ax.legend(fontsize=6, ncol=2, loc='best')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "[model]hmag_vs_forget.png"), dpi=600)
    plt.close(fig)
    print(f"Saved h_mag vs forget plot to {out_dir}/")


if __name__ == "__main__":
    main()
