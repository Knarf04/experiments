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


def accumulate_means(disp_name, skip_layers, max_records):
    """Load per-layer JSONL files and accumulate per-layer, per-head means for dt, forget, erf."""
    pattern = f"/gpfs/hshen/mmd/{disp_name}_layer*.jsonl"
    layer_files = sorted(glob.glob(pattern))
    if not layer_files:
        print(f"No files found matching {pattern}")
        return [], {}, {}, {}

    erf_sum, erf_n = {}, {}
    dt_sum, dt_n = {}, {}
    forget_sum, forget_n = {}, {}

    for fpath in layer_files:
        records = read_jsonl(fpath, max_records)
        if not records:
            continue
        layer_idx = records[0]['layer_idx']
        if layer_idx in skip_layers:
            continue
        print(f"Loaded {len(records)} records from {fpath}")

        for rec in records:
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

    layer_indices = sorted(erf_sum.keys())
    erf_mean = {k: erf_sum[k] / erf_n[k] for k in layer_indices}
    dt_mean = {k: dt_sum[k] / dt_n[k] for k in layer_indices}
    forget_mean = {k: forget_sum[k] / forget_n[k] for k in layer_indices}

    return layer_indices, erf_mean, dt_mean, forget_mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disp-name1", type=str, required=True,
                        help="First display name / subdirectory for JSONL path")
    parser.add_argument("--disp-name2", type=str, required=True,
                        help="Second display name / subdirectory for JSONL path")
    parser.add_argument("--model-type", type=str, default="nemotronh")
    parser.add_argument("--output-dir", type=str, default="/gpfs/hshen/plots")
    parser.add_argument("--max-records", type=int, default=10000)
    args = parser.parse_args()

    model_type = args.model_type
    name1, name2 = args.disp_name1, args.disp_name2

    attn_layers = {
        "bamba2": [9, 18, 27],
        "nemotronh": [7, 18, 29, 40],
        "mamba2": [],
    }
    skip_layers = attn_layers.get(model_type, [])

    # Accumulate means for each
    layers1, erf1, dt1, forget1 = accumulate_means(name1, skip_layers, args.max_records)
    layers2, erf2, dt2, forget2 = accumulate_means(name2, skip_layers, args.max_records)

    # Use intersection of layers present in both
    layer_indices = sorted(set(layers1) & set(layers2))
    assert len(layer_indices) > 0, "No common layers between the two files"

    # Compute signed differences per layer per head (name2 - name1)
    diff_dt = {k: dt2[k] - dt1[k] for k in layer_indices}
    diff_forget = {k: forget2[k] - forget1[k] for k in layer_indices}
    log_erf_diff = {k: np.log(erf2[k] + 1e-12) - np.log(erf1[k] + 1e-12) for k in layer_indices}

    # Unified x-axis for log-ERF difference
    all_log_erf_diff = np.concatenate([log_erf_diff[k] for k in layer_indices])
    erf_xlim = (all_log_erf_diff.min(), all_log_erf_diff.max())
    erf_pad = (erf_xlim[1] - erf_xlim[0]) * 0.05
    erf_xlim = (erf_xlim[0] - erf_pad, erf_xlim[1] + erf_pad)

    out_dir = os.path.join(args.output_dir, f"{name1}_vs_{name2}")
    os.makedirs(out_dir, exist_ok=True)
    title_prefix = f"{name2} − {name1}"

    cmap = plt.cm.viridis(np.linspace(0, 1, len(layer_indices)))

    # ---- Combined plot: log(ERF_2)-log(ERF_1) vs Δdt ----
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, layer_idx in enumerate(layer_indices):
        ax.scatter(log_erf_diff[layer_idx], diff_dt[layer_idx],
                   c=[cmap[i]], marker='.', s=10, label=f"L{layer_idx}")
    ax.set_xlim(erf_xlim)
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

    # ---- Combined plot: log(ERF_2)-log(ERF_1) vs Δforget ----
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, layer_idx in enumerate(layer_indices):
        ax.scatter(log_erf_diff[layer_idx], diff_forget[layer_idx],
                   c=[cmap[i]], marker='.', s=10, label=f"L{layer_idx}")
    ax.set_xlim(erf_xlim)
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
