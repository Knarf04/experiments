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


def accumulate_means(records, skip_layers):
    """Accumulate per-layer, per-head means for dt, forget, erf."""
    erf_sum, erf_n = {}, {}
    dt_sum, dt_n = {}, {}
    forget_sum, forget_n = {}, {}

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

    # Load both files
    path1 = f"/gpfs/hshen/mmd/{name1}.jsonl"
    path2 = f"/gpfs/hshen/mmd/{name2}.jsonl"
    records1 = read_jsonl(path1, args.max_records)
    records2 = read_jsonl(path2, args.max_records)
    print(f"Loaded {len(records1)} records from {path1}")
    print(f"Loaded {len(records2)} records from {path2}")

    # Accumulate means for each
    layers1, erf1, dt1, forget1 = accumulate_means(records1, skip_layers)
    layers2, erf2, dt2, forget2 = accumulate_means(records2, skip_layers)

    # Use intersection of layers present in both
    layer_indices = sorted(set(layers1) & set(layers2))
    assert len(layer_indices) > 0, "No common layers between the two files"

    # Compute absolute differences per layer per head
    diff_erf = {k: np.abs(erf1[k] - erf2[k]) for k in layer_indices}
    diff_dt = {k: np.abs(dt1[k] - dt2[k]) for k in layer_indices}
    diff_forget = {k: np.abs(forget1[k] - forget2[k]) for k in layer_indices}

    # Log-ERF difference and unified x-axis
    log_diff_erf = {k: np.log(diff_erf[k] + 1e-12) for k in layer_indices}
    all_log_erf = np.concatenate([log_diff_erf[k] for k in layer_indices])
    erf_xlim = (all_log_erf.min(), all_log_erf.max())
    erf_pad = (erf_xlim[1] - erf_xlim[0]) * 0.05
    erf_xlim = (erf_xlim[0] - erf_pad, erf_xlim[1] + erf_pad)

    out_dir = os.path.join(args.output_dir, f"{name1}_vs_{name2}")
    os.makedirs(out_dir, exist_ok=True)
    title_prefix = f"{name1} vs {name2}"

    # ---- Combined plot: |ΔERF| vs |Δdt| ----
    cmap = plt.cm.viridis(np.linspace(0, 1, len(layer_indices)))

    fig, ax = plt.subplots(figsize=(6, 5))
    for i, layer_idx in enumerate(layer_indices):
        ax.scatter(log_diff_erf[layer_idx], diff_dt[layer_idx],
                   c=[cmap[i]], marker='.', s=10, label=f"L{layer_idx}")
    ax.set_xlim(erf_xlim)
    ax.set_title(f"{title_prefix} — all layers", fontsize=11)
    ax.set_xlabel(r"log |$\Delta$ERF|", fontsize=11, fontweight='bold')
    ax.set_ylabel(r"|$\Delta$$\Delta_t$|", fontsize=11, fontweight='bold')
    ax.legend(fontsize=6, ncol=2, loc='best')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "[model]diff_erf_vs_dt.png"), dpi=600)
    plt.close(fig)
    print(f"Saved diff ERF vs dt plot to {out_dir}/")

    # ---- Combined plot: |ΔERF| vs |Δforget| ----
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, layer_idx in enumerate(layer_indices):
        ax.scatter(log_diff_erf[layer_idx], diff_forget[layer_idx],
                   c=[cmap[i]], marker='.', s=10, label=f"L{layer_idx}")
    ax.set_xlim(erf_xlim)
    ax.set_title(f"{title_prefix} — all layers", fontsize=11)
    ax.set_xlabel(r"log |$\Delta$ERF|", fontsize=11, fontweight='bold')
    ax.set_ylabel(r"|$\Delta$forget gate|", fontsize=11, fontweight='bold')
    ax.legend(fontsize=6, ncol=2, loc='best')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "[model]diff_erf_vs_forget.png"), dpi=600)
    plt.close(fig)
    print(f"Saved diff ERF vs forget plot to {out_dir}/")


if __name__ == "__main__":
    main()
