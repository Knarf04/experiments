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
                        help="Display name / subdirectory used for JSONL path")
    parser.add_argument("--model-type", type=str, default="nemotronh")
    parser.add_argument("--output-dir", type=str, default="/gpfs/hshen/plots")
    parser.add_argument("--max-records", type=int, default=10000)
    args = parser.parse_args()

    disp_name = args.disp_name
    model_type = args.model_type
    out_dir = os.path.join(args.output_dir, disp_name)
    os.makedirs(out_dir, exist_ok=True)

    attn_layers = {
        "bamba2": [9, 18, 27],
        "nemotronh": [7, 18, 29, 40],
        "mamba2": [],
    }
    skip_layers = attn_layers.get(model_type, [])

    steps = list(range(0, 6500, 500))  # 0, 500, 1000, ..., 6000

    # forget_by_step[step][layer_idx] = mean forget per head, shape (H,)
    forget_by_step = {}

    for step in steps:
        pattern = f"/gpfs/hshen/mmd/{disp_name}/step_{step}.jsonl"
        if not os.path.exists(pattern):
            print(f"Skipping step {step}: {pattern} not found")
            continue

        records = read_jsonl(pattern, args.max_records)
        if not records:
            print(f"Skipping step {step}: no records")
            continue

        print(f"Step {step}: loaded {len(records)} records from {pattern}")

        # Accumulate forget gates per layer
        forget_sum = {}
        forget_n = {}

        for rec in records:
            layer_idx = rec['layer_idx']
            if layer_idx in skip_layers:
                continue
            forget = np.array(rec['forget'])  # (B, T, H)
            # Average over batch and sequence length -> (H,)
            H = forget.shape[-1]
            f_flat = forget.reshape(-1, H)  # (B*T, H)
            if layer_idx not in forget_sum:
                forget_sum[layer_idx] = np.zeros(H, dtype=np.float64)
                forget_n[layer_idx] = 0
            forget_sum[layer_idx] += f_flat.sum(axis=0)
            forget_n[layer_idx] += f_flat.shape[0]

        forget_mean = {k: forget_sum[k] / forget_n[k] for k in sorted(forget_sum.keys())}
        forget_by_step[step] = forget_mean

    available_steps = sorted(forget_by_step.keys())
    if len(available_steps) < 2:
        print("Not enough steps with data to plot evolution.")
        return

    # Get all layer indices present across steps
    all_layers = sorted(set(l for step in available_steps for l in forget_by_step[step].keys()))
    H = len(next(iter(forget_by_step[available_steps[0]].values())))

    # ---- Plot 1: Per-layer individual plots, each head as a separate line ----
    head_cmap = plt.cm.tab20(np.linspace(0, 1, H))
    for layer_idx in all_layers:
        fig, ax = plt.subplots(figsize=(8, 5))
        for h in range(H):
            vals = []
            step_list = []
            for step in available_steps:
                if layer_idx in forget_by_step[step]:
                    vals.append(forget_by_step[step][layer_idx][h])
                    step_list.append(step)
            ax.plot(step_list, vals, marker='.', markersize=3, linewidth=0.8,
                    color=head_cmap[h % len(head_cmap)])
        ax.set_xlabel("Training step", fontsize=11, fontweight='bold')
        ax.set_ylabel("Forget gate", fontsize=11, fontweight='bold')
        ax.set_title(f"{disp_name} Layer {layer_idx}: Per-head forget gate", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"forget_gate_layer{layer_idx}.png"), dpi=600)
        plt.close(fig)
    print(f"Saved per-layer forget gate plots")

    # ---- Plot 2: Std of forget gates across heads, one line per layer ----
    fig, ax = plt.subplots(figsize=(8, 5))
    layer_cmap = plt.cm.viridis(np.linspace(0, 1, len(all_layers)))
    for i, layer_idx in enumerate(all_layers):
        std_vals = []
        step_list = []
        for step in available_steps:
            if layer_idx in forget_by_step[step]:
                std_vals.append(forget_by_step[step][layer_idx].std())
                step_list.append(step)
        ax.plot(step_list, std_vals, marker='.', markersize=3, linewidth=0.8,
                color=layer_cmap[i], label=f"L{layer_idx}")
    ax.set_xlabel("Training step", fontsize=11, fontweight='bold')
    ax.set_ylabel("Std of forget gate across heads", fontsize=11, fontweight='bold')
    ax.set_title(f"{disp_name}: Forget gate std evolution", fontsize=11)
    ax.legend(fontsize=6, ncol=4, loc='best')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "forget_gate_std_evolution.png"), dpi=600)
    plt.close(fig)
    print(f"Saved forget_gate_std_evolution.png")

    print(f"All plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
