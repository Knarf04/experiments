import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


STEPS = list(range(0, 6001, 500))


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
    parser.add_argument("--root-dir", type=str, default="/gpfs/hshen/mmd",
                        help="Root directory containing {disp_name}/step_{n}/summary.json")
    parser.add_argument("--output-dir", type=str, default="/gpfs/hshen/plots")
    parser.add_argument("--disp-name", type=str, required=True,
                        help="Subdirectory name under root-dir (also used for output)")
    args = parser.parse_args()

    out_dir = os.path.join(args.output_dir, args.disp_name)
    os.makedirs(out_dir, exist_ok=True)

    # Load all step summaries
    step_data = {}  # step -> {layer_idx: dict}
    for step in STEPS:
        path = os.path.join(args.root_dir, args.disp_name, f"step_{step}", "summary.json")
        if not os.path.isfile(path):
            print(f"Warning: missing {path}, skipping step {step}")
            continue
        step_data[step] = load_summary(path)

    if not step_data:
        print("No summary files found.")
        return

    available_steps = sorted(step_data.keys())

    # Collect layer indices present across all steps (intersection)
    layer_sets = [set(k for k in step_data[s] if "spectrum_std" in step_data[s][k])
                  for s in available_steps]
    layer_indices = sorted(set.intersection(*layer_sets)) if layer_sets else []
    if not layer_indices:
        print("No layers with spectrum_std found across steps.")
        return

    # Build spectrum matrix: (n_steps, n_layers)
    spectrum = np.array([
        [float(step_data[s][k]["spectrum_std"]) for k in layer_indices]
        for s in available_steps
    ])

    step_cmap = plt.cm.viridis(np.linspace(0, 1, len(available_steps)))

    # ---- Per-layer: spectrum_std vs training step ----
    for li, layer_idx in enumerate(layer_indices):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(available_steps, spectrum[:, li], marker='o', markersize=4, linewidth=1.2)
        ax.set_xlabel("Training step", fontsize=11, fontweight='bold')
        ax.set_ylabel("Spectrum std (forget gate)", fontsize=11, fontweight='bold')
        ax.set_title(f"Layer {layer_idx} — head diversity over training", fontsize=11)
        ax.set_ylim(bottom=0.0)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"[layer{layer_idx}]spectrum_std.png"), dpi=600)
        plt.close(fig)

    # ---- Model-wise: spectrum_std vs layer index, one line per step ----
    fig, ax = plt.subplots(figsize=(max(8, len(layer_indices) * 0.3 + 2), 4.5))
    for si, step in enumerate(available_steps):
        ax.plot(layer_indices, spectrum[si], marker='o', markersize=3, linewidth=1.0,
                color=step_cmap[si], label=f"step {step}")
    ax.set_xlabel("Layer index", fontsize=11, fontweight='bold')
    ax.set_ylabel("Spectrum std (forget gate)", fontsize=11, fontweight='bold')
    ax.set_title("Head diversity of forget gate across layers", fontsize=11)
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=6, ncol=2, loc='best')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "[model]spectrum_std.png"), dpi=600)
    plt.close(fig)

    print(f"Saved spectrum_std plots to {out_dir}/")


if __name__ == "__main__":
    main()
