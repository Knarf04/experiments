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
    parser.add_argument("--summary", type=str, required=True, nargs="+",
                        help="One or more summary JSONs (from summarize.py --out)")
    parser.add_argument("--label", type=str, default=None, nargs="+",
                        help="Legend labels (one per --summary, defaults to filename stems)")
    parser.add_argument("--model-type", type=str, default="nemotronh")
    parser.add_argument("--output-dir", type=str, default="/gpfs/hshen/plots")
    parser.add_argument("--disp-name", type=str, default=None,
                        help="Output subdirectory name (defaults to first summary stem)")
    args = parser.parse_args()

    labels = args.label or [
        os.path.splitext(os.path.basename(p))[0] for p in args.summary
    ]
    assert len(labels) == len(args.summary), "--label count must match --summary count"

    disp_name = args.disp_name or labels[0]
    out_dir = os.path.join(args.output_dir, disp_name)
    os.makedirs(out_dir, exist_ok=True)

    attn_layers = {
        "bamba2": [9, 18, 27],
        "nemotronh": [7, 18, 29, 40],
        "mamba2": [],
    }
    skip_layers = set(attn_layers.get(args.model_type, []))

    # Load all summaries
    all_data = []
    for path, label in zip(args.summary, labels):
        data = load_summary(path)
        layer_indices = sorted(
            k for k in data
            if k not in skip_layers and "spectrum_std" in data[k]
        )
        spectrum = np.array([float(data[k]["spectrum_std"]) for k in layer_indices])
        all_data.append((label, layer_indices, spectrum))
        print(f"{label}: {len(layer_indices)} SSM layers")

    if not all_data:
        print("No spectrum_std data found.")
        return

    cmap = plt.cm.tab10(np.linspace(0, 1, len(all_data)))

    # ---- Line plot ----
    max_layers = max(len(layer_indices) for _, layer_indices, _ in all_data)
    fig, ax = plt.subplots(figsize=(max(8, max_layers * 0.3 + 2), 4.5))
    for i, (label, layer_indices, spectrum) in enumerate(all_data):
        ax.plot(layer_indices, spectrum, marker='o', markersize=4,
                linewidth=1.2, color=cmap[i], label=label)

    ax.set_xlabel("Layer index", fontsize=11, fontweight='bold')
    ax.set_ylabel("Spectrum std (forget gate)", fontsize=11, fontweight='bold')
    ax.set_title("Head diversity of forget gate across layers", fontsize=11)
    ax.set_ylim(bottom=0.0)
    if len(all_data) > 1:
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "spectrum_std_line.png"), dpi=600)
    plt.close(fig)
    print(f"Saved spectrum_std_line.png")


if __name__ == "__main__":
    main()
