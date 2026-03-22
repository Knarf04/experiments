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


METRICS = [
    ("mean_dt",     "dt_mean",     np.mean, "Mean dt"),
    ("std_dt",      "dt_mean",     np.std,  "Std dt (across heads)"),
    ("mean_forget", "forget_mean", np.mean, "Mean forget"),
    ("std_forget",  "forget_mean", np.std,  "Std forget (across heads)"),
]

REQUIRED_FIELDS = {"dt_mean", "forget_mean"}


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
    layer_sets = [
        set(k for k in step_data[s]
            if REQUIRED_FIELDS.issubset(step_data[s][k].keys()))
        for s in available_steps
    ]
    layer_indices = sorted(set.intersection(*layer_sets)) if layer_sets else []
    if not layer_indices:
        print("No layers with dt_mean/forget_mean found across steps.")
        return

    # Build metric matrices: {metric_name: ndarray (n_steps, n_layers)}
    series = {}
    for metric_name, field, agg_fn, _ in METRICS:
        series[metric_name] = np.array([
            [float(agg_fn(step_data[s][k][field])) for k in layer_indices]
            for s in available_steps
        ])

    layer_cmap = plt.cm.viridis(np.linspace(0, 1, len(layer_indices)))

    # ================================================================
    # 1) Model-level: one figure with 4 subplots (2x2)
    #    x = step, y = metric, one line per layer
    # ================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for ax, (metric_name, _, _, label) in zip(axes.flat, METRICS):
        mat = series[metric_name]  # (n_steps, n_layers)
        for li, layer_idx in enumerate(layer_indices):
            ax.plot(available_steps, mat[:, li], marker='o', markersize=3,
                    linewidth=1.0, color=layer_cmap[li], label=f"L{layer_idx}")
        ax.set_ylabel(label, fontsize=10, fontweight='bold')
        ax.set_ylim(bottom=0.0)
        ax.set_title(label, fontsize=11)
    for ax in axes[1]:
        ax.set_xlabel("Training step", fontsize=10, fontweight='bold')
    # shared legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=6, ncol=min(len(layer_indices), 6),
               loc='lower center', bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("SSM parameter evolution over training", fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(os.path.join(out_dir, "[model]dt_forget_series.png"), dpi=600)
    plt.close(fig)

    # ================================================================
    # 2) Layer-level: per layer, 2 subplots (dt / forget)
    #    x = step, line = mean across heads, shaded band = +/- std across heads
    # ================================================================
    mean_dt = series["mean_dt"]
    std_dt = series["std_dt"]
    mean_forget = series["mean_forget"]
    std_forget = series["std_forget"]

    for li, layer_idx in enumerate(layer_indices):
        fig, (ax_dt, ax_fg) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

        # dt subplot
        mu = mean_dt[:, li]
        sd = std_dt[:, li]
        ax_dt.plot(available_steps, mu, marker='o', markersize=4, linewidth=1.2,
                   color='tab:blue', label='mean dt')
        ax_dt.fill_between(available_steps, mu - sd, mu + sd,
                           alpha=0.25, color='tab:blue', label='\u00b1 1 std')
        ax_dt.set_ylabel("dt magnitude", fontsize=10, fontweight='bold')
        ax_dt.set_title(f"Layer {layer_idx} — dt over training", fontsize=11)
        ax_dt.set_ylim(bottom=0.0)
        ax_dt.legend(fontsize=8)

        # forget subplot
        mu = mean_forget[:, li]
        sd = std_forget[:, li]
        ax_fg.plot(available_steps, mu, marker='o', markersize=4, linewidth=1.2,
                   color='tab:orange', label='mean forget')
        ax_fg.fill_between(available_steps, mu - sd, mu + sd,
                           alpha=0.25, color='tab:orange', label='\u00b1 1 std')
        ax_fg.set_xlabel("Training step", fontsize=10, fontweight='bold')
        ax_fg.set_ylabel("forget magnitude", fontsize=10, fontweight='bold')
        ax_fg.set_title(f"Layer {layer_idx} — forget gate over training", fontsize=11)
        ax_fg.set_ylim(bottom=0.0)
        ax_fg.legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"[layer{layer_idx}]dt_forget_series.png"), dpi=600)
        plt.close(fig)

    print(f"Saved dt/forget series plots to {out_dir}/")


if __name__ == "__main__":
    main()
