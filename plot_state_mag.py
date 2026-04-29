"""
plot_state_mag.py — heatmaps and per-layer trajectories for state_mag .npz files.

Consumes the .npz produced by `summarize.py --state-mag-out`. Supports one or
more checkpoints in a single invocation: with multiple inputs, an extra ratio
heatmap (each non-baseline / baseline) is rendered for cross-checkpoint diff.

Usage:
    python plot_state_mag.py base.npz \
        --out plots/8k_base --train-len 8192

    python plot_state_mag.py 8k_base.npz 128k_instruct.npz dt_scaled.npz \
        --out plots/cmp --train-len 8192 --names "8K base" "128K Instruct" "dt-scaled"

The first .npz is treated as the baseline for ratio comparisons.

Outputs (per checkpoint):
    heatmap.png        rows = (layer, head) flattened, cols = bin index
    layer_lines/L{n}.png   per-layer line plot, one curve per head
    drift_hist.png     histogram of per-head drift_ratio across all layers

Outputs (cross-checkpoint, when ≥2 inputs):
    ratio_heatmap_{name}.png   log2(mean_other / mean_baseline) per (layer, head, bin)
"""

import argparse
import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm


REQUIRED_KEYS = (
    "layer_idx", "positions", "seq_len", "state_bin_size",
    "mean", "std", "sem", "median", "p25", "p75",
    "drift_ratio", "log_slope",
)


def load_npz(path: str) -> dict:
    z = np.load(path)
    missing = [k for k in REQUIRED_KEYS if k not in z.files]
    if missing:
        raise ValueError(f"{path}: missing keys {missing}")
    return {k: z[k] for k in z.files}


def _ckpt_name(path: str, override: Optional[str]) -> str:
    return override or os.path.splitext(os.path.basename(path))[0]


def _bin_for_train_len(train_len: int, bin_size: int) -> float:
    """X-coordinate (in bin units) of the trained-length boundary."""
    return train_len / bin_size - 0.5


def plot_heatmap(data: dict, name: str, out_dir: str, train_len: Optional[int]) -> None:
    mean = data["mean"]            # [L, H, n_bins]
    L, H, n_bins = mean.shape
    flat = mean.reshape(L * H, n_bins)
    bin_size = int(data["state_bin_size"])

    fig, ax = plt.subplots(figsize=(max(8, n_bins * 0.18), max(6, L * H * 0.025)))
    vmin = max(float(flat[flat > 0].min()) if (flat > 0).any() else 1e-6, 1e-6)
    vmax = float(flat.max()) if flat.max() > vmin else vmin * 10
    im = ax.imshow(flat, aspect="auto", origin="lower",
                   norm=LogNorm(vmin=vmin, vmax=vmax), cmap="viridis")
    fig.colorbar(im, ax=ax, label="‖h‖_F (mean)")

    layer_centers = np.arange(L) * H + H / 2 - 0.5
    ax.set_yticks(layer_centers)
    ax.set_yticklabels([str(int(li)) for li in data["layer_idx"]])
    ax.set_ylabel("layer (× H heads each)")
    ax.set_xlabel(f"position bin (bin_size = {bin_size} tokens)")
    ax.set_title(f"{name}: per-head state magnitude vs. position")

    for li in range(1, L):
        ax.axhline(li * H - 0.5, color="white", linewidth=0.3, alpha=0.4)
    if train_len is not None:
        ax.axvline(_bin_for_train_len(train_len, bin_size),
                   color="red", linestyle="--", linewidth=1.0,
                   label=f"train_len = {train_len}")
        ax.legend(loc="upper right", fontsize="small")

    fig.tight_layout()
    path = os.path.join(out_dir, "heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


def plot_layer_lines(data: dict, name: str, out_dir: str, train_len: Optional[int]) -> None:
    mean = data["mean"]
    sem = data["sem"]
    positions = data["positions"]
    layer_idx = data["layer_idx"]
    L, H, n_bins = mean.shape
    sub_dir = os.path.join(out_dir, "layer_lines")
    os.makedirs(sub_dir, exist_ok=True)

    for li in range(L):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for h in range(H):
            ax.plot(positions, mean[li, h], linewidth=0.8, alpha=0.7)
            ax.fill_between(
                positions,
                mean[li, h] - sem[li, h],
                mean[li, h] + sem[li, h],
                alpha=0.08, linewidth=0,
            )
        ax.set_xlabel("position (last token of bin)")
        ax.set_ylabel("‖h‖_F")
        ax.set_yscale("log")
        ax.set_title(f"{name} — layer {int(layer_idx[li])}: per-head magnitude")
        if train_len is not None:
            ax.axvline(train_len, color="red", linestyle="--", linewidth=1.0,
                       label=f"train_len = {train_len}")
            ax.legend(loc="upper left", fontsize="small")
        fig.tight_layout()
        path = os.path.join(sub_dir, f"L{int(layer_idx[li]):03d}.png")
        fig.savefig(path, dpi=140)
        plt.close(fig)
    print(f"  wrote {L} per-layer plots under {sub_dir}/")


def plot_drift_hist(data: dict, name: str, out_dir: str) -> None:
    drift = data["drift_ratio"].reshape(-1)  # [L*H]
    log_slope = data["log_slope"].reshape(-1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(np.log10(np.maximum(drift, 1e-6)), bins=60, color="steelblue", alpha=0.85)
    axes[0].set_xlabel("log10(drift_ratio = max(mean) / min(mean))")
    axes[0].set_ylabel("# (layer, head) pairs")
    axes[0].set_title(f"{name}: drift_ratio distribution")
    axes[0].axvline(0, color="black", linewidth=0.5)

    axes[1].hist(log_slope, bins=60, color="darkorange", alpha=0.85)
    axes[1].set_xlabel("log_slope (slope of log(mean) vs. position)")
    axes[1].set_ylabel("# (layer, head) pairs")
    axes[1].set_title(f"{name}: log_slope distribution")
    axes[1].axvline(0, color="black", linewidth=0.5)

    fig.tight_layout()
    path = os.path.join(out_dir, "drift_hist.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


def plot_ratio_heatmap(
    base: dict, other: dict, base_name: str, other_name: str,
    out_dir: str, train_len: Optional[int]
) -> None:
    if base["mean"].shape != other["mean"].shape:
        raise ValueError(
            f"shape mismatch for ratio: base {base['mean'].shape} vs other {other['mean'].shape}"
        )
    eps = 1e-12
    ratio = np.log2((other["mean"] + eps) / (base["mean"] + eps))  # [L, H, n_bins]
    L, H, n_bins = ratio.shape
    flat = ratio.reshape(L * H, n_bins)
    bin_size = int(base["state_bin_size"])

    vmax = max(1.0, float(np.abs(flat).max()))
    fig, ax = plt.subplots(figsize=(max(8, n_bins * 0.18), max(6, L * H * 0.025)))
    im = ax.imshow(flat, aspect="auto", origin="lower",
                   norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
                   cmap="RdBu_r")
    fig.colorbar(im, ax=ax, label=f"log2( {other_name} / {base_name} )")

    layer_centers = np.arange(L) * H + H / 2 - 0.5
    ax.set_yticks(layer_centers)
    ax.set_yticklabels([str(int(li)) for li in base["layer_idx"]])
    ax.set_ylabel("layer (× H heads each)")
    ax.set_xlabel(f"position bin (bin_size = {bin_size} tokens)")
    ax.set_title(f"{other_name} vs {base_name} — log2 magnitude ratio")
    for li in range(1, L):
        ax.axhline(li * H - 0.5, color="black", linewidth=0.2, alpha=0.3)
    if train_len is not None:
        ax.axvline(_bin_for_train_len(train_len, bin_size),
                   color="black", linestyle="--", linewidth=1.0,
                   label=f"train_len = {train_len}")
        ax.legend(loc="upper right", fontsize="small")

    fig.tight_layout()
    safe = other_name.replace(" ", "_").replace("/", "_")
    path = os.path.join(out_dir, f"ratio_heatmap_{safe}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot state-magnitude heatmaps from summarize.py --state-mag-out npz files.",
    )
    parser.add_argument("inputs", nargs="+",
                        help="One or more state_mag .npz files. The first is the baseline for ratio plots.")
    parser.add_argument("--out", default="/gpfs/hshen/plots/state_mag",
                        help="Output directory root")
    parser.add_argument("--names", nargs="+", default=None,
                        help="Display names for inputs (defaults to filename stems)")
    parser.add_argument("--train-len", type=int, default=None,
                        help="Trained context length; drawn as a red dashed line")
    args = parser.parse_args()

    if args.names is not None and len(args.names) != len(args.inputs):
        parser.error("--names must have one entry per input")

    os.makedirs(args.out, exist_ok=True)

    loaded = []
    for i, path in enumerate(args.inputs):
        name = args.names[i] if args.names else _ckpt_name(path, None)
        data = load_npz(path)
        loaded.append((name, data))
        sub_dir = os.path.join(args.out, name.replace(" ", "_").replace("/", "_"))
        os.makedirs(sub_dir, exist_ok=True)
        print(f"[{name}]  L={data['mean'].shape[0]}  H={data['mean'].shape[1]}  "
              f"bins={data['mean'].shape[2]}  seq_len={int(data['seq_len'])}  "
              f"bin_size={int(data['state_bin_size'])}")

        plot_heatmap(data, name, sub_dir, args.train_len)
        plot_layer_lines(data, name, sub_dir, args.train_len)
        plot_drift_hist(data, name, sub_dir)

    if len(loaded) >= 2:
        base_name, base = loaded[0]
        cmp_dir = os.path.join(args.out, "_compare")
        os.makedirs(cmp_dir, exist_ok=True)
        for name, data in loaded[1:]:
            try:
                plot_ratio_heatmap(base, data, base_name, name, cmp_dir, args.train_len)
            except ValueError as e:
                print(f"  [skip ratio] {name}: {e}")


if __name__ == "__main__":
    main()
