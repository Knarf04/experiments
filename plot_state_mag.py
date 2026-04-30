"""
plot_state_mag.py — heatmaps and per-layer trajectories for state_mag .npz files.

Consumes the .npz produced by `summarize.py --state-mag-out`. Supports one or
more checkpoints in a single invocation: with multiple inputs, an extra ratio
heatmap (each non-baseline / baseline) is rendered for cross-checkpoint diff.

Usage:
    python plot_state_mag.py base.npz \
        --out plots/8k_base --train-len 8192 \
        --model-config /path/to/config.json

    python plot_state_mag.py 8k_base.npz 128k_instruct.npz dt_scaled.npz \
        --out plots/cmp --train-len 8192 \
        --names "8K base" "128K Instruct" "dt-scaled" \
        --normalize all \
        --model-config /path/to/config.json

The first .npz is treated as the baseline for ratio comparisons.

Outputs (per checkpoint):
    heatmap.png         (layer × head) rows × bin cols, log absolute (shared
                        vmin/vmax across checkpoints).
    heatmap_row.png     (layer × head) rows × bin cols, log2 ratio of each row
                        vs its own bin-0 (diverging RdBu_r).
    heatmap_layer.png   (layer × head) rows × bin cols, log2 ratio vs each row's
                        layer-mean bin-0 (diverging RdBu_r).
    layer_heatmap.png   layer × bin (head dim collapsed via median by default),
                        row-normalized vs bin-0 with diverging colormap.
    layer_lines/L{n}.png   per-layer line plot, one curve per head, log-y.
    drift_hist.png      histogram of per-head drift_ratio + log_slope.

Outputs (cross-checkpoint, when ≥2 inputs):
    ratio_heatmap_{name}.png   log2(mean_other / mean_baseline) per (layer, head, bin).

Attention-layer markers:
    Pass --model-config to a HuggingFace config.json. We auto-resolve attention
    layer indices from `hybrid_override_pattern` (Nemotron-H) or
    `attn_layer_indices` (Bamba). Mamba layers immediately following an
    attention layer get crimson, bold y-tick labels in every heatmap. Use
    --attn-layer-idx CSV only as an ad-hoc fallback.
"""

import argparse
import json
import os
from typing import List, Optional, Sequence, Set, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm


REQUIRED_KEYS = (
    "layer_idx", "positions", "seq_len", "state_bin_size",
    "mean", "std", "sem", "median", "p25", "p75",
    "drift_ratio", "log_slope",
)

EPS = 1e-12
RATIO_OCTAVE_CAP = 6.0   # cap log2-ratio color range at ±6 (= 64×) so outliers don't wash out


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


def _log_range_across(loaded: Sequence[Tuple[str, dict]]) -> Tuple[float, float]:
    """Shared positive-value vmin/vmax across all checkpoints' `mean` arrays."""
    mins, maxs = [], []
    for _, data in loaded:
        m = data["mean"]
        pos = m[m > 0]
        if pos.size:
            mins.append(float(pos.min()))
            maxs.append(float(m.max()))
    if not mins:
        return 1e-6, 1.0
    vmin = max(min(mins), 1e-12)
    vmax = max(max(maxs), vmin * 10)
    return vmin, vmax


def resolve_attn_set(model_config_path: Optional[str],
                     csv_fallback: str) -> Set[int]:
    """Return the set of attention layer indices in the model.

    Resolution order:
      1. --model-config <config.json> with hybrid_override_pattern → indices of '*'.
      2. --model-config <config.json> with attn_layer_indices → set(that list).
      3. --attn-layer-idx CSV.
      4. Empty set.

    Mamba2-only configs (no attention) yield an empty set without error.
    """
    if model_config_path:
        with open(model_config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        pat = cfg.get("hybrid_override_pattern")
        if isinstance(pat, str) and pat:
            attn = {i for i, c in enumerate(pat) if c == "*"}
            print(f"  resolved {len(attn)} attention layer(s) from hybrid_override_pattern: {sorted(attn)}")
            return attn
        idx_list = cfg.get("attn_layer_indices")
        if isinstance(idx_list, list):
            attn = {int(i) for i in idx_list}
            print(f"  resolved {len(attn)} attention layer(s) from attn_layer_indices: {sorted(attn)}")
            return attn
        print("  no attention indices in config (hybrid_override_pattern / attn_layer_indices both absent)")
        return set()
    if csv_fallback.strip():
        attn = {int(x) for x in csv_fallback.split(",") if x.strip()}
        print(f"  using --attn-layer-idx fallback: {sorted(attn)}")
        return attn
    return set()


def _attn_follow_rows(layer_idx_arr: np.ndarray, attn_set: Set[int]) -> List[int]:
    """Indices into layer_idx_arr whose layer immediately follows an attention layer."""
    if not attn_set:
        return []
    return [i for i, li in enumerate(layer_idx_arr) if int(li) - 1 in attn_set]


def _color_attn_ticks(ax, layer_idx_arr: np.ndarray, attn_set: Set[int]) -> bool:
    """Recolor y-tick labels for layers that follow an attention layer.
    Returns True if any rows were highlighted (so caller can add the legend entry).
    """
    follow = _attn_follow_rows(layer_idx_arr, attn_set)
    if not follow:
        return False
    labels = ax.get_yticklabels()
    for i in follow:
        if 0 <= i < len(labels):
            labels[i].set_color("crimson")
            labels[i].set_fontweight("bold")
    ax.plot([], [], color="crimson", marker="s", linestyle="",
            label="layer follows attn")
    return True


def _maybe_thin_ticks(positions: np.ndarray, labels: List[str], thresh: int = 40
                      ) -> Tuple[np.ndarray, List[str]]:
    """Drop every other tick when there are more than `thresh` rows."""
    if len(positions) > thresh:
        return positions[::2], labels[::2]
    return positions, labels


def plot_heatmap(
    data: dict, name: str, out_dir: str, train_len: Optional[int],
    vmin: float, vmax: float, normalize: str, attn_set: Set[int]
) -> None:
    mean = data["mean"]            # [L, H, n_bins]
    L, H, n_bins = mean.shape
    bin_size = int(data["state_bin_size"])

    if normalize == "none":
        flat = mean.reshape(L * H, n_bins)
        cbar_label = "‖h‖_F (mean)"
        cmap = "viridis"
        norm = LogNorm(vmin=vmin, vmax=vmax)
    elif normalize == "row":
        baseline = mean[:, :, :1] + EPS                                    # [L, H, 1]
        ratio = (mean + EPS) / baseline
        flat = np.log2(ratio).reshape(L * H, n_bins)
        cbar_label = "log2( ‖h‖_F(t) / ‖h‖_F(t=0) ) per row"
        cmap = "RdBu_r"
        absmax = min(max(1.0, float(np.abs(flat).max())), RATIO_OCTAVE_CAP)
        norm = TwoSlopeNorm(vmin=-absmax, vcenter=0.0, vmax=absmax)
    elif normalize == "layer":
        baseline = mean[:, :, 0].mean(axis=1, keepdims=True)[..., None] + EPS  # [L, 1, 1]
        ratio = (mean + EPS) / baseline
        flat = np.log2(ratio).reshape(L * H, n_bins)
        cbar_label = "log2( ‖h‖_F(t) / mean_h(‖h‖_F(t=0)) ) per layer"
        cmap = "RdBu_r"
        absmax = min(max(1.0, float(np.abs(flat).max())), RATIO_OCTAVE_CAP)
        norm = TwoSlopeNorm(vmin=-absmax, vcenter=0.0, vmax=absmax)
    else:
        raise ValueError(f"unknown normalize mode: {normalize!r}")

    fig, ax = plt.subplots(figsize=(max(10, n_bins * 0.20), max(6, L * H * 0.025)))
    im = ax.imshow(flat, aspect="auto", origin="lower", norm=norm, cmap=cmap)
    fig.colorbar(im, ax=ax, label=cbar_label)

    layer_centers = np.arange(L) * H + H / 2 - 0.5
    layer_labels = [str(int(li)) for li in data["layer_idx"]]
    ticks, labels = _maybe_thin_ticks(layer_centers, layer_labels, thresh=40)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_ylabel("layer (× H heads each)")
    ax.set_xlabel(f"position bin (bin_size = {bin_size} tokens)")
    ax.set_title(f"{name}: per-head state magnitude vs. position [{normalize}]")

    sep_color = "white" if normalize == "none" else "black"
    for li in range(1, L):
        ax.axhline(li * H - 0.5, color=sep_color, linewidth=0.3, alpha=0.4)

    legend_needed = False
    if train_len is not None:
        ax.axvline(_bin_for_train_len(train_len, bin_size),
                   color="red" if normalize == "none" else "black",
                   linestyle="--", linewidth=1.0,
                   label=f"train_len = {train_len}")
        legend_needed = True
    # Note: tick recolor must run AFTER the y-tick labels are set.
    if _color_attn_ticks(ax, data["layer_idx"], attn_set):
        legend_needed = True
    if legend_needed:
        ax.legend(loc="upper right", fontsize="small")

    fig.tight_layout()
    suffix = "" if normalize == "none" else f"_{normalize}"
    path = os.path.join(out_dir, f"heatmap{suffix}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


def plot_layer_heatmap(
    data: dict, name: str, out_dir: str, train_len: Optional[int],
    attn_set: Set[int], head_reduce: str = "median"
) -> None:
    """Aggregate head dim, row-normalize vs bin-0 per layer, render [L, n_bins]."""
    mean = data["mean"]                                  # [L, H, n_bins]
    L, H, n_bins = mean.shape
    bin_size = int(data["state_bin_size"])

    if head_reduce == "median":
        agg = np.median(mean, axis=1)
    elif head_reduce == "max":
        agg = mean.max(axis=1)
    elif head_reduce == "mean":
        agg = mean.mean(axis=1)
    else:
        raise ValueError(f"unknown head_reduce: {head_reduce!r}")

    baseline = agg[:, :1] + EPS
    log2_ratio = np.log2((agg + EPS) / baseline)
    absmax = min(max(1.0, float(np.abs(log2_ratio).max())), RATIO_OCTAVE_CAP)

    fig, ax = plt.subplots(figsize=(max(10, n_bins * 0.20), max(5, L * 0.22)))
    im = ax.imshow(
        log2_ratio, aspect="auto", origin="lower",
        norm=TwoSlopeNorm(vmin=-absmax, vcenter=0.0, vmax=absmax),
        cmap="RdBu_r",
    )
    fig.colorbar(im, ax=ax,
                 label=f"log2( {head_reduce}_h(‖h‖_F)(t) / {head_reduce}_h(‖h‖_F)(t=0) )")

    layer_pos = np.arange(L)
    layer_labels = [str(int(li)) for li in data["layer_idx"]]
    ax.set_yticks(layer_pos)
    ax.set_yticklabels(layer_labels)
    ax.set_ylabel("layer")
    ax.set_xlabel(f"position bin (bin_size = {bin_size} tokens)")
    ax.set_title(f"{name}: per-layer drift ({head_reduce} over heads, vs bin-0)")

    legend_needed = False
    if train_len is not None:
        ax.axvline(_bin_for_train_len(train_len, bin_size),
                   color="black", linestyle="--", linewidth=1.0,
                   label=f"train_len = {train_len}")
        legend_needed = True
    if _color_attn_ticks(ax, data["layer_idx"], attn_set):
        legend_needed = True
    if legend_needed:
        ax.legend(loc="upper right", fontsize="small")

    fig.tight_layout()
    path = os.path.join(out_dir, "layer_heatmap.png")
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
    out_dir: str, train_len: Optional[int], attn_set: Set[int]
) -> None:
    if base["mean"].shape != other["mean"].shape:
        raise ValueError(
            f"shape mismatch for ratio: base {base['mean'].shape} vs other {other['mean'].shape}"
        )
    ratio = np.log2((other["mean"] + EPS) / (base["mean"] + EPS))  # [L, H, n_bins]
    L, H, n_bins = ratio.shape
    flat = ratio.reshape(L * H, n_bins)
    bin_size = int(base["state_bin_size"])

    vmax = min(max(1.0, float(np.abs(flat).max())), RATIO_OCTAVE_CAP)
    fig, ax = plt.subplots(figsize=(max(10, n_bins * 0.20), max(6, L * H * 0.025)))
    im = ax.imshow(flat, aspect="auto", origin="lower",
                   norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
                   cmap="RdBu_r")
    fig.colorbar(im, ax=ax, label=f"log2( {other_name} / {base_name} )")

    layer_centers = np.arange(L) * H + H / 2 - 0.5
    layer_labels = [str(int(li)) for li in base["layer_idx"]]
    ticks, labels = _maybe_thin_ticks(layer_centers, layer_labels, thresh=40)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_ylabel("layer (× H heads each)")
    ax.set_xlabel(f"position bin (bin_size = {bin_size} tokens)")
    ax.set_title(f"{other_name} vs {base_name} — log2 magnitude ratio")
    for li in range(1, L):
        ax.axhline(li * H - 0.5, color="black", linewidth=0.2, alpha=0.3)

    legend_needed = False
    if train_len is not None:
        ax.axvline(_bin_for_train_len(train_len, bin_size),
                   color="black", linestyle="--", linewidth=1.0,
                   label=f"train_len = {train_len}")
        legend_needed = True
    if _color_attn_ticks(ax, base["layer_idx"], attn_set):
        legend_needed = True
    if legend_needed:
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
                        help="Trained context length; drawn as a dashed line on every heatmap")
    parser.add_argument("--normalize", choices=("none", "row", "layer", "all"),
                        default="all",
                        help="Per-(layer, head) heatmap normalization. 'none' = log absolute, "
                             "shared color scale across checkpoints. 'row' = each row vs its own "
                             "bin-0. 'layer' = each row vs its layer-mean bin-0. "
                             "'all' (default) = produce all three.")
    parser.add_argument("--head-reduce", choices=("median", "mean", "max"),
                        default="median",
                        help="How to collapse the head dim for layer_heatmap.png (default: median).")
    parser.add_argument("--model-config", default=None,
                        help="Path to the model's HuggingFace config.json. Used to auto-resolve "
                             "attention layer indices via hybrid_override_pattern (Nemotron-H) or "
                             "attn_layer_indices (Bamba). Mamba layers immediately following an "
                             "attention layer get crimson, bold y-tick labels in every heatmap.")
    parser.add_argument("--attn-layer-idx", type=str, default="",
                        help="Comma-separated attention layer indices. Ad-hoc fallback; prefer "
                             "--model-config so the indices are derived from the actual model.")
    args = parser.parse_args()

    if args.names is not None and len(args.names) != len(args.inputs):
        parser.error("--names must have one entry per input")

    print("Resolving attention layer set...")
    attn_set = resolve_attn_set(args.model_config, args.attn_layer_idx)

    os.makedirs(args.out, exist_ok=True)

    loaded: List[Tuple[str, dict]] = []
    for i, path in enumerate(args.inputs):
        name = args.names[i] if args.names else _ckpt_name(path, None)
        data = load_npz(path)
        loaded.append((name, data))
        print(f"[{name}]  L={data['mean'].shape[0]}  H={data['mean'].shape[1]}  "
              f"bins={data['mean'].shape[2]}  seq_len={int(data['seq_len'])}  "
              f"bin_size={int(data['state_bin_size'])}")

    vmin_shared, vmax_shared = _log_range_across(loaded)
    print(f"shared log color range: vmin={vmin_shared:.3e}  vmax={vmax_shared:.3e}")

    modes = ("none", "row", "layer") if args.normalize == "all" else (args.normalize,)

    for name, data in loaded:
        sub_dir = os.path.join(args.out, name.replace(" ", "_").replace("/", "_"))
        os.makedirs(sub_dir, exist_ok=True)

        for mode in modes:
            plot_heatmap(data, name, sub_dir, args.train_len,
                         vmin_shared, vmax_shared, mode, attn_set)
        plot_layer_heatmap(data, name, sub_dir, args.train_len, attn_set,
                           head_reduce=args.head_reduce)
        plot_layer_lines(data, name, sub_dir, args.train_len)
        plot_drift_hist(data, name, sub_dir)

    if len(loaded) >= 2:
        base_name, base = loaded[0]
        cmp_dir = os.path.join(args.out, "_compare")
        os.makedirs(cmp_dir, exist_ok=True)
        for name, data in loaded[1:]:
            try:
                plot_ratio_heatmap(base, data, base_name, name, cmp_dir,
                                   args.train_len, attn_set)
            except ValueError as e:
                print(f"  [skip ratio] {name}: {e}")


if __name__ == "__main__":
    main()
