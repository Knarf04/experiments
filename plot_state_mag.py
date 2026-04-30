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
DRIFT_NATS_CAP = 8.0     # cap |log-ratio| (natural log, "nats") drift heatmaps at this max
PATTERN_DRIFT_CAP = 1.0  # cap (1 − cos_sim) within-group pattern-drift heatmaps at this max


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


def resolve_n_groups(model_config_path: Optional[str], fallback: int) -> int:
    """Pull `mamba_n_groups` (or `n_groups`) from a HF config.json; else fallback.

    For Nemotron-H 8B / Bamba this is `mamba_n_groups`; for pure Mamba2 it's
    `n_groups`. Mamba2-only configs without either key fall back silently.
    """
    if model_config_path:
        with open(model_config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        for key in ("mamba_n_groups", "n_groups"):
            if key in cfg:
                ng = int(cfg[key])
                print(f"  resolved n_groups={ng} from model-config[{key!r}]")
                return ng
        print(f"  no mamba_n_groups / n_groups key in model-config; "
              f"using fallback n_groups={fallback}")
    return fallback


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


def _agg_per_head_mean_abs_log(mean: np.ndarray, ref_bins: int) -> np.ndarray:
    """|log(mean_h(t) / median_first_refbins(mean_h))| averaged across heads → [L, n_bins].

    "Typical head drift" — smooths over head-to-head spread, so a uniform
    layer-wide shift shows up cleanly but a few catastrophic outlier heads
    are diluted by the H-1 quiet ones.
    """
    log_mag = np.log(mean + EPS)                                              # [L, H, n_bins]
    ref = np.median(log_mag[:, :, :ref_bins], axis=2, keepdims=True)          # [L, H, 1]
    return np.abs(log_mag - ref).mean(axis=1)                                 # [L, n_bins]


def _agg_per_head_max_abs_log(mean: np.ndarray, ref_bins: int) -> np.ndarray:
    """max_h |log(mean_h(t) / median_first_refbins(mean_h))| → [L, n_bins].

    "Worst head in the layer" — outlier-sensitive. A layer with even one
    catastrophically drifting head registers immediately, regardless of how
    quiet the other heads are.
    """
    log_mag = np.log(mean + EPS)
    ref = np.median(log_mag[:, :, :ref_bins], axis=2, keepdims=True)
    return np.abs(log_mag - ref).max(axis=1)                                  # [L, n_bins]


def _agg_l2_total_abs_log(mean: np.ndarray, ref_bins: int) -> np.ndarray:
    """|log(sqrt(sum_h ||h_h||²)(t) / median_first_refbins(total))| → [L, n_bins].

    "RMSNorm-relevant total drift." The L2-aggregated total is exactly what
    RMSNorm divides by (modulo a 1/sqrt(H·D) constant), so a single head
    blowing up dominates `total_mag` the same way it dominates the post-norm
    output direction.
    """
    total = np.sqrt((mean ** 2).sum(axis=1) + EPS)                            # [L, n_bins]
    log_total = np.log(total + EPS)
    ref = np.median(log_total[:, :ref_bins], axis=1, keepdims=True)           # [L, 1]
    return np.abs(log_total - ref)                                            # [L, n_bins]


def _agg_within_group_pattern_drift(
    mean: np.ndarray, ref_bins: int, n_groups: int
) -> np.ndarray:
    """1 − cos_sim of within-group, mean-centered log-magnitude pattern vs early
    context, averaged over groups → [L, n_bins].

    Measures the RMSNorm-unabsorbable component of per-head magnitude drift.
    Nemotron-H's MambaRMSNormGated has ngroups=8 over nheads=128 (group_size=16);
    each group's RMS denominator absorbs uniform within-group scaling. What
    remains — and what downstream layers see as a structurally different
    direction — is the relative pattern of magnitudes among the 16 heads of
    each group.

    For each (layer, position, group):
      v(t) = log(mean[layer, group_heads, t]) ∈ R^{group_size}
      v(t) := v(t) - mean(v(t))     # remove uniform-scale component
      v_ref similar from median over first `ref_bins` bins
      drift = 1 - cosine(v(t), v_ref)
    Then average drift across the n_groups groups in each layer.
    """
    L, H, n_bins = mean.shape
    if H % n_groups != 0:
        raise ValueError(f"H={H} not divisible by n_groups={n_groups}")
    group_size = H // n_groups

    log_mag = np.log(mean + EPS)                                                # [L, H, n_bins]
    ref = np.median(log_mag[:, :, :ref_bins], axis=2)                           # [L, H]

    log_mag_g = log_mag.reshape(L, n_groups, group_size, n_bins)                # [L, G, S, n_bins]
    ref_g     = ref.reshape(L, n_groups, group_size)                            # [L, G, S]

    cur_c = log_mag_g - log_mag_g.mean(axis=2, keepdims=True)                   # [L, G, S, n_bins]
    ref_c = ref_g - ref_g.mean(axis=2, keepdims=True)                           # [L, G, S]

    dot = (cur_c * ref_c[..., None]).sum(axis=2)                                # [L, G, n_bins]
    cur_norm = np.linalg.norm(cur_c, axis=2)                                    # [L, G, n_bins]
    ref_norm = np.linalg.norm(ref_c, axis=2, keepdims=True)                     # [L, G, 1]
    cos_sim  = dot / (cur_norm * ref_norm + EPS)                                # [L, G, n_bins]

    return (1.0 - cos_sim).mean(axis=1)                                         # [L, n_bins]


def plot_layer_drift_panel(
    loaded, out_path: str, ref_bins: int, n_groups: int,
    attn_set: Set[int], train_len: Optional[int]
) -> None:
    """Multi-aggregator panel: rows = aggregators, cols = checkpoints.

    Each cell is a [L, n_bins] heatmap on the magma colormap. One shared
    colorbar per row (so the gap between e.g. mean-abs-log and L2-total is
    *visible* — that gap itself diagnoses head-concentration of drift).

    Aggregators (rows):
      1. per-head mean |Δ log|       — typical head drift
      2. L2-total |Δ log|            — RMSNorm-input scale drift
      3. per-head max |Δ log|        — outlier-sensitive (worst head)
      4. within-group pattern drift  — 1 − cos_sim of mean-centered log-mag
                                       pattern within each RMSNorm group;
                                       what RMSNorm cannot absorb.
    """
    N = len(loaded)
    if N == 0:
        return

    aggregators = [
        ("per-head mean |Δ log|",
         lambda m: _agg_per_head_mean_abs_log(m, ref_bins),
         DRIFT_NATS_CAP, "|Δ log| (nats)"),
        ("L2-total |Δ log|",
         lambda m: _agg_l2_total_abs_log(m, ref_bins),
         DRIFT_NATS_CAP, "|Δ log| (nats)"),
        ("per-head max |Δ log|",
         lambda m: _agg_per_head_max_abs_log(m, ref_bins),
         DRIFT_NATS_CAP, "|Δ log| (nats)"),
        (f"within-group pattern drift\n(G={n_groups}, group_size={None})"
         if any(d["mean"].shape[1] % n_groups for _, d in loaded)
         else f"within-group pattern drift\n(G={n_groups}, group_size={loaded[0][1]['mean'].shape[1] // n_groups})",
         lambda m: _agg_within_group_pattern_drift(m, ref_bins, n_groups),
         PATTERN_DRIFT_CAP, "1 − cos_sim"),
    ]
    K = len(aggregators)

    # Compute every cell's metric first so we can derive per-row vmax.
    metrics: List[List[np.ndarray]] = []
    for label, fn, _, _ in aggregators:
        row = []
        for _, data in loaded:
            try:
                row.append(fn(data["mean"]))
            except ValueError as e:
                print(f"  [skip drift-panel row {label!r}] {e}")
                row.append(None)
        metrics.append(row)

    # Figure layout: K rows × (N + 1) cols, the trailing col holds colorbars.
    L_max = max(data["mean"].shape[0] for _, data in loaded)
    n_bins_max = max(data["mean"].shape[2] for _, data in loaded)
    cell_w = max(4.5, n_bins_max * 0.075)
    cell_h = max(2.6, L_max * 0.16)
    fig = plt.figure(figsize=(cell_w * N + 1.6, cell_h * K + 0.5))
    gs = fig.add_gridspec(K, N + 1, width_ratios=[*([1.0] * N), 0.05],
                          hspace=0.40, wspace=0.18)

    for r, (label, _, cap, cbar_label) in enumerate(aggregators):
        row_metrics = [m for m in metrics[r] if m is not None]
        if not row_metrics:
            continue
        row_max = max(float(m.max()) for m in row_metrics)
        vmax = min(max(row_max, 1e-3), cap)

        last_im = None
        for c in range(N):
            name, data = loaded[c]
            metric = metrics[r][c]
            if metric is None:
                ax = fig.add_subplot(gs[r, c])
                ax.text(0.5, 0.5, "(skipped)", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
                ax.set_xticks([]); ax.set_yticks([])
                continue
            L, n_bins = metric.shape
            bin_size = int(data["state_bin_size"])

            ax = fig.add_subplot(gs[r, c])
            im = ax.imshow(metric, aspect="auto", origin="lower",
                           cmap="magma", vmin=0.0, vmax=vmax)
            last_im = im

            if r == 0:
                ax.set_title(name, fontsize=10)
            if r == K - 1:
                ax.set_xlabel(f"position bin (bin_size={bin_size})")
            else:
                ax.set_xticklabels([])

            if c == 0:
                ax.set_yticks(np.arange(L))
                ax.set_yticklabels([str(int(li)) for li in data["layer_idx"]],
                                   fontsize=7)
                ax.set_ylabel(f"{label}\nlayer", fontsize=9)
                _color_attn_ticks(ax, data["layer_idx"], attn_set)
            else:
                ax.set_yticks([])

            if train_len is not None:
                x = _bin_for_train_len(train_len, bin_size)
                if 0 <= x <= n_bins - 1:
                    ax.axvline(x, color="cyan", linestyle="--", linewidth=0.8, alpha=0.85)

        cax = fig.add_subplot(gs[r, N])
        if last_im is not None:
            fig.colorbar(last_im, cax=cax, label=cbar_label)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_quantile_fan(loaded, out_path: str, ref_bins: int = 4) -> None:
    """Overlay per-head normalized magnitude quantile bands across checkpoints.

    For each (layer, head), divide its trajectory by the median of its first
    `ref_bins` position bins (its own early-context baseline). Pool all
    (layer, head) curves and plot median + p25-p75 inner band + p10-p90 outer
    band per checkpoint, all on the same axes.

    loaded: list of (name, data) pairs, where data is the dict returned by load_npz.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab10.colors

    for i, (name, data) in enumerate(loaded):
        mean = data["mean"]                   # [L, H, n_bins]
        positions = data["positions"]         # [n_bins]
        L, H, n_bins = mean.shape
        if n_bins < ref_bins:
            print(f"  [skip fan] {name}: n_bins={n_bins} < ref_bins={ref_bins}")
            continue

        ref = np.median(mean[:, :, :ref_bins], axis=2, keepdims=True)  # [L, H, 1]
        rel = mean / np.maximum(ref, 1e-12)                            # [L, H, n_bins]
        flat = rel.reshape(L * H, n_bins)

        med = np.median(flat, axis=0)
        p10, p25, p75, p90 = (np.percentile(flat, q, axis=0) for q in (10, 25, 75, 90))

        c = colors[i % len(colors)]
        ax.fill_between(positions, p10, p90, color=c, alpha=0.15, linewidth=0)
        ax.fill_between(positions, p25, p75, color=c, alpha=0.30, linewidth=0)
        ax.plot(positions, med, color=c, linewidth=2.0, label=name)

    ax.axhline(1.0, color="black", linewidth=0.6, linestyle=":")
    ax.set_yscale("log")
    ax.set_xlabel("position (tokens)")
    ax.set_ylabel(r"$\|h\|_F$ / early-context baseline")
    ax.set_title("Per-head SSM state magnitude relative to early context "
                 "(median, p25-p75, p10-p90)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


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
    parser.add_argument("--ref-bins", type=int, default=4,
                        help="Number of leading bins to use as the early-context baseline for "
                             "the layer-drift panel and quantile fan (default: 4).")
    parser.add_argument("--n-groups", type=int, default=8,
                        help="Number of RMSNorm groups (mamba_n_groups). Used by the within-group "
                             "pattern-drift row of the drift panel. Auto-resolved from "
                             "--model-config if available; this is the fallback (default: 8, "
                             "matches Nemotron-H 8B / Bamba 9B).")
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
    print("Resolving n_groups...")
    n_groups = resolve_n_groups(args.model_config, args.n_groups)

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

    if len(loaded) >= 1:
        plot_quantile_fan(loaded, os.path.join(args.out, "quantile_fan.png"),
                          ref_bins=args.ref_bins)
        plot_layer_drift_panel(loaded, os.path.join(args.out, "layer_drift_panel.png"),
                               ref_bins=args.ref_bins, n_groups=n_groups,
                               attn_set=attn_set, train_len=args.train_len)

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
