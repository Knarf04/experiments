"""
plot_geometric_extension.py — geometric-extension diagnostic for SSM long-context drift.

Consumes one or more .npz files produced by `summarize.py --state-mag-out` that
include the geometric-extension fields (ext_mean_k1/k2/k3/kinf, log_decay_mean).
Those fields are emitted when the per-record JSONL recorder also emits
`log_decay_cumulative` and `state_inner_product` (see modeling_nemotron_h.py).

For each (layer, head, bin) we have the *true mean* (across N samples) of the
exact extended magnitude

    ||α_{0:t} · S_k · h_T + h_t||         k ∈ {1, 2, 3, ∞}

where S_k = sum_{j=0}^{k-1} α^j and α = exp(log_decay_total). k=1 is the actual
single-pass magnitude; k=2/3 are 1- and 2-document repeats; k=∞ is the
algebraic-limit projection.

For each k we apply the same "Row 4" within-group cosine-similarity drift metric
that plot_state_mag.py uses for the actual-magnitude trajectory. The reference
for *all four* drift curves is the actual measured early-context (mean[..., :ref_bins]),
so drift always means "how far has this trajectory diverged from real early
context at this hypothetical extension level." k=1 reproduces the existing Row 4.

Usage:
    python plot_geometric_extension.py step6000.npz \
        --out plots/geo_ext_step6000

    python plot_geometric_extension.py 8k_base.npz step500.npz step6000.npz 128k_inst.npz \
        --out plots/geo_ext_compare \
        --names "8K base" "step 500" "step 6000" "128K Instruct"

Outputs (per checkpoint):
    <ckpt>/layer_lines/L{n:03d}.png      4 overlaid drift trajectories per layer
    <ckpt>/extension_aggregate.png       median + p25-p75 across layers, 4 lines

Outputs (aggregate, written under --out):
    extension_heatmap.png                4 × N_checkpoints heatmap (rows = k)
    extension_aggregate_compare.png      4-panel cross-checkpoint overlay (≥2 inputs)
"""

import argparse
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt


REQUIRED_KEYS = (
    "layer_idx", "positions", "seq_len", "state_bin_size",
    "mean", "ext_mean_k1", "ext_mean_k2", "ext_mean_k3", "ext_mean_kinf",
    "log_decay_mean",
)

EPS = 1e-12
PATTERN_DRIFT_CAP = 1.0
K_LABELS = ("k=1 (actual)", "k=2 (1 doc repeat)", "k=3 (2 doc repeats)", "k=∞ (limit)")
K_KEYS   = ("ext_mean_k1", "ext_mean_k2", "ext_mean_k3", "ext_mean_kinf")
# Sequential viridis: lighter = closer to actual, darker = deeper into the algebraic future.
K_COLORS = plt.cm.viridis(np.linspace(0.15, 0.85, len(K_LABELS)))


def load_npz(path: str) -> dict:
    z = np.load(path)
    missing = [k for k in REQUIRED_KEYS if k not in z.files]
    if missing:
        raise ValueError(
            f"{path}: missing geo-extension keys {missing}. "
            f"Re-run summarize.py on JSONL with `log_decay_cumulative` and "
            f"`state_inner_product` fields."
        )
    return {k: z[k] for k in z.files}


def _ckpt_name(path: str, override: Optional[str]) -> str:
    return override or os.path.splitext(os.path.basename(path))[0]


def within_group_drift(
    mag: np.ndarray, ref_mag: np.ndarray, ngroups: int, ref_bins: int,
    eps: float = EPS,
) -> np.ndarray:
    """1 - cos_sim of within-group, mean-centered log-mag pattern vs an
    *external* reference's early-bin pattern, averaged over the n_groups groups.

    mag, ref_mag: [L, H, n_bins]
    Returns       [L, n_bins]

    For mag == ref_mag this reduces exactly to plot_state_mag._agg_within_group_pattern_drift.
    """
    if mag.shape != ref_mag.shape:
        raise ValueError(f"shape mismatch: mag={mag.shape}, ref_mag={ref_mag.shape}")
    L, H, n_bins = mag.shape
    if H % ngroups != 0:
        raise ValueError(f"H={H} not divisible by ngroups={ngroups}")
    S = H // ngroups

    log_mag = np.log(mag + eps)
    log_ref = np.log(ref_mag + eps)
    log_mag_g = log_mag.reshape(L, ngroups, S, n_bins)
    log_ref_g = log_ref.reshape(L, ngroups, S, n_bins)

    ref = np.median(log_ref_g[:, :, :, :ref_bins], axis=3)              # [L, G, S]

    cur_c = log_mag_g - log_mag_g.mean(axis=2, keepdims=True)           # [L, G, S, n_bins]
    ref_c = ref - ref.mean(axis=2, keepdims=True)                       # [L, G, S]

    dot      = (cur_c * ref_c[..., None]).sum(axis=2)                   # [L, G, n_bins]
    cur_norm = np.linalg.norm(cur_c, axis=2)                            # [L, G, n_bins]
    ref_norm = np.linalg.norm(ref_c, axis=2, keepdims=True)             # [L, G, 1]
    cos_sim  = dot / (cur_norm * ref_norm + eps)                        # [L, G, n_bins]

    return (1.0 - cos_sim).mean(axis=1)                                 # [L, n_bins]


def compute_all_drifts(data: dict, ngroups: int, ref_bins: int) -> List[np.ndarray]:
    """Returns [drift_k1, drift_k2, drift_k3, drift_kinf], each [L, n_bins]."""
    ref = data["mean"]
    return [within_group_drift(data[k], ref, ngroups, ref_bins) for k in K_KEYS]


def per_layer_alpha_summary(data: dict) -> dict:
    """α_total = exp(log_decay_mean[..., -1]); per-layer median/min/max across heads."""
    log_total = data["log_decay_mean"][..., -1]                         # [L, H]
    alpha = np.exp(log_total)                                           # [L, H]
    return {
        "alpha":  alpha,
        "median": np.median(alpha, axis=1),
        "min":    alpha.min(axis=1),
        "max":    alpha.max(axis=1),
    }


def within_group_alpha_spread(data: dict, ngroups: int) -> np.ndarray:
    """Per-(layer, group) spread of α (max − min). Returns [L, ngroups]."""
    log_total = data["log_decay_mean"][..., -1]                         # [L, H]
    alpha = np.exp(log_total)
    L, H = alpha.shape
    if H % ngroups != 0:
        return np.empty((L, 0))
    S = H // ngroups
    g = alpha.reshape(L, ngroups, S)
    return g.max(axis=2) - g.min(axis=2)                                # [L, ngroups]


def print_sanity(name: str, data: dict, drifts: List[np.ndarray], ngroups: int) -> None:
    print(f"\n=== {name} ===")
    a = per_layer_alpha_summary(data)
    L = a["alpha"].shape[0]
    bad_mask = a["alpha"] > 0.9999
    n_bad = int(bad_mask.sum())
    print(f"  α_total per layer (median across heads): "
          f"min={a['median'].min():.4f}  max={a['median'].max():.4f}")
    print(f"  α_total min/max across heads (worst layer): "
          f"min={a['min'].min():.4f}  max={a['max'].max():.4f}")
    if n_bad:
        bad_layers = sorted({int(li) for li in np.where(bad_mask)[0]})
        print(f"  ⚠ {n_bad} (layer, head) entries with α_total > 0.9999 — "
              f"these blow up 1/(1-α). Layers: {bad_layers}")
    else:
        print("  (no heads with α_total > 0.9999)")

    for label, drift in zip(K_LABELS, drifts):
        print(f"  drift {label:<22s}  median={float(np.median(drift)):.4f}  "
              f"max={float(drift.max()):.4f}")

    spread = within_group_alpha_spread(data, ngroups)
    if spread.size:
        print(f"  within-group α-spread (max - min inside a group of "
              f"{data['mean'].shape[1] // ngroups}): "
              f"median={float(np.median(spread)):.4f}  max={float(spread.max()):.4f}")

    n_clipped = int(data.get("geo_clipped_neg_count", 0))
    if n_clipped:
        print(f"  ⚠ {n_clipped} per-sample sq<0 entries clipped during summarize.py "
              f"(should be ~0 if recording is faithful)")


# ─────────────────────────── plotting ───────────────────────────

def plot_layer_lines(
    name: str, data: dict, drifts: List[np.ndarray], out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    layer_idx = data["layer_idx"]
    positions = data["positions"]
    bin_size = int(data["state_bin_size"])
    alpha_med = per_layer_alpha_summary(data)["median"]
    L = drifts[0].shape[0]

    for li in range(L):
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        for label, drift, color in zip(K_LABELS, drifts, K_COLORS):
            ax.plot(positions, drift[li, :], color=color, linewidth=1.6,
                    marker="o", markersize=2.5, label=label)
        ax.set_xlabel("position (token index, end of bin)")
        ax.set_ylabel("1 − cos_sim (within-group pattern drift)")
        ax.set_ylim(0.0, PATTERN_DRIFT_CAP)
        ax.grid(True, alpha=0.3)
        ax.set_title(
            f"{name} — Mamba layer {int(layer_idx[li])}  "
            f"(median α = {alpha_med[li]:.4f}, bin_size={bin_size})"
        )
        ax.legend(loc="best", fontsize=8, framealpha=0.85)
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"L{int(layer_idx[li]):03d}.png")
        fig.savefig(out_path, dpi=120)
        plt.close(fig)


def plot_extension_heatmap(
    loaded: Sequence[Tuple[str, dict, List[np.ndarray]]], out_path: str,
) -> None:
    N = len(loaded)
    if N == 0:
        return
    K = len(K_LABELS)

    L_max = max(d["mean"].shape[0] for _, d, _ in loaded)
    n_bins_max = max(d["mean"].shape[2] for _, d, _ in loaded)
    cell_w = max(4.5, n_bins_max * 0.075)
    cell_h = max(2.6, L_max * 0.16)

    fig = plt.figure(figsize=(cell_w * N + 1.6, cell_h * K + 0.6))
    gs = fig.add_gridspec(K, N + 1, width_ratios=[*([1.0] * N), 0.05],
                          hspace=0.40, wspace=0.18)

    # Shared vmax across all cells (per plan §3e).
    global_max = 0.0
    for _, _, drifts in loaded:
        for d in drifts:
            global_max = max(global_max, float(d.max()))
    vmax = min(max(global_max, 1e-3), PATTERN_DRIFT_CAP)

    for r in range(K):
        last_im = None
        for c in range(N):
            name, data, drifts = loaded[c]
            metric = drifts[r]                                          # [L, n_bins]
            L, n_bins = metric.shape
            bin_size = int(data["state_bin_size"])

            ax = fig.add_subplot(gs[r, c])
            im = ax.imshow(
                metric, aspect="auto", origin="upper", cmap="magma",
                vmin=0.0, vmax=vmax, interpolation="nearest",
            )
            last_im = im

            ax.set_yticks(np.arange(L))
            ax.set_yticklabels([str(int(li)) for li in data["layer_idx"]],
                               fontsize=6)
            ax.tick_params(axis="x", labelsize=7)

            xticks = np.linspace(0, n_bins - 1, min(6, n_bins)).astype(int)
            ax.set_xticks(xticks)
            ax.set_xticklabels([str((i + 1) * bin_size) for i in xticks])

            if c == 0:
                ax.set_ylabel(f"{K_LABELS[r]}\nlayer", fontsize=9)
            if r == K - 1:
                ax.set_xlabel("position", fontsize=9)
            if r == 0:
                ax.set_title(name, fontsize=10)

        if last_im is not None:
            cax = fig.add_subplot(gs[r, N])
            cb = fig.colorbar(last_im, cax=cax)
            cb.ax.tick_params(labelsize=7)
            cb.set_label("1 − cos_sim", fontsize=8)

    fig.suptitle("Geometric-extension drift  (rows: extension level, cols: checkpoint)",
                 fontsize=12, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_aggregate_ribbon(
    name: str, data: dict, drifts: List[np.ndarray], out_path: str,
) -> None:
    """One figure per checkpoint: median across layers + p25-p75 ribbon, 4 lines."""
    positions = data["positions"]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for label, drift, color in zip(K_LABELS, drifts, K_COLORS):
        med = np.median(drift, axis=0)
        p25 = np.percentile(drift, 25, axis=0)
        p75 = np.percentile(drift, 75, axis=0)
        ax.fill_between(positions, p25, p75, color=color, alpha=0.18)
        ax.plot(positions, med, color=color, linewidth=1.8, label=label)
    ax.set_xlabel("position (token index, end of bin)")
    ax.set_ylabel("1 − cos_sim (median ± p25-p75 across layers)")
    ax.set_ylim(0.0, PATTERN_DRIFT_CAP)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{name} — geometric-extension drift, layer-aggregated")
    ax.legend(loc="best", fontsize=9, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_aggregate_compare(
    loaded: Sequence[Tuple[str, dict, List[np.ndarray]]], out_path: str,
) -> None:
    """4-panel figure (one per k) with all checkpoints overlaid as median + ribbon."""
    if len(loaded) < 2:
        return
    K = len(K_LABELS)
    fig, axes = plt.subplots(K, 1, figsize=(8.0, 3.4 * K), sharex=False)
    ckpt_colors = plt.cm.tab10(np.arange(len(loaded)) % 10)

    for r, ax in enumerate(axes):
        for ci, (name, data, drifts) in enumerate(loaded):
            positions = data["positions"]
            d = drifts[r]
            med = np.median(d, axis=0)
            p25 = np.percentile(d, 25, axis=0)
            p75 = np.percentile(d, 75, axis=0)
            ax.fill_between(positions, p25, p75, color=ckpt_colors[ci], alpha=0.15)
            ax.plot(positions, med, color=ckpt_colors[ci], linewidth=1.8, label=name)
        ax.set_title(K_LABELS[r], fontsize=10)
        ax.set_ylabel("1 − cos_sim")
        ax.set_ylim(0.0, PATTERN_DRIFT_CAP)
        ax.grid(True, alpha=0.3)
        if r == 0:
            ax.legend(loc="best", fontsize=9, framealpha=0.85)
        if r == K - 1:
            ax.set_xlabel("position (token index, end of bin)")
    fig.suptitle("Cross-checkpoint comparison (median ± p25-p75 across layers)",
                 fontsize=12, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ─────────────────────────── main ───────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Geometric-extension drift diagnostic plots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("inputs", nargs="+",
                        help="One or more .npz files from summarize.py")
    parser.add_argument("--out", default="plots/geometric_extension")
    parser.add_argument("--names", nargs="+", default=None,
                        help="Display names per input; defaults to .npz basenames.")
    parser.add_argument("--ngroups", type=int, default=8,
                        help="RMSNorm groups (Nemotron-H default: 8).")
    parser.add_argument("--ref-bins", type=int, default=4,
                        help="Number of early bins used as Row-4 reference.")
    parser.add_argument("--no-per-layer", action="store_true",
                        help="Skip per-layer line PNGs (default: emit all).")
    args = parser.parse_args()

    if args.names is not None and len(args.names) != len(args.inputs):
        parser.error("--names must match the number of input .npz files")

    os.makedirs(args.out, exist_ok=True)

    loaded: List[Tuple[str, dict, List[np.ndarray]]] = []
    for i, path in enumerate(args.inputs):
        name = _ckpt_name(path, args.names[i] if args.names else None)
        data = load_npz(path)
        drifts = compute_all_drifts(data, ngroups=args.ngroups, ref_bins=args.ref_bins)
        loaded.append((name, data, drifts))
        print_sanity(name, data, drifts, ngroups=args.ngroups)

    # Per-checkpoint outputs
    for name, data, drifts in loaded:
        ckpt_dir = os.path.join(args.out, name)
        os.makedirs(ckpt_dir, exist_ok=True)
        if not args.no_per_layer:
            plot_layer_lines(name, data, drifts, os.path.join(ckpt_dir, "layer_lines"))
        plot_aggregate_ribbon(name, data, drifts,
                              os.path.join(ckpt_dir, "extension_aggregate.png"))

    # Aggregate outputs
    plot_extension_heatmap(loaded, os.path.join(args.out, "extension_heatmap.png"))
    if len(loaded) >= 2:
        plot_aggregate_compare(loaded,
                               os.path.join(args.out, "extension_aggregate_compare.png"))

    print(f"\nWrote outputs to {args.out}")


if __name__ == "__main__":
    main()
