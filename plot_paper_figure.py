"""
Four-panel retention figure: plots model retention curves (across layers) for
four training-stage summaries (e.g. 8k init -> 32k @500 -> 32k @6k -> 128k final).

Summary JSONs are loaded with the same logic as plot_retention_cos.py: each
summary contains per-layer state_cos_sim of shape (H, npos, npos), which we
head-average and convert to a per-layer lag profile.
"""

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, rcParams
from matplotlib.ticker import FuncFormatter, MaxNLocator


def _format_k(x, _pos):
    """Format token-position values as compact 'Nk' / 'N.Mk' labels."""
    if x <= 0:
        return "0"
    k = x / 1000.0
    if k >= 10:
        return f"{int(round(k))}k"
    if abs(k - round(k)) < 1e-6:
        return f"{int(round(k))}k"
    return f"{k:g}k"


rcParams['font.family'] = 'serif'
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 10
rcParams['axes.titlesize'] = 10
rcParams['legend.fontsize'] = 8.5
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['axes.linewidth'] = 0.8
rcParams['xtick.major.width'] = 0.8
rcParams['ytick.major.width'] = 0.8


def load_summary(path: str) -> dict[int, dict]:
    """Load summary JSON (from summarize.py --out) keyed by layer_idx."""
    with open(path, encoding="utf-8") as f:
        entries = json.load(f)
    return {
        int(e["layer_idx"]): {k: np.array(v) if isinstance(v, list) else v
                               for k, v in e.items()}
        for e in entries
    }


def lag_profile(mat_h, npos):
    """Compute mean cosine-sim by lag for a single (npos, npos) matrix."""
    lag_sum = np.zeros(npos, dtype=np.float64)
    lag_count = np.zeros(npos, dtype=np.int64)
    for lag in range(1, npos):
        diag_vals = np.diag(mat_h, k=-lag)
        lag_sum[lag] += diag_vals.sum()
        lag_count[lag] += len(diag_vals)
    valid = lag_count[1:] > 0
    lags = np.arange(1, npos)[valid]
    means = lag_sum[1:][valid] / lag_count[1:][valid]
    return lags, means


def panel_from_summary(path: str):
    """Return (lags, arr, layer_indices) where arr is (n_layers, n_lags)
    of head-averaged retention curves."""
    data = load_summary(path)
    layer_indices = sorted(k for k in data if "state_cos_sim" in data[k])
    if not layer_indices:
        raise ValueError(f"No state_cos_sim data found in {path}")

    rows = []
    ref_lags = None
    for layer_idx in layer_indices:
        mat = data[layer_idx]["state_cos_sim"]   # (H, npos, npos)
        mat_avg = mat.mean(axis=0)               # (npos, npos)
        npos = mat_avg.shape[0]
        lags, means = lag_profile(mat_avg, npos)
        if ref_lags is None:
            ref_lags = lags
        elif len(lags) != len(ref_lags) or not np.array_equal(lags, ref_lags):
            # Truncate to the shortest common prefix so all panels stack cleanly
            n = min(len(ref_lags), len(lags))
            ref_lags = ref_lags[:n]
            rows = [r[:n] for r in rows]
            means = means[:n]
        rows.append(means)
    arr = np.stack(rows, axis=0)
    return ref_lags, arr, layer_indices


def layer_avg_variance(arr):
    """Layer-averaged variance across lags (matches V(t) in recipe section)."""
    return float(np.mean(np.var(arr, axis=1)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries", type=str, nargs=4, required=True,
                        metavar=("S1", "S2", "S3", "S4"),
                        help="Four summary JSONs, plotted left-to-right.")
    parser.add_argument("--titles", type=str, nargs=4, default=None,
                        metavar=("T1", "T2", "T3", "T4"),
                        help="Panel titles (default: filename stems).")
    parser.add_argument("--output-dir", type=str, default="/gpfs/hshen/plots")
    parser.add_argument("--disp-name", type=str, default="retention_4panel",
                        help="Output filename stem (saved as <stem>.pdf and "
                             "<stem>_per_layer.pdf).")
    parser.add_argument("--sample-interval", type=int, default=2048,
                        help="Tokens per sampled position (state_sample_interval "
                             "used at logging time). The x-axis is scaled by "
                             "this factor and labeled in 'Nk' tokens.")
    parser.add_argument("--max-xticks", type=int, default=6,
                        help="Cap on the number of x-axis ticks per panel.")
    args = parser.parse_args()

    titles = args.titles or [
        os.path.splitext(os.path.basename(p))[0] for p in args.summaries
    ]

    panels = []
    for path, title in zip(args.summaries, titles):
        lags, arr, layer_indices = panel_from_summary(path)
        panels.append((title, lags, arr, layer_indices))

    os.makedirs(args.output_dir, exist_ok=True)

    # Shared y-axis range from observed data (clip to [0, 1] visually)
    all_min = min(arr.min() for _, _, arr, _ in panels)
    all_max = max(arr.max() for _, _, arr, _ in panels)
    pad = 0.05 * (all_max - all_min if all_max > all_min else 1.0)
    y_lo = max(0.0, all_min - pad)
    y_hi = min(1.0, all_max + pad)

    # ------------------------------------------------------------------
    # Summary panel: median + percentile bands across layers
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.0), sharey=True)
    for ax, (title, lags, arr, _) in zip(axes, panels):
        x = lags * args.sample_interval
        median = np.median(arr, axis=0)
        p25, p75 = np.percentile(arr, [25, 75], axis=0)
        p05, p95 = np.percentile(arr, [5, 95], axis=0)
        ax.fill_between(x, p05, p95, alpha=0.18, color='#1f77b4',
                        linewidth=0, label='5--95th pct')
        ax.fill_between(x, p25, p75, alpha=0.35, color='#1f77b4',
                        linewidth=0, label='25--75th pct')
        ax.plot(x, median, color='#0d3b66', linewidth=1.8, label='median')
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel('Position offset (tokens)')
        ax.set_title(title)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=args.max_xticks,
                                               steps=[1, 2, 5, 10],
                                               integer=True))
        ax.xaxis.set_major_formatter(FuncFormatter(_format_k))
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)
        v = layer_avg_variance(arr)
        ax.text(0.97, 0.05, f'$V = {v:.4f}$',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=8.5,
                bbox=dict(boxstyle='round,pad=0.3', fc='white',
                          ec='#888', linewidth=0.5))

    axes[0].set_ylabel(r'Retention score $L_{\mathrm{retention}}(k)$')
    axes[0].legend(loc='upper right', framealpha=0.95, frameon=True,
                   edgecolor='#888', fontsize=8)

    plt.tight_layout()
    summary_path = os.path.join(args.output_dir, f"{args.disp_name}.pdf")
    plt.savefig(summary_path, bbox_inches='tight')
    plt.close(fig)

    # ------------------------------------------------------------------
    # Per-layer appendix figure: every layer drawn, viridis by layer index
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), sharey=True)
    cmap = cm.viridis

    max_n_layers = max(arr.shape[0] for _, _, arr, _ in panels)
    for ax, (title, lags, arr, layer_indices) in zip(axes, panels):
        x = lags * args.sample_interval
        n_layers = arr.shape[0]
        for i in range(n_layers):
            denom = max(max_n_layers - 1, 1)
            color = cmap(i / denom)
            ax.plot(x, arr[i], color=color, linewidth=0.9, alpha=0.85)
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel('Position offset (tokens)')
        ax.set_title(title)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=args.max_xticks,
                                               steps=[1, 2, 5, 10],
                                               integer=True))
        ax.xaxis.set_major_formatter(FuncFormatter(_format_k))
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)

    axes[0].set_ylabel(r'Retention score $L_{\mathrm{retention}}(k)$')

    sm = cm.ScalarMappable(cmap=cmap,
                           norm=plt.Normalize(vmin=0, vmax=max_n_layers - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, location='right', shrink=0.85, pad=0.015,
                        aspect=22)
    cbar.set_label('Layer index (early $\\to$ late)', fontsize=9)
    per_layer_path = os.path.join(args.output_dir,
                                  f"{args.disp_name}_per_layer.pdf")
    plt.savefig(per_layer_path, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved {summary_path}")
    print(f"Saved {per_layer_path}")


if __name__ == "__main__":
    main()
