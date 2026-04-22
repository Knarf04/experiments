import argparse
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm

FNAME_RE = re.compile(r"layer(\d+)_head(\d+)_id(\d+)\.pt$")


def scan_files(root: str) -> dict:
    """Return {(layer_idx, sample_id): {head_idx: path}} for files in root."""
    out = defaultdict(dict)
    for name in os.listdir(root):
        m = FNAME_RE.match(name)
        if not m:
            continue
        layer, head, sid = (int(x) for x in m.groups())
        out[(layer, sid)][head] = os.path.join(root, name)
    return out


def load_map(path: str) -> np.ndarray:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    t = blob["attn_map"]
    if t.dtype == torch.bfloat16:
        t = t.float()
    return t.numpy()


def block_reduce_mean(x: np.ndarray, target: int) -> np.ndarray:
    """Downsample a 2D array so that each dim is <= `target` via block-mean."""
    q, k = x.shape
    bq = max(1, int(np.ceil(q / target)))
    bk = max(1, int(np.ceil(k / target)))
    if bq == 1 and bk == 1:
        return x
    q_trim = (q // bq) * bq
    k_trim = (k // bk) * bk
    x = x[:q_trim, :k_trim]
    return x.reshape(q_trim // bq, bq, k_trim // bk, bk).mean(axis=(1, 3))


def plot_one(attn: np.ndarray, title: str, out_path: str, log: bool, cmap: str):
    fig, ax = plt.subplots(figsize=(5, 5))
    if log:
        pos = attn[attn > 0]
        vmin = float(pos.min()) if pos.size else 1e-8
        vmax = float(attn.max()) if attn.max() > 0 else 1.0
        im = ax.imshow(np.clip(attn, vmin, None), cmap=cmap, aspect="auto",
                       norm=LogNorm(vmin=vmin, vmax=max(vmin * 10, vmax)),
                       interpolation="nearest", origin="upper")
    else:
        im = ax.imshow(attn, cmap=cmap, aspect="auto", interpolation="nearest", origin="upper")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Key position", fontsize=10)
    ax.set_ylabel("Query position", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disp-name", type=str, required=True,
                        help="Model disp_name used when saving attention maps")
    parser.add_argument("--input-dir", type=str, default="/gpfs/hshen/hybrid_attn_map",
                        help="Root dir where attention maps were saved")
    parser.add_argument("--output-dir", type=str, default="/gpfs/hshen/plots",
                        help="Root dir for plot output")
    parser.add_argument("--layer", type=int, default=None,
                        help="Only plot this layer (default: all layers found)")
    parser.add_argument("--sample-id", type=int, default=0,
                        help="Which saved sample id (attn_map_num) to plot")
    parser.add_argument("--linear", action="store_true",
                        help="Use linear colormap (default is log-scale)")
    parser.add_argument("--cmap", type=str, default="viridis")
    parser.add_argument("--max-size", type=int, default=2048,
                        help="Downsample maps to at most this many cells per axis for plotting")
    args = parser.parse_args()

    input_dir = os.path.join(args.input_dir, args.disp_name)
    output_dir = os.path.join(args.output_dir, args.disp_name, "attn_map")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isdir(input_dir):
        print(f"Input dir does not exist: {input_dir}")
        return

    files = scan_files(input_dir)
    if not files:
        print(f"No attention-map files found under {input_dir}")
        return

    layers = sorted({l for (l, sid) in files if sid == args.sample_id})
    if args.layer is not None:
        layers = [l for l in layers if l == args.layer]
    if not layers:
        print(f"No layers for sample_id={args.sample_id} (requested layer={args.layer}).")
        return

    log = not args.linear

    for layer in layers:
        head_files = files[(layer, args.sample_id)]
        heads = sorted(head_files)
        print(f"[layer {layer}] {len(heads)} heads, sample_id={args.sample_id}")

        mean_sum = None
        for h in heads:
            attn = load_map(head_files[h])
            attn_vis = block_reduce_mean(attn, args.max_size)
            plot_one(
                attn_vis,
                title=f"Layer {layer} | Head {h}",
                out_path=os.path.join(
                    output_dir, f"[layer{layer}][id{args.sample_id}]head{h}.png"
                ),
                log=log,
                cmap=args.cmap,
            )
            mean_sum = attn.astype(np.float64) if mean_sum is None else mean_sum + attn.astype(np.float64)
            del attn, attn_vis

        mean_attn = (mean_sum / len(heads)).astype(np.float32)
        mean_vis = block_reduce_mean(mean_attn, args.max_size)
        plot_one(
            mean_vis,
            title=f"Layer {layer} | Mean over {len(heads)} heads",
            out_path=os.path.join(
                output_dir, f"[layer{layer}][id{args.sample_id}]mean.png"
            ),
            log=log,
            cmap=args.cmap,
        )

    print(f"Saved plots to {output_dir}/")


if __name__ == "__main__":
    main()
