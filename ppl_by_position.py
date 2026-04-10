import os
import argparse
import json
from typing import List

import torch
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM

import datasets
from torch.distributed._composable.fsdp import fully_shard, CPUOffloadPolicy


# ---------------------------------------------------------------------------
# Data helpers (mirrored from ppl_fsdp.py)
# ---------------------------------------------------------------------------

def make_windows(tokens: List[int], window_size: int, stride: int) -> List[List[int]]:
    assert stride > 0
    T = len(tokens)
    if T < window_size:
        return []
    return [tokens[start : start + window_size]
            for start in range(0, T - window_size + 1, stride)]


def build_window_dataset(args, tokenizer, window_size: int, stride: int):
    if args.streaming:
        raw = datasets.load_dataset(
            args.dataset, name=args.subset, split=args.split,
            streaming=True, trust_remote_code=True,
        )
        if args.shuffle_buffer > 0:
            raw = raw.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    else:
        raw = datasets.load_dataset(
            args.dataset, name=args.subset, split=args.split,
            trust_remote_code=True,
        )

    all_windows: List[List[int]] = []
    processed_docs = 0
    for doc in raw:
        if args.sample_size > 0 and processed_docs >= args.sample_size:
            break
        processed_docs += 1
        tokens = tokenizer(
            doc[args.feature],
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
        )["input_ids"]
        wins = make_windows(tokens, window_size, stride)
        all_windows.extend(wins)

    if not all_windows:
        raise RuntimeError(
            f"No windows produced. All documents may be shorter than "
            f"window_size={window_size}. Check --seq-len and --stride."
        )
    return datasets.Dataset.from_dict({"input_ids": all_windows})


def collate_windows(batch):
    return {"input_ids": torch.tensor([x["input_ids"] for x in batch], dtype=torch.long)}


def infer_transformer_layer_types(model):
    base = getattr(model, getattr(model, "base_model_prefix", "backbone"), model)
    candidates = []
    for name in ("layers", "h", "decoder", "encoder", "block", "transformer"):
        mod = getattr(base, name, None)
        if mod is not None:
            if hasattr(mod, "__iter__"):
                for m in mod:
                    candidates.append(type(m))
            else:
                candidates.append(type(mod))
    return set(candidates) if candidates else None


# ---------------------------------------------------------------------------
# Per-position NLL collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_per_position_nll(args, model, dataloader, device, rank, window_size):
    """Compute per-position NLL averaged over all windows.

    Returns (positions, mean_nll) arrays of length (window_size - 1),
    where positions[i] = i+1 (the predicted token position).
    Only rank 0 needs the final result; other ranks return None.
    """
    # Accumulate NLL sums per position across all windows on this rank
    L = window_size - 1  # number of predicted positions
    nll_sum = torch.zeros(L, device=device, dtype=torch.float64)
    count = torch.zeros(1, device=device, dtype=torch.long)

    pbar = tqdm(dataloader, desc=f"collecting (len={window_size})", disable=(rank != 0))
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)  # [B, window_size]
        B = input_ids.size(0)

        model_type = getattr(model.config, "model_type", "").lower()
        model_id = args.model.lower()
        if "zamba2" in model_type or "zamba2" in model_id:
            outputs = model(input_ids=input_ids)
        elif any(k in model_type for k in ("bamba", "nemotron", "mamba")) or \
             any(k in model_id for k in ("bamba", "nemotron", "mamba")):
            outputs = model(input_ids=input_ids, use_cache=False)
        else:
            inference_params = {
                "max_seqlen": window_size + 1,
                "max_batch_size": B,
                "seqlen_offset": 0,
                "batch_size_offset": 0,
            }
            outputs, _ = model(input_ids, inference_params=inference_params)

        logits = outputs.logits.float()  # [B, window_size, V]
        # shift: predict position i+1 from position i
        shift_logits = logits[:, :-1, :]      # [B, L, V]
        shift_labels = input_ids[:, 1:]        # [B, L]

        # Per-position cross-entropy: [B, L]
        per_pos_nll = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="none",
        ).view(B, L)

        # Sum over batch dimension, accumulate per position
        nll_sum += per_pos_nll.double().sum(dim=0)
        count[0] += B

    # All-reduce across ranks
    dist.all_reduce(nll_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)

    if rank == 0:
        mean_nll = (nll_sum / count[0].double()).cpu().numpy()
        positions = np.arange(1, L + 1)
        return positions, mean_nll
    return None, None


# ---------------------------------------------------------------------------
# Plotting (saving convention from plot_retention_cos.py)
# ---------------------------------------------------------------------------

def plot_ppl_by_position(positions, mean_nll, window_size, out_dir, disp_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ppl = np.exp(mean_nll)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Left: NLL per position
    ax = axes[0]
    ax.plot(positions, mean_nll, linewidth=0.6, alpha=0.8)
    ax.set_xlabel("Token position", fontsize=11, fontweight="bold")
    ax.set_ylabel("NLL (nats)", fontsize=11, fontweight="bold")
    ax.set_title(f"{disp_name} — NLL by position (window={window_size})", fontsize=11)
    ax.set_xlim(left=0)

    # Right: PPL per position
    ax = axes[1]
    ax.plot(positions, ppl, linewidth=0.6, alpha=0.8, color="tab:orange")
    ax.set_xlabel("Token position", fontsize=11, fontweight="bold")
    ax.set_ylabel("Perplexity", fontsize=11, fontweight="bold")
    ax.set_title(f"{disp_name} — PPL by position (window={window_size})", fontsize=11)
    ax.set_xlim(left=0)

    fig.tight_layout()
    fname = os.path.join(out_dir, f"[window{window_size}]ppl_by_position.png")
    fig.savefig(fname, dpi=600)
    plt.close(fig)
    return fname


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    # Model & precision
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--bf16", action="store_true")
    # Dataset
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--feature", type=str, default="text")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--shuffle-buffer", type=int, default=0)
    # Window extraction
    parser.add_argument("--seq-len", type=int, nargs="+", default=[2048])
    parser.add_argument("--stride", type=int, default=0)
    # DataLoader
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    # FSDP
    parser.add_argument("--cpu-offload", action="store_true")
    # Output
    parser.add_argument("--output-dir", type=str, default="/gpfs/hshen/plots")
    parser.add_argument("--disp-name", type=str, default=None,
                        help="Output subdirectory name (defaults to model basename)")

    args = parser.parse_args()

    # Distributed init
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank,
                            world_size=world_size, device_id=device)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = getattr(tokenizer, "eos_token", tokenizer.unk_token)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model_kwargs = dict(
        trust_remote_code=True,
        dtype=torch.bfloat16 if args.bf16 else torch.float16,
    )
    if any(k in args.model.lower() for k in ("nemotron", "bamba", "mamba")):
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    layer_types = infer_transformer_layer_types(model) or set()
    fsdp_kwargs = {}
    if args.cpu_offload:
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()

    for module in model.modules():
        if type(module) in layer_types:
            fully_shard(module, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)
    if not args.cpu_offload:
        model.to(device)
    fsdp_model = model

    disp_name = args.disp_name or os.path.basename(args.model.rstrip("/"))
    out_dir = os.path.join(args.output_dir, disp_name)
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    try:
        for window_size in sorted(args.seq_len):
            stride = window_size if args.stride == 0 else args.stride

            if rank == 0:
                print(f"\n{'='*60}")
                print(f"Extracting windows (size={window_size}, stride={stride}) ...")
            window_dataset = build_window_dataset(args, tokenizer, window_size, stride)
            if rank == 0:
                print(f"  {len(window_dataset)} windows extracted.")
            dist.barrier()

            sampler = DistributedSampler(
                window_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=True,
            )
            dataloader = DataLoader(
                window_dataset,
                batch_size=args.batch_size,
                sampler=sampler,
                num_workers=args.num_workers,
                collate_fn=collate_windows,
                pin_memory=True,
                persistent_workers=False if args.num_workers == 0 else True,
            )

            if rank == 0:
                print(f"[rank {rank}] Collecting per-position PPL (window={window_size}) ...")

            positions, mean_nll = collect_per_position_nll(
                args, fsdp_model, dataloader, device, rank, window_size,
            )

            if rank == 0 and positions is not None:
                fname = plot_ppl_by_position(positions, mean_nll, window_size, out_dir, disp_name)
                print(f"  Saved: {fname}")

                # Also save raw data as JSON for later analysis
                data_fname = os.path.join(out_dir, f"[window{window_size}]ppl_by_position.json")
                with open(data_fname, "w", encoding="utf-8") as f:
                    json.dump({
                        "window_size": window_size,
                        "positions": positions.tolist(),
                        "mean_nll": mean_nll.tolist(),
                        "mean_ppl": np.exp(mean_nll).tolist(),
                    }, f)
                print(f"  Saved: {data_fname}")

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
