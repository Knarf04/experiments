import os
import argparse
from typing import List

import torch
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM

import datasets
from torch.distributed._composable.fsdp import fully_shard, CPUOffloadPolicy


def make_windows(tokens: List[int], window_size: int, stride: int) -> List[List[int]]:
    """Slice a token list into fixed-length sliding windows.
    Returns [] when the document is shorter than window_size.
    """
    assert stride > 0, "stride must be positive"
    T = len(tokens)
    if T < window_size:
        return []
    return [tokens[start : start + window_size]
            for start in range(0, T - window_size + 1, stride)]


def build_window_dataset(args, tokenizer, window_size: int, stride: int):
    """Tokenize each document and emit sliding windows of exactly window_size tokens.

    No padding, no truncation.  Documents shorter than window_size are skipped.
    The full dataset is built on every rank; DistributedSampler handles sharding.
    """
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
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        )["input_ids"]
        wins = make_windows(tokens, window_size, stride)
        all_windows.extend(wins)

    if not all_windows:
        raise RuntimeError(
            f"No windows produced. All documents may be shorter than "
            f"window_size={window_size}. Check --max-length and --stride."
        )

    return datasets.Dataset.from_dict({"input_ids": all_windows})


def collate_windows(batch):
    """Stack fixed-length windows into a batch tensor — no padding needed."""
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


@torch.no_grad()
def eval_windows(args, model, dataloader, device, rank, experiments=None):
    """Score tail-perplexity on pre-extracted fixed-length windows.

    Every window is exactly window_size real tokens with no padding, so no
    attention_mask or validity filtering is needed.
    DistributedSampler + drop_last=True guarantees all ranks execute the same
    number of forward passes — FSDP-safe by construction.
    """
    k_tail = 100

    total_nll = torch.zeros(1, device=device, dtype=torch.float64)
    total_tok = torch.zeros(1, device=device, dtype=torch.long)

    pbar = tqdm(dataloader, desc="windows", disable=(rank != 0))
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)  # [B, L] — real tokens only ✓
        B = input_ids.size(0)

        # Every window in this batch is padding-free — inform recording layers
        if experiments is not None:
            experiments["valid_mask"] = [True] * B

        if "zamba2" in args.model.lower():
            outputs = model(input_ids=input_ids, num_logits_to_keep=k_tail + 1)
        elif any(k in args.model.lower() for k in ("bamba", "nemotron", "mamba")):
            outputs = model(input_ids=input_ids, num_logits_to_keep=k_tail + 1, use_cache=False)
        else:
            inference_params = {
                "max_seqlen": args.max_length + 1,
                "max_batch_size": B,
                "seqlen_offset": 0,
                "batch_size_offset": 0,
            }
            outputs, _ = model(input_ids, num_last_tokens=k_tail + 1,
                               inference_params=inference_params)

        logits = outputs.logits
        V = logits.size(-1)
        T = min(logits.size(1), input_ids.size(1))
        logits = logits[:, :T, :]
        labels = input_ids[:, -T:]

        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = labels[:, 1:].contiguous()
        k_eff = min(k_tail, shift_labels.size(1))
        if k_eff == 0:
            continue

        nll_sum = F.cross_entropy(
            shift_logits[:, -k_eff:, :].reshape(-1, V),
            shift_labels[:, -k_eff:].reshape(-1),
            reduction="sum",
        )
        total_nll[0] += nll_sum.double()
        total_tok[0] += B * k_eff

    dist.all_reduce(total_nll, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tok, op=dist.ReduceOp.SUM)
    return total_nll[0], total_tok[0]


def main():
    parser = argparse.ArgumentParser()
    # Model & precision
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--bf16", action="store_true",
                        help="Use bfloat16 (recommended on Ampere/ADA/Hopper).")
    # Dataset
    parser.add_argument("--dataset", type=str, required=True,
                        help='HF dataset id or local script, e.g. "emozilla/pg19"')
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--sample-size", type=int, default=0,
                        help="Max documents to process (0 = all).")
    parser.add_argument("--feature", type=str, default="text",
                        help="Text column name in the dataset.")
    parser.add_argument("--streaming", action="store_true",
                        help="Use HF streaming mode.")
    parser.add_argument("--shuffle-buffer", type=int, default=0,
                        help="Streaming shuffle buffer size; 0 = no shuffle.")
    # Window extraction
    parser.add_argument("--max-length", type=int, default=2048,
                        help="Window size in tokens. Every evaluated chunk is exactly this long.")
    parser.add_argument("--stride", type=int, default=0,
                        help="Stride between windows in tokens. "
                             "0 = non-overlapping (stride == max-length).")
    # DataLoader
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    # FSDP
    parser.add_argument("--cpu-offload", action="store_true",
                        help="FSDP CPU parameter offload (slower but lighter on GPU memory).")

    args = parser.parse_args()
    if args.stride == 0:
        args.stride = args.max_length  # non-overlapping by default

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

    # Build the window dataset on every rank (same windows, DistributedSampler shards it).
    # All windows are exactly max_length real tokens — no padding anywhere.
    if rank == 0:
        print(f"Extracting windows (size={args.max_length}, stride={args.stride}) ...")
    window_dataset = build_window_dataset(args, tokenizer, args.max_length, args.stride)
    if rank == 0:
        print(f"  {len(window_dataset)} windows extracted.")
    dist.barrier()

    # drop_last=True: truncate to a multiple of (world_size * batch_size) so every rank
    # processes the same number of batches — FSDP AllGather deadlock cannot occur.
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

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        low_cpu_mem_usage=True,
    )
    if any(k in args.model.lower() for k in ("nemotron", "bamba", "mamba")):
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # Capture experiments dict before FSDP wrapping.
    # Every Bamba/NemotronH/Mamba2 layer stores a reference to this same dict,
    # so mutations made in eval_windows are immediately visible to all layers.
    experiments = getattr(model.config, "experiments", None)

    layer_types = infer_transformer_layer_types(model) or set()
    fsdp_kwargs = {}
    if args.cpu_offload:
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()

    # FSDP2: shard leaf transformer layers first, then the root (bottom-up)
    for module in model.modules():
        if type(module) in layer_types:
            fully_shard(module, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)
    model.to(device)  # each rank holds only 1/world_size of params after sharding
    fsdp_model = model

    try:
        print(f"[rank {rank}] Starting evaluation ...")
        total_nll, total_tok = eval_windows(args, fsdp_model, dataloader, device, rank,
                                            experiments=experiments)
        if rank == 0:
            if total_tok == 0:
                print("PPL: N/A (no windows scored — check window size vs dataset lengths)")
            else:
                ppl = torch.exp(total_nll / total_tok.double()).item()
                n_windows = total_tok.item() // 100  # k_tail = 100 tokens per window
                print(f"[Perplexity]  window={args.max_length}  stride={args.stride}  "
                      f"ppl={ppl:.4f}  scored_windows={n_windows}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
