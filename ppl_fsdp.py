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
def eval_windows(args, model, dataloader, device, rank, window_size, experiments=None):
    """Score tail-perplexity on pre-extracted fixed-length windows.

    Every window is exactly window_size real tokens with no padding, so no
    attention_mask or validity filtering is needed.
    DistributedSampler + drop_last=True guarantees all ranks execute the same
    number of forward passes — FSDP-safe by construction.
    """
    k_tail = 100

    total_nll = torch.zeros(1, device=device, dtype=torch.float64)
    total_tok = torch.zeros(1, device=device, dtype=torch.long)

    import sys
    batch_idx = 0
    pbar = tqdm(dataloader, desc=f"windows (len={window_size})", disable=(rank != 0))
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)  # [B, L] — real tokens only ✓
        B = input_ids.size(0)
        print(f"[DIAG][rank={rank}] batch={batch_idx} shape={list(input_ids.shape)} "
              f"mem={torch.cuda.memory_allocated(device)/1e9:.2f}GB", file=sys.stderr, flush=True)
        batch_idx += 1

        # Every window in this batch is padding-free — inform recording layers
        if experiments is not None:
            experiments["valid_mask"] = [True] * B

        model_type = getattr(model.config, "model_type", "").lower()
        model_id = args.model.lower()
        if "zamba2" in model_type or "zamba2" in model_id:
            outputs = model(input_ids=input_ids, num_logits_to_keep=k_tail + 1)
        elif any(k in model_type for k in ("bamba", "nemotron", "mamba")) or \
             any(k in model_id for k in ("bamba", "nemotron", "mamba")):
            outputs = model(input_ids=input_ids, num_logits_to_keep=k_tail + 1, use_cache=False)
        else:
            inference_params = {
                "max_seqlen": window_size + 1,
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


def _install_sigbus_handler():
    """Register a handler that prints a Python traceback on SIGBUS before dying."""
    import signal, sys, traceback, faulthandler
    faulthandler.enable(file=sys.stderr, all_threads=True)

    def _handler(signum, frame):
        print(f"\n{'='*60}", file=sys.stderr, flush=True)
        print(f"CAUGHT SIGNAL {signum} (SIGBUS) — Python traceback:", file=sys.stderr, flush=True)
        traceback.print_stack(frame, file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr, flush=True)
        sys.stderr.flush()
        # Re-raise so the process still exits with the right code
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    signal.signal(signal.SIGBUS, _handler)


def main():
    _install_sigbus_handler()
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
    parser.add_argument("--seq-len", type=int, nargs="+", default=[2048],
                        help="Window size(s) in tokens. Pass multiple values to sweep, "
                             "e.g. --seq-len 2048 4096 8192.")
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
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
    )
    if any(k in args.model.lower() for k in ("nemotron", "bamba", "mamba")):
        model_kwargs["attn_implementation"] = "flash_attention_2"

    import sys
    print(f"[DIAG][rank={rank}] Loading model...", file=sys.stderr, flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    print(f"[DIAG][rank={rank}] Model loaded.", file=sys.stderr, flush=True)

    # Capture experiments dict before FSDP wrapping.
    # Every Bamba/NemotronH/Mamba2 layer stores a reference to this same dict,
    # so mutations made in eval_windows are immediately visible to all layers.
    experiments = getattr(model.config, "experiments", None)

    layer_types = infer_transformer_layer_types(model) or set()
    fsdp_kwargs = {}
    if args.cpu_offload:
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()

    # FSDP2: shard leaf transformer layers first, then the root (bottom-up)
    print(f"[DIAG][rank={rank}] Sharding FSDP (layer_types={[t.__name__ for t in layer_types]})...", file=sys.stderr, flush=True)
    for module in model.modules():
        if type(module) in layer_types:
            fully_shard(module, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)
    if not args.cpu_offload:
        model.to(device)  # each rank holds only 1/world_size of params after sharding
    fsdp_model = model
    print(f"[DIAG][rank={rank}] FSDP ready. GPU mem={torch.cuda.memory_allocated(device)/1e9:.2f}GB", file=sys.stderr, flush=True)

    # Iterate over each requested sequence length
    results = []  # list of (window_size, stride, ppl, n_windows)

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

            if rank == 0:
                print(f"[rank {rank}] Evaluating window_size={window_size} ...")
            total_nll, total_tok = eval_windows(args, fsdp_model, dataloader, device, rank,
                                                window_size=window_size, experiments=experiments)

            if rank == 0:
                if total_tok == 0:
                    print(f"  window={window_size}: N/A (no windows scored)")
                    results.append((window_size, stride, None, 0))
                else:
                    ppl = torch.exp(total_nll / total_tok.double()).item()
                    n_windows = total_tok.item() // 100  # k_tail = 100 tokens per window
                    print(f"  [Perplexity]  window={window_size}  stride={stride}  "
                          f"ppl={ppl:.4f}  scored_windows={n_windows}")
                    results.append((window_size, stride, ppl, n_windows))

        # Print summary table
        if rank == 0 and len(results) > 1:
            print(f"\n{'='*60}")
            print(f"  PPL Summary: {args.model}")
            print(f"{'='*60}")
            print(f"  {'Window':>8}  {'Stride':>8}  {'PPL':>10}  {'Windows':>8}")
            print(f"  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*8}")
            for window_size, stride, ppl, n_windows in results:
                ppl_str = f"{ppl:.4f}" if ppl is not None else "N/A"
                print(f"  {window_size:>8}  {stride:>8}  {ppl_str:>10}  {n_windows:>8}")
            print(f"{'='*60}")

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
