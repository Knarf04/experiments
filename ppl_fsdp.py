import os
import sys
import argparse
import math
from typing import Dict, Iterable
import functools

import torch
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)

import datasets
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, CPUOffload
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


# CHANGED: add feature_name, world_size, rank, local_sample_size and DDP-aware sharding
def build_streaming_iterable(
    hf_iterable,
    tokenizer,
    max_length,
    feature_name: str,
    world_size: int,
    rank: int,
    local_sample_size: int,
) -> Iterable[Dict[str, list]]:
    processed = 0
    for idx, sequence in enumerate(hf_iterable):
        # simple per-sample sharding by index
        if idx % world_size != rank:
            continue

        if local_sample_size > 0 and processed >= local_sample_size:
            break

        text = sequence[feature_name]  # CHANGED: use configurable feature, not hard-coded "text"
        tok = tokenizer(
            text,
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
        )
        yield {
            "input_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"],
            "tokenized_len": len(tok["input_ids"]),
        }
        processed += 1


def build_dataloader(args, tokenizer, rank, world_size, local_sample_size) -> DataLoader:
    if args.streaming:
        raw = datasets.load_dataset(
            args.dataset,
            name=args.subset,
            split=args.split,
            streaming=True,
            trust_remote_code=True,
        )

        # CHANGED: remove .shard(); not supported in streaming mode
        # raw = raw.shard(num_shards=world_size, index=rank)

        if args.shuffle_buffer > 0:
            raw = raw.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

        # CHANGED: wrap HF iterable with our DDP-aware, tokenizing generator
        tokenized_iterable = datasets.IterableDataset.from_generator(
            lambda: build_streaming_iterable(
                hf_iterable=raw,
                tokenizer=tokenizer,
                max_length=args.max_length,
                feature_name=args.feature,
                world_size=world_size,
                rank=rank,
                local_sample_size=local_sample_size,
            )
        )

        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        loader = DataLoader(
            tokenized_iterable,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,  # CHANGED: generator + multiprocessing is messy; keep 0
            pin_memory=True,
            collate_fn=collator,
        )
        return loader

    else:
        raw = datasets.load_dataset(
            args.dataset,
            name=args.subset,
            split=args.split,
            trust_remote_code=True,
        )

        def tok_fn(batch):
            texts = batch[args.feature]
            tok = tokenizer(
                texts,
                add_special_tokens=False,
                padding="max_length",
                truncation=True,
                max_length=args.max_length,
                return_attention_mask=True,
            )
            return {
                "input_ids": tok["input_ids"],
                "attention_mask": tok["attention_mask"],
                "tokenized_len": [len(x) for x in tok["input_ids"]],
            }

        remove_cols = [c for c in raw.column_names if c not in {args.feature}]
        tokenized = raw.map(
            tok_fn,
            batched=True,
            num_proc=args.num_proc,
            remove_columns=remove_cols,
            desc="Tokenizing (in-memory)",
        )

        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(
            tokenized, rank=rank, num_replicas=world_size, shuffle=True, drop_last=False
        )

        loader = DataLoader(
            tokenized,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collator,
            persistent_workers=False if args.num_workers == 0 else True,
        )
        return loader


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


def synchronized_dataloader_iterator(dataloader, rank):
    iterator = iter(dataloader)
    device = torch.device(f"cuda:{rank}")
    world_size = dist.get_world_size()

    while True:
        has_local_data = 1
        try:
            batch = next(iterator)
        except StopIteration:
            has_local_data = 0

        has_data_tensor = torch.tensor([has_local_data], dtype=torch.int, device=device)
        dist.all_reduce(has_data_tensor, op=dist.ReduceOp.SUM)

        # Stop as soon as any rank runs out — uneven load causes FSDP AllGather deadlock
        if has_data_tensor.item() < world_size:
            break

        yield batch


@torch.no_grad()
def sliding_window_ppl(args, model, dataloader, rank, experiments=None):
    device = torch.device(f"cuda:{rank}")
    max_amount_of_windows = 10
    k_tail = 100

    total_nll = torch.zeros(len(args.lengths), device=device, dtype=torch.float64)
    total_tok = torch.zeros(len(args.lengths), device=device, dtype=torch.long)

    pbar = tqdm(synchronized_dataloader_iterator(dataloader, rank), desc="batches", disable=(rank != 0))
    for batch in pbar:
        batch_input_ids      = batch["input_ids"].to(device, non_blocking=True)       # [B, max_len]
        batch_attention_mask = batch["attention_mask"].to(device, non_blocking=True)  # [B, max_len]
        seq_len = batch_input_ids.size(1)  # always max_length — same on all ranks ✓

        for i, L in enumerate(args.lengths):
            window_size = int(L)
            if seq_len < window_size:
                continue  # consistent across all ranks ✓
            stride = max(10, (seq_len - window_size) // max_amount_of_windows)
            # stride is deterministic from seq_len — same on all ranks ✓

            for begin_loc in range(0, seq_len - window_size + 1, stride):
                end_loc = begin_loc + window_size
                input_ids      = batch_input_ids[:, begin_loc:end_loc]      # [B, W]
                attention_mask = batch_attention_mask[:, begin_loc:end_loc]  # [B, W]

                # A sample is valid only if the entire window contains no padding
                valid_samples = attention_mask.all(dim=1)  # [B] bool

                # Tell model which samples are valid (for logit/state recording in Bamba layers)
                if experiments is not None:
                    experiments["valid_mask"] = valid_samples.cpu().tolist()

                # Always call model — FSDP requires identical collectives on all ranks ✓
                if "zamba2" in args.model.lower():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                    num_logits_to_keep=k_tail + 1)
                elif any(k in args.model.lower() for k in ("bamba", "nemotron", "mamba")):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                    num_logits_to_keep=k_tail + 1, use_cache=False)
                else:
                    inference_params = {
                        "max_seqlen": window_size + 1,
                        "max_batch_size": input_ids.shape[0],
                        "seqlen_offset": 0,
                        "batch_size_offset": 0,
                    }
                    outputs, _ = model(input_ids, attention_mask=attention_mask,
                                       num_last_tokens=k_tail + 1,
                                       inference_params=inference_params)

                # Discard results if no sample in the batch is fully padding-free
                if not valid_samples.any():
                    continue

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

                # Accumulate only for valid (fully padding-free) samples
                valid_logits = shift_logits[valid_samples, -k_eff:, :]  # [n_valid, k_eff, V]
                valid_labels = shift_labels[valid_samples, -k_eff:]     # [n_valid, k_eff]

                nll_sum = F.cross_entropy(
                    valid_logits.reshape(-1, V),
                    valid_labels.reshape(-1),
                    reduction="sum",
                )
                total_nll[i] += nll_sum.double()
                total_tok[i] += valid_samples.sum().item() * k_eff

    dist.all_reduce(total_nll, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tok, op=dist.ReduceOp.SUM)
    return total_nll, total_tok


def main():
    parser = argparse.ArgumentParser()
    # Model & precision
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 autocast (recommended on Ampere/ADA/Hopper).")
    # Dataset
    parser.add_argument("--dataset", type=str, required=True, help='HF dataset id or local script, e.g. "pg19"')
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--sample-size", type=int, default=0, help="Number of data points sampled from the dataset.")
    parser.add_argument("--feature", type=str, default="text", help="Text column name")
    parser.add_argument("--streaming", action="store_true", help="Use HF streaming (recommended for large datasets).")
    parser.add_argument("--shuffle-buffer", type=int, default=0, help="Streaming shuffle buffer size; 0 = no shuffle.")
    parser.add_argument("--max-length", type=int, default=2048, help="Truncate each sample to this length.")
    # Loader
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-proc", type=int, default=4, help="Tokenize map() workers (non-streaming mode).")
    parser.add_argument("--seed", type=int, default=42)
    # FSDP
    parser.add_argument("--cpu-offload", action="store_true", help="FSDP CPU parameter offload (slower but lighter).")

    args = parser.parse_args()

    # Distributed init
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = getattr(tokenizer, "eos_token", tokenizer.unk_token)
    tokenizer.padding_side = "right"

    # lengths = [131072, 65536, 32768, 16384, 8192, 4096, 2048]
    # args.lengths = [int(L) for L in lengths if int(L) <= args.max_length]
    args.lengths = [args.max_length]

    local_sample_size = args.sample_size // world_size + int(rank < args.sample_size % world_size)

    # DataLoader (on-the-fly tokenization)
    dataloader = build_dataloader(args, tokenizer, rank, world_size, local_sample_size)

    # Model
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
    # Some models keep large KV cache flags on; not needed for loss-only eval
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # Grab the experiments dict before FSDP moves the module.
    # self.experiments in each Bamba layer is a reference to this same dict,
    # so mutations made between forward calls remain visible to all layers.
    experiments = getattr(model.config, "experiments", None)

    # FSDP wrapping
    # CHANGED: use infer_transformer_layer_types to make this robust
    layer_types = infer_transformer_layer_types(model) or set()
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=layer_types,
    )

    fsdp_model = FSDP(
        model,
        device_id=torch.device(f"cuda:{rank}"),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        cpu_offload=CPUOffload(offload_params=True) if args.cpu_offload else None,
        auto_wrap_policy=auto_wrap_policy,
        limit_all_gathers=True,
    )

    try:
        print("Starting evaluation...")
        total_nll, total_tok = sliding_window_ppl(args, fsdp_model, dataloader, rank,
                                                  experiments=experiments)

        if rank == 0:
            print("[Perplexity]")
            for i, L in enumerate(args.lengths):
                if total_tok[i] == 0:
                    print(f"{L}: N/A (no valid windows — all samples shorter than window size)")
                else:
                    ppl = torch.exp(total_nll[i] / total_tok[i].double()).item()
                    print(f"{L}: {ppl:.2f}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
