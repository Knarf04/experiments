import os
import sys
import argparse
import math
from typing import Dict, Iterable
import functools

import torch
import torch.distributed as dist
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


def build_streaming_iterable(hf_iterable, tokenizer, max_length) -> Iterable[Dict[str, list]]:
    for sequence in hf_iterable:
        tok = tokenizer(
            sequence["text"],
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
        )
        yield {
            "input_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"],
            "tokenized_len": len(tok["input_ids"])
        }

def build_dataloader(args, tokenizer, rank, world_size, local_sample_size) -> DataLoader:
    if args.streaming:
        raw = datasets.load_dataset(
            args.dataset,
            name=args.subset,
            split=args.split,
            streaming=True,
            trust_remote_code=True,
        )
        raw = raw.shard(num_shards=world_size, index=rank)
        if args.shuffle_buffer > 0:
            raw = raw.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

        tokenized_iterable = datasets.IterableDataset.from_generator(
            lambda: build_streaming_iterable(raw, tokenizer, args.max_length)
        )
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        loader = DataLoader(
            tokenized_iterable,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
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
            tok = tokenizer(
                batch["text"],
                add_special_tokens=False,
                padding="max_length",
                truncation=True,
                max_length=args.max_length,
                return_attention_mask=True,
            )
            return {
                "input_ids": tok["input_ids"],
                "attention_mask": tok["attention_mask"],
                "tokenized_len": len(tok["input_ids"])
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
    base = getattr(model, getattr(model, "base_model_prefix", ""), model)
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

    while True:
        batch = None
        has_local_data = 1
        try:
            batch = next(iterator)
        except StopIteration:
            has_local_data = 0

        has_data_tensor = torch.tensor([has_local_data], dtype=torch.int, device=device)
        dist.all_reduce(has_data_tensor, op=dist.ReduceOp.SUM)

        if has_data_tensor.item() == 0:
            break

        yield batch

@torch.no_grad()
def sliding_window_ppl(args, model, dataloader, rank):
    device = torch.device(f"cuda:{rank}")
    max_amount_of_windows = 10
    k_tail = 100

    ppls = torch.zeros(len(args.lengths), device=device)
    valid_ppls = torch.zeros(len(args.lengths), device=device)

    for _, batch in enumerate(synchronized_dataloader_iterator(dataloader, rank)):
        if batch is None:
            continue

        batch_input_ids = batch["input_ids"].to(device, non_blocking=True)
        seq_len = batch_input_ids.size(1)

        for i, L in enumerate(args.lengths):
            window_size = int(L)
            if seq_len < window_size:
                continue

            stride = max(10, (seq_len - window_size) // max_amount_of_windows)
            nlls = []

            for begin_loc in range(0, seq_len - window_size, stride):
                end_loc = begin_loc + window_size
                input_ids = batch_input_ids[:, begin_loc:end_loc]  # [B, W]

                if "zamba2" in args.model.lower():
                    outputs = model(input_ids, num_logits_to_keep=k_tail + 1)
                elif any(k in args.model.lower() for k in ("bamba", "nemotron", "mamba")):
                    outputs = model(input_ids, num_logits_to_keep=k_tail + 1, use_cache=False)
                else:
                    inference_params = {
                        "max_seqlen": window_size + 1,
                        "max_batch_size": input_ids.shape[0],
                        "seqlen_offset": 0,
                        "batch_size_offset": 0,
                    }
                    outputs, _ = model(input_ids, num_last_tokens=k_tail + 1, inference_params=inference_params)

                logits = outputs.logits  # expected [B, T, V]; T may be <= window_size or == k_tail+1
                B, T_logits, V = logits.shape
                T_in = input_ids.size(1)

                T = min(T_logits, T_in)
                logits = logits[:, :T, :]            # [B, T, V]
                labels_all = input_ids[:, :T]        # [B, T]

                logits_shift = logits[:, :-1, :]     # [B, T-1, V]
                labels_shift = labels_all[:, 1:]     # [B, T-1]
                k_eff = min(k_tail, logits_shift.size(1))
                if k_eff == 0:
                    continue

                logits_tail = logits_shift[:, -k_eff:, :].contiguous()   # [B, k, V]
                labels_tail = labels_shift[:, -k_eff:].contiguous()      # [B, k]

                loss = F.cross_entropy(
                    logits_tail.view(-1, V),
                    labels_tail.view(-1),
                    reduction="mean",
                )
                nlls.append(loss)

            if nlls:
                ppls[i] += torch.exp(torch.stack(nlls).mean()).to(torch.float32)
                valid_ppls[i] += 1

        print(ppls/(valid_ppls+1e-6))
        dist.barrier()

    dist.all_reduce(ppls, op=dist.ReduceOp.SUM)
    dist.all_reduce(valid_ppls, op=dist.ReduceOp.SUM)
    return ppls, valid_ppls


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
    tokenizer.padding_side = "left"

    # lengths = [131072, 65536, 32768, 16384, 8192, 4096, 2048]
    # args.lengths = [int(L) for L in lengths if int(L) <= args.max_length]
    args.lengths = [2048]
    
    local_sample_size = args.sample_size // world_size + int(rank < args.sample_size % world_size)

    # DataLoader (on-the-fly tokenization)
    dataloader = build_dataloader(args, tokenizer, rank, world_size, local_sample_size)

    # Model
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    # Some models keep large KV cache flags on; not needed for loss-only eval
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # FSDP wrapping
    base = getattr(model, model.base_model_prefix)
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={type(layer) for layer in base.layers},
    )

    fsdp_model = FSDP(
        model,
        device_id=torch.device(f"cuda:{rank}"),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        cpu_offload=CPUOffload(offload_params=True) if args.cpu_offload else None,
        auto_wrap_policy=auto_wrap_policy,
        limit_all_gathers=True
    )

    try:
        ppls, valid_ppls = sliding_window_ppl(args, fsdp_model, dataloader, rank)
        ppls = ppls / valid_ppls

        if rank == 0:
            print("[Perplexity]")
            for i, ppl in enumerate(ppls):
                print(f"{args.lengths[i]}: {ppl}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
