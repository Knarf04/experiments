"""
A/B test: compare real model output with and without MiniKV on the same input.
Tests whether prefill logits match (they should) and where decode diverges.

Usage:
    python experiments/test_minikv_real.py \
        --model_path /path/to/checkpoint.pth \
        --variant llama_1b_snapKV \
        --tokenizer meta-llama/Llama-3.2-1B
"""

import argparse
import sys
import os
import copy

import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "foundation-model-stack-sandbox"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "fms-fsdp"))

from fms.models.llama import LLaMA, LLaMAConfig
from fms_fsdp.utils.config_utils import get_model_config


def load_model(variant, model_path):
    config = get_model_config(variant)
    # Force no eviction for loading
    config_no_eviction = copy.deepcopy(config)
    config_no_eviction.kv_eviction = None

    model = LLaMA(config_no_eviction)
    if model_path.endswith('.pth'):
        ckpt = torch.load(model_path, map_location="cpu")
        sd = ckpt.get("model_state", ckpt)
        # Strip _orig_mod. prefix from compiled checkpoints
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd)
    model.to(dtype=torch.bfloat16, device='cuda')
    model.eval()
    return model, config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--variant", type=str, default="llama_1b_snapKV")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--prompt", type=str, default="The capital of France is")
    parser.add_argument("--seq_len", type=int, default=512,
                        help="Pad prompt to this length with repeated filler")
    args = parser.parse_args()

    print("Loading model...")
    model, config = load_model(args.variant, args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    print(f"Model loaded. Config kv_eviction: {config.kv_eviction}")

    # Create input
    tokens = tokenizer.encode(args.prompt, return_tensors="pt").to('cuda')
    prompt_len = tokens.shape[1]

    # Optionally pad to seq_len with filler
    if args.seq_len > prompt_len:
        filler = tokenizer.encode("This is filler text. " * 100, return_tensors="pt").to('cuda')
        # Repeat filler to reach target length
        while filler.shape[1] < args.seq_len - prompt_len:
            filler = torch.cat([filler, filler], dim=1)
        filler = filler[:, :args.seq_len - prompt_len]
        input_ids = torch.cat([filler, tokens], dim=1)
    else:
        input_ids = tokens

    print(f"Input shape: {input_ids.shape}")

    # ================================================================
    # TEST A: Baseline (no eviction)
    # ================================================================
    print("\n" + "=" * 60)
    print("TEST A: Baseline (no eviction, standard SDPA)")
    print("=" * 60)

    with torch.no_grad():
        baseline_out = model(input_ids, use_cache=True)
    baseline_logits = baseline_out[0]
    baseline_cache = baseline_out[1]

    last_logit_baseline = baseline_logits[0, -1, :]
    next_token_baseline = last_logit_baseline.argmax().item()
    print(f"  Prefill logits: nan={baseline_logits.isnan().any().item()}, "
          f"shape={list(baseline_logits.shape)}")
    print(f"  Next token: {next_token_baseline} = '{tokenizer.decode([next_token_baseline])}'")
    print(f"  Top-5: {torch.topk(last_logit_baseline, 5).indices.tolist()}")
    print(f"  Cache[0] shape: {baseline_cache[0][0].shape}")

    # Generate 20 tokens
    from fms.utils.generation import generate
    baseline_gen = generate(
        model, input_ids, max_new_tokens=20, use_cache=True, do_sample=False,
    )
    baseline_text = tokenizer.decode(baseline_gen[0, input_ids.shape[1]:])
    print(f"  Generated: '{baseline_text}'")

    # ================================================================
    # TEST B: With MiniKV eviction
    # ================================================================
    print("\n" + "=" * 60)
    print(f"TEST B: MiniKV ({config.kv_eviction}, sparsity={config.kv_eviction_sparsity_ratio})")
    print("=" * 60)

    # Create model with eviction
    from fms.utils.minikv.attention_op import MiniKVConfig, create_minikv_kwargs
    minikv_config = MiniKVConfig(
        selection_method=config.kv_eviction,
        heavy_ratio=config.kv_eviction_heavy_ratio,
        recent_ratio=config.kv_eviction_recent_ratio,
        window_size=config.kv_eviction_window_size,
        prompt_sparsity_ratio=config.kv_eviction_sparsity_ratio,
        kernel_size=config.kv_eviction_kernel_size,
        pooling=config.kv_eviction_pooling,
    )
    extra_kwargs = create_minikv_kwargs(minikv_config, num_layers=config.nlayers)

    with torch.no_grad():
        minikv_out = model(input_ids, use_cache=True, **extra_kwargs)
    minikv_logits = minikv_out[0]
    minikv_cache = minikv_out[1]

    last_logit_minikv = minikv_logits[0, -1, :]
    next_token_minikv = last_logit_minikv.argmax().item()
    print(f"  Prefill logits: nan={minikv_logits.isnan().any().item()}, "
          f"shape={list(minikv_logits.shape)}")
    print(f"  Next token: {next_token_minikv} = '{tokenizer.decode([next_token_minikv])}'")
    print(f"  Top-5: {torch.topk(last_logit_minikv, 5).indices.tolist()}")

    from fms.utils.minikv.cache import EvictedKVCache
    kv0 = minikv_cache[0][0]
    if isinstance(kv0, EvictedKVCache):
        print(f"  Cache[0]: physical={kv0.keys.shape[2]}, logical={kv0.seq_len_logical}")
    else:
        print(f"  Cache[0] type: {type(kv0).__name__} — NOT EvictedKVCache!")

    # ================================================================
    # COMPARISON
    # ================================================================
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    # Prefill logits should be IDENTICAL (eviction only affects cache)
    prefill_diff = (baseline_logits - minikv_logits).abs().max().item()
    print(f"  Prefill max logit diff: {prefill_diff:.2e}")
    if prefill_diff < 1e-3:
        print(f"  PASS: Prefill logits match")
    else:
        print(f"  FAIL: Prefill logits DIFFER — bug in compute_prefill_op")
        # Find where they differ
        diff = (baseline_logits - minikv_logits).abs()
        print(f"    Mean diff: {diff.mean().item():.2e}")
        print(f"    Diff at last token: {diff[0, -1, :].max().item():.2e}")
        print(f"    Diff at first token: {diff[0, 0, :].max().item():.2e}")

    print(f"  Next token match: {next_token_baseline == next_token_minikv}")
    print(f"    Baseline: {next_token_baseline} = '{tokenizer.decode([next_token_baseline])}'")
    print(f"    MiniKV:   {next_token_minikv} = '{tokenizer.decode([next_token_minikv])}'")

    # Generate with MiniKV using FMS generate (bypasses HF adapter)
    print("\n  --- FMS generate (not HF adapter) ---")
    extra_kwargs2 = create_minikv_kwargs(minikv_config, num_layers=config.nlayers)
    minikv_gen = generate(
        model, input_ids, max_new_tokens=20, use_cache=True, do_sample=False,
        extra_kwargs=extra_kwargs2,
    )
    minikv_text = tokenizer.decode(minikv_gen[0, input_ids.shape[1]:])
    print(f"  Baseline: '{baseline_text}'")
    print(f"  MiniKV:   '{minikv_text}'")


if __name__ == "__main__":
    main()
