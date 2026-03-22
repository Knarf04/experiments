"""
Test MiniKV through the HF adapter path (same path as RULER).
Compares HF adapter generate vs FMS generate on the same model.

Usage:
    python experiments/test_minikv_hf.py \
        --model_path /path/to/checkpoint.pth \
        --variant llama_1b_snapKV \
        --tokenizer /gpfs/hshen/tokenizer/llama3
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
from fms.models.hf.llama.modeling_llama_hf import HFAdaptedLLaMAForCausalLM, HFAdaptedLLaMAConfig
from fms_fsdp.utils.config_utils import get_model_config
from fms.utils.minikv.cache import EvictedKVCache


def load_fms_model(variant, model_path):
    config = get_model_config(variant)
    model = LLaMA(config)
    if model_path.endswith('.pth'):
        ckpt = torch.load(model_path, map_location="cpu")
        sd = ckpt.get("model_state", ckpt)
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd)
    model.to(dtype=torch.bfloat16, device='cuda')
    model.eval()
    return model, config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--variant", type=str, default="llama_1b_snapKV")
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=512)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    # Build input
    prompt = "The capital of France is"
    tokens = tokenizer.encode(prompt, return_tensors="pt").to('cuda')
    if args.seq_len > tokens.shape[1]:
        filler = tokenizer.encode("This is filler text. " * 100, return_tensors="pt").to('cuda')
        while filler.shape[1] < args.seq_len - tokens.shape[1]:
            filler = torch.cat([filler, filler], dim=1)
        filler = filler[:, :args.seq_len - tokens.shape[1]]
        input_ids = torch.cat([filler, tokens], dim=1)
    else:
        input_ids = tokens
    print(f"Input shape: {input_ids.shape}")

    # ================================================================
    # Load FMS model with MiniKV
    # ================================================================
    print("\nLoading FMS model with MiniKV...")
    fms_model, config = load_fms_model(args.variant, args.model_path)
    print(f"  kv_eviction={config.kv_eviction}")
    print(f"  _minikv_kwargs={'YES' if fms_model._minikv_kwargs else 'NO'}")

    # ================================================================
    # TEST 1: FMS generate (works correctly)
    # ================================================================
    print("\n" + "=" * 60)
    print("TEST 1: FMS generate (direct, no HF adapter)")
    print("=" * 60)
    from fms.utils.generation import generate as fms_generate
    from fms.utils.minikv.attention_op import MiniKVConfig, create_minikv_kwargs

    extra_kwargs = create_minikv_kwargs(
        MiniKVConfig(
            selection_method=config.kv_eviction,
            heavy_ratio=config.kv_eviction_heavy_ratio,
            recent_ratio=config.kv_eviction_recent_ratio,
            window_size=config.kv_eviction_window_size,
            prompt_sparsity_ratio=config.kv_eviction_sparsity_ratio,
            kernel_size=config.kv_eviction_kernel_size,
            pooling=config.kv_eviction_pooling,
        ),
        num_layers=config.nlayers,
    )
    torch.set_grad_enabled(False)
    fms_result = fms_generate(
        fms_model, input_ids, max_new_tokens=20, use_cache=True, do_sample=False,
        extra_kwargs=extra_kwargs,
    )
    fms_text = tokenizer.decode(fms_result[0, input_ids.shape[1]:])
    print(f"  FMS generate: '{fms_text}'")

    # ================================================================
    # TEST 2: HF adapter generate (same path as RULER)
    # ================================================================
    print("\n" + "=" * 60)
    print("TEST 2: HF adapter generate (same path as RULER)")
    print("=" * 60)

    from transformers.modeling_utils import no_init_weights
    fms_hf_config = HFAdaptedLLaMAConfig.from_fms_config(fms_model.get_config())
    with no_init_weights():
        hf_model = HFAdaptedLLaMAForCausalLM.from_fms_model(fms_model, **fms_hf_config.to_dict())
    hf_model.eval()

    # Check: does the inner model still have _minikv_kwargs?
    inner = hf_model.decoder.model
    print(f"  inner model type: {type(inner).__name__}")
    print(f"  inner._minikv_kwargs: {'YES' if hasattr(inner, '_minikv_kwargs') and inner._minikv_kwargs else 'NO'}")
    print(f"  inner is fms_model: {inner is fms_model}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    hf_inputs = tokenizer.decode(input_ids[0].tolist())
    hf_encoded = tokenizer(hf_inputs, return_tensors="pt", padding=True).to('cuda')

    hf_result = hf_model.generate(
        **hf_encoded, max_new_tokens=20, do_sample=False, use_cache=True,
    )
    hf_text = tokenizer.decode(hf_result[0, hf_encoded['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"  HF generate:  '{hf_text}'")

    # ================================================================
    # TEST 3: HF adapter single forward (prefill only)
    # ================================================================
    print("\n" + "=" * 60)
    print("TEST 3: HF adapter single forward vs FMS forward")
    print("=" * 60)

    # FMS forward
    fms_extra = create_minikv_kwargs(
        MiniKVConfig(
            selection_method=config.kv_eviction,
            prompt_sparsity_ratio=config.kv_eviction_sparsity_ratio,
        ),
        num_layers=config.nlayers,
    )
    with torch.no_grad():
        fms_out = fms_model(input_ids, use_cache=True, **fms_extra)
    fms_logits = fms_out[0]
    fms_cache = fms_out[1]

    # HF forward
    with torch.no_grad():
        hf_out = hf_model(input_ids=input_ids, use_cache=True, return_dict=True)
    hf_logits = hf_out.logits if hasattr(hf_out, 'logits') else hf_out[0]
    hf_cache = hf_out.past_key_values if hasattr(hf_out, 'past_key_values') else hf_out[1]

    print(f"  FMS logits: shape={list(fms_logits.shape)}, nan={fms_logits.isnan().any().item()}")
    print(f"  HF logits:  shape={list(hf_logits.shape)}, nan={hf_logits.isnan().any().item()}")

    logit_diff = (fms_logits - hf_logits).abs().max().item()
    print(f"  Max logit diff (FMS vs HF): {logit_diff:.2e}")

    # Check cache types
    print(f"\n  FMS cache[0][0] type: {type(fms_cache[0][0]).__name__}")
    if hf_cache:
        print(f"  HF cache type: {type(hf_cache).__name__}")
        if hasattr(hf_cache, '__len__'):
            print(f"  HF cache len: {len(hf_cache)}")
            if len(hf_cache) > 0:
                layer0 = hf_cache[0]
                print(f"  HF cache[0] type: {type(layer0).__name__}")
                if hasattr(layer0, '__len__'):
                    print(f"  HF cache[0] len: {len(layer0)}")
                    for j, item in enumerate(layer0):
                        print(f"    HF cache[0][{j}] type: {type(item).__name__}, "
                              f"is EvictedKVCache: {isinstance(item, EvictedKVCache)}")
                        if isinstance(item, EvictedKVCache):
                            print(f"      keys shape: {item.keys.shape}, logical: {item.seq_len_logical}")
                        elif isinstance(item, torch.Tensor):
                            print(f"      shape: {item.shape}")

    # Next token comparison
    fms_next = fms_logits[0, -1, :].argmax().item()
    hf_next = hf_logits[0, -1, :].argmax().item()
    print(f"\n  FMS next token: {fms_next} = '{tokenizer.decode([fms_next])}'")
    print(f"  HF  next token: {hf_next} = '{tokenizer.decode([hf_next])}'")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  FMS generate: '{fms_text}'")
    print(f"  HF generate:  '{hf_text}'")
    print(f"  Match: {fms_text == hf_text}")


if __name__ == "__main__":
    main()
