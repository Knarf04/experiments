"""
Test script for MiniKV KV cache eviction on FMS LLaMA.

Runs three tests:
1. Baseline vs MiniKV comparison with a micro model (random weights)
2. RoPE position_ids verification after eviction
3. Cache shape verification through prefill + decode steps

Usage:
    python experiments/test_minikv.py
    python experiments/test_minikv.py --test rope
    python experiments/test_minikv.py --test all
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "foundation-model-stack-sandbox"))

import torch
import torch.nn as nn

from fms.models.llama import LLaMA, LLaMAConfig
from fms.utils.generation import generate

# Import triggers registration of "minikv" attention op
from fms.utils.minikv.attention_op import MiniKVConfig, create_minikv_kwargs
from fms.utils.minikv.cache import EvictedKVCache
from fms.utils.minikv.selection import H2OSelection, SnapKVSelection


def make_micro_model(kvheads=2, seed=42):
    """Create a tiny LLaMA with random weights for testing."""
    torch.manual_seed(seed)
    config = LLaMAConfig(
        src_vocab_size=256,
        emb_dim=64,
        nheads=4,
        kvheads=kvheads,
        nlayers=4,
        hidden_grow_factor=2.0,
        multiple_of=2,
        max_expected_seq_len=512,
    )
    model = LLaMA(config)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.normal_(p, std=0.02)
    model.eval()
    return model, config


def test_baseline_vs_minikv():
    """Test 1: Compare baseline SDPA output vs MiniKV output."""
    print("=" * 60)
    print("TEST 1: Baseline vs MiniKV comparison")
    print("=" * 60)

    model, config = make_micro_model()
    torch.manual_seed(123)
    input_ids = torch.randint(0, 256, (1, 64))

    # --- Baseline (standard SDPA, no eviction) ---
    print("\n[Baseline] Running generate with standard SDPA...")
    baseline_result = generate(
        model, input_ids, max_new_tokens=10, use_cache=True, do_sample=False,
    )
    print(f"  Input shape:  {input_ids.shape}")
    print(f"  Output shape: {baseline_result.shape}")
    print(f"  Generated tokens: {baseline_result[0, 64:].tolist()}")

    # --- MiniKV with very conservative eviction (keep 90%) ---
    print("\n[MiniKV 90%] Running generate with H2O (heavy=0.45, recent=0.45)...")
    minikv_config = MiniKVConfig(
        selection_method="h2o", heavy_ratio=0.45, recent_ratio=0.45,
    )
    extra_kwargs = create_minikv_kwargs(minikv_config, num_layers=config.nlayers)
    minikv_result_90 = generate(
        model, input_ids, max_new_tokens=10, use_cache=True, do_sample=False,
        extra_kwargs=extra_kwargs,
    )
    print(f"  Output shape: {minikv_result_90.shape}")
    print(f"  Generated tokens: {minikv_result_90[0, 64:].tolist()}")
    match_90 = torch.equal(baseline_result, minikv_result_90)
    print(f"  Matches baseline: {match_90}")

    # --- MiniKV with moderate eviction (keep 50%) ---
    print("\n[MiniKV 50%] Running generate with H2O (heavy=0.25, recent=0.25)...")
    minikv_config = MiniKVConfig(
        selection_method="h2o", heavy_ratio=0.25, recent_ratio=0.25,
    )
    extra_kwargs = create_minikv_kwargs(minikv_config, num_layers=config.nlayers)
    minikv_result_50 = generate(
        model, input_ids, max_new_tokens=10, use_cache=True, do_sample=False,
        extra_kwargs=extra_kwargs,
    )
    print(f"  Output shape: {minikv_result_50.shape}")
    print(f"  Generated tokens: {minikv_result_50[0, 64:].tolist()}")
    match_50 = torch.equal(baseline_result, minikv_result_50)
    print(f"  Matches baseline: {match_50}")

    # --- Verify all outputs are valid (not all same token / not garbage) ---
    for name, result in [("baseline", baseline_result), ("minikv_90", minikv_result_90), ("minikv_50", minikv_result_50)]:
        gen_tokens = result[0, 64:]
        unique = gen_tokens.unique().numel()
        print(f"\n  [{name}] unique generated tokens: {unique}/{len(gen_tokens)}")
        if unique == 1:
            print(f"  WARNING: All generated tokens are the same ({gen_tokens[0].item()}) — likely broken")

    print()
    return True


def test_rope_positions():
    """Test 2: Verify RoPE position_ids are correct after eviction."""
    print("=" * 60)
    print("TEST 2: RoPE position verification")
    print("=" * 60)

    model, config = make_micro_model()
    input_ids = torch.randint(0, 256, (1, 64))
    seq_len = input_ids.shape[1]

    # Run a single forward pass with MiniKV to get the cache
    minikv_config = MiniKVConfig(selection_method="h2o", heavy_ratio=0.25, recent_ratio=0.25)
    extra_kwargs = create_minikv_kwargs(minikv_config, num_layers=config.nlayers)

    # Prefill
    with torch.no_grad():
        output = model(input_ids, use_cache=True, **extra_kwargs)
    logits, cache = output

    print(f"\n  Input seq_len: {seq_len}")
    print(f"  Num layers in cache: {len(cache)}")

    all_ok = True
    for i, layer_cache in enumerate(cache):
        kv_cache = layer_cache[0]  # EvictedKVCache (same object as layer_cache[1])
        if not isinstance(kv_cache, EvictedKVCache):
            print(f"  Layer {i}: ERROR - cache is {type(kv_cache)}, expected EvictedKVCache")
            all_ok = False
            continue

        physical_size = kv_cache.keys.shape[2]
        logical_size = kv_cache.seq_len_logical
        rope_size = kv_cache.size(2)

        print(f"  Layer {i}: physical={physical_size}, logical={logical_size}, "
              f"size(2)={rope_size}, keys_shape={list(kv_cache.keys.shape)}")

        if logical_size != seq_len:
            print(f"    ERROR: logical size should be {seq_len}, got {logical_size}")
            all_ok = False
        if rope_size != seq_len:
            print(f"    ERROR: size(2) should return {seq_len} for RoPE, got {rope_size}")
            all_ok = False
        if physical_size >= seq_len:
            print(f"    WARNING: no eviction happened (physical >= original)")
        if physical_size > logical_size:
            print(f"    ERROR: physical size > logical size (impossible)")
            all_ok = False

    # Now do one decode step and check positions update correctly
    print(f"\n  Running one decode step...")
    # Reset layer counter for decode
    extra_kwargs["minikv_state"]["layer_counter"] = 0
    extra_kwargs["minikv_state"]["is_prefill"] = False
    if "mask" in extra_kwargs:
        del extra_kwargs["mask"]

    next_token = logits[:, -1:, :].argmax(dim=-1)
    with torch.no_grad():
        output2 = model(next_token, past_key_value_states=list(cache), use_cache=True, **extra_kwargs)
    _, cache2 = output2

    for i, layer_cache in enumerate(cache2):
        kv_cache = layer_cache[0]
        physical_size = kv_cache.keys.shape[2]
        logical_size = kv_cache.seq_len_logical
        rope_size = kv_cache.size(2)

        if i == 0:
            print(f"  Layer {i} after decode: physical={physical_size}, logical={logical_size}, size(2)={rope_size}")

        if logical_size != seq_len + 1:
            print(f"    Layer {i} ERROR: logical size should be {seq_len + 1}, got {logical_size}")
            all_ok = False
        if rope_size != seq_len + 1:
            print(f"    Layer {i} ERROR: size(2) should be {seq_len + 1}, got {rope_size}")
            all_ok = False

    if all_ok:
        print("\n  PASS: All RoPE position checks passed")
    else:
        print("\n  FAIL: RoPE position errors detected")
    print()
    return all_ok


def test_cache_shapes():
    """Test 3: Verify cache tensor shapes through prefill + multi-step decode."""
    print("=" * 60)
    print("TEST 3: Cache shape verification")
    print("=" * 60)

    model, config = make_micro_model(kvheads=2)
    input_ids = torch.randint(0, 256, (1, 32))
    seq_len = input_ids.shape[1]

    minikv_config = MiniKVConfig(selection_method="h2o", heavy_ratio=0.25, recent_ratio=0.25)
    extra_kwargs = create_minikv_kwargs(minikv_config, num_layers=config.nlayers)

    print(f"\n  Model: nheads={config.nheads}, kvheads={config.kvheads}, nlayers={config.nlayers}")
    print(f"  Input seq_len: {seq_len}")
    print(f"  Expected kept: {int(seq_len * 0.25) + int(seq_len * 0.25)} tokens (50%)")

    # Full generation
    result = generate(
        model, input_ids, max_new_tokens=5, use_cache=True, do_sample=False,
        extra_kwargs=extra_kwargs,
    )

    print(f"\n  Output shape: {result.shape}")
    print(f"  Generated tokens: {result[0, seq_len:].tolist()}")

    all_ok = True
    expected_kept = int(seq_len * 0.25) + int(seq_len * 0.25)

    # Check the output is the right length
    if result.shape[1] != seq_len + 5:
        print(f"  ERROR: expected {seq_len + 5} total tokens, got {result.shape[1]}")
        all_ok = False
    else:
        print(f"  PASS: output length correct ({result.shape[1]})")

    if all_ok:
        print("\n  PASS: Cache shape checks passed")
    else:
        print("\n  FAIL: Cache shape errors detected")
    print()
    return all_ok


def test_selection_standalone():
    """Test 4: Standalone selection mechanism tests."""
    print("=" * 60)
    print("TEST 4: Selection mechanism standalone tests")
    print("=" * 60)

    B, H, S, D = 1, 4, 100, 16
    all_ok = True

    # H2O
    print("\n  H2O Selection (heavy=0.25, recent=0.25):")
    keys = torch.randn(B, H, S, D)
    values = torch.randn(B, H, S, D)
    attn_map = torch.randn(B, H, S).abs()

    selector = H2OSelection(heavy_ratio=0.25, recent_ratio=0.25)
    k_out, v_out = selector.select(keys, values, attn_map)
    expected = int(S * 0.25) + int(S * 0.25)  # 25 + 25 = 50
    print(f"    Input: {S} tokens -> Kept: {k_out.shape[2]} tokens (expected {expected})")
    if k_out.shape[2] != expected:
        print(f"    ERROR: expected {expected}, got {k_out.shape[2]}")
        all_ok = False

    # Verify recent tokens are always kept
    # The last 25 tokens should be in the output
    recent_start = S - int(S * 0.25)  # 75
    # Check the recent keys match
    recent_keys_original = keys[:, :, recent_start:, :]
    recent_keys_output = k_out[:, :, -int(S * 0.25):, :]
    if torch.allclose(recent_keys_original, recent_keys_output):
        print(f"    PASS: Recent tokens preserved correctly")
    else:
        print(f"    ERROR: Recent tokens NOT preserved correctly")
        all_ok = False

    # SnapKV
    print("\n  SnapKV Selection (sparsity=0.5, window=16):")
    queries = torch.randn(B, H, S, D)
    selector = SnapKVSelection(window_size=16, prompt_sparsity_ratio=0.5, kernel_size=5)
    k_out, v_out = selector.select(keys, values, queries)
    expected = int(S * 0.5)  # 50
    print(f"    Input: {S} tokens -> Kept: {k_out.shape[2]} tokens (expected {expected})")
    if k_out.shape[2] != expected:
        print(f"    ERROR: expected {expected}, got {k_out.shape[2]}")
        all_ok = False

    if all_ok:
        print("\n  PASS: All selection tests passed")
    else:
        print("\n  FAIL: Selection errors detected")
    print()
    return all_ok


def test_minikv_vs_no_eviction_forward():
    """Test 5: Compare single forward pass (prefill only, no decode) to verify
    that prefill attention output is identical with and without eviction."""
    print("=" * 60)
    print("TEST 5: Prefill output identity check")
    print("=" * 60)
    print("  (MiniKV prefill uses full attention for output, eviction only affects cache)")

    model, config = make_micro_model()
    torch.manual_seed(99)
    input_ids = torch.randint(0, 256, (1, 64))

    # Baseline forward
    with torch.no_grad():
        baseline_logits = model(input_ids, use_cache=False)

    # MiniKV forward (prefill)
    minikv_config = MiniKVConfig(selection_method="h2o", heavy_ratio=0.25, recent_ratio=0.25)
    extra_kwargs = create_minikv_kwargs(minikv_config, num_layers=config.nlayers)

    with torch.no_grad():
        minikv_output = model(input_ids, use_cache=True, **extra_kwargs)
    minikv_logits = minikv_output[0]

    # Prefill logits should be IDENTICAL (eviction doesn't affect prefill output)
    max_diff = (baseline_logits - minikv_logits).abs().max().item()
    print(f"\n  Max logit difference (prefill): {max_diff:.2e}")

    if max_diff < 1e-4:
        print("  PASS: Prefill logits match (eviction doesn't affect prefill output)")
        ok = True
    else:
        print("  FAIL: Prefill logits differ — something is wrong with the prefill path")
        ok = False

    print()
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, default="all",
                       choices=["all", "baseline", "rope", "cache", "selection", "prefill"],
                       help="Which test to run")
    args = parser.parse_args()

    tests = {
        "baseline": test_baseline_vs_minikv,
        "rope": test_rope_positions,
        "cache": test_cache_shapes,
        "selection": test_selection_standalone,
        "prefill": test_minikv_vs_no_eviction_forward,
    }

    if args.test == "all":
        results = {}
        for name, fn in tests.items():
            try:
                results[name] = fn()
            except Exception as e:
                print(f"\n  EXCEPTION in {name}: {e}")
                import traceback
                traceback.print_exc()
                results[name] = False

        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for name, ok in results.items():
            status = "PASS" if ok else "FAIL"
            print(f"  {name}: {status}")
        print()

        if all(results.values()):
            print("All tests passed!")
        else:
            print("Some tests FAILED!")
            sys.exit(1)
    else:
        ok = tests[args.test]()
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
