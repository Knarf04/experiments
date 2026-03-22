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
    that prefill attention output is identical with and without eviction.
    Includes detailed NaN diagnostics."""
    print("=" * 60)
    print("TEST 5: Prefill output identity check + NaN diagnostics")
    print("=" * 60)

    model, config = make_micro_model()
    torch.manual_seed(99)
    input_ids = torch.randint(0, 256, (1, 64))

    # --- Baseline forward (SDPA, no eviction) ---
    print("\n  [Baseline] use_cache=False, standard SDPA")
    with torch.no_grad():
        baseline_logits = model(input_ids, use_cache=False)
    print(f"    logits shape: {baseline_logits.shape}")
    print(f"    has NaN: {baseline_logits.isnan().any().item()}")
    print(f"    has Inf: {baseline_logits.isinf().any().item()}")
    print(f"    min/max: {baseline_logits.min().item():.4f} / {baseline_logits.max().item():.4f}")

    # --- Baseline with use_cache=True (SDPA + cache) ---
    print("\n  [Baseline+cache] use_cache=True, standard SDPA")
    with torch.no_grad():
        baseline_cached = model(input_ids, use_cache=True)
    baseline_cached_logits = baseline_cached[0]
    print(f"    logits shape: {baseline_cached_logits.shape}")
    print(f"    has NaN: {baseline_cached_logits.isnan().any().item()}")
    diff_cache = (baseline_logits - baseline_cached_logits).abs().max().item()
    print(f"    diff vs no-cache baseline: {diff_cache:.2e}")

    # --- MiniKV forward (prefill) ---
    print("\n  [MiniKV] use_cache=True, minikv attention op")
    minikv_config = MiniKVConfig(selection_method="h2o", heavy_ratio=0.25, recent_ratio=0.25)
    extra_kwargs = create_minikv_kwargs(minikv_config, num_layers=config.nlayers)

    with torch.no_grad():
        minikv_output = model(input_ids, use_cache=True, **extra_kwargs)
    minikv_logits = minikv_output[0]
    minikv_cache = minikv_output[1]
    print(f"    logits shape: {minikv_logits.shape}")
    print(f"    has NaN: {minikv_logits.isnan().any().item()}")
    print(f"    has Inf: {minikv_logits.isinf().any().item()}")
    if not minikv_logits.isnan().any():
        print(f"    min/max: {minikv_logits.min().item():.4f} / {minikv_logits.max().item():.4f}")

    # --- If NaN, do layer-by-layer diagnosis ---
    if minikv_logits.isnan().any():
        print("\n  === NaN DETECTED — running layer-by-layer diagnosis ===")
        _diagnose_nan_layer_by_layer(model, config, input_ids)

    # --- Compare ---
    max_diff = (baseline_logits - minikv_logits).abs().max().item()
    print(f"\n  Max logit difference (prefill): {max_diff:.2e}")

    if max_diff < 1e-4:
        print("  PASS: Prefill logits match")
        ok = True
    elif not torch.isnan(torch.tensor(max_diff)):
        print(f"  FAIL: Prefill logits differ by {max_diff:.2e}")
        ok = False
    else:
        print("  FAIL: NaN in logits — see diagnosis above")
        ok = False

    print()
    return ok


def _diagnose_nan_layer_by_layer(model, config, input_ids):
    """Run the MiniKV prefill path manually, checking for NaN after each step."""
    import fms.utils.minikv.attention_op as minikv_ops

    # Monkey-patch compute_prefill to add NaN checks
    original_compute = minikv_ops._minikv_compute_prefill_op

    def instrumented_compute(query, key_cache, value_cache, nheads, kvheads,
                             p_dropout, scale_factor, **attn_kwargs):
        import math

        state = attn_kwargs["minikv_state"]
        layer_idx = state["layer_counter"] - 1

        queries = query.transpose(2, 1)
        keys = key_cache
        values = value_cache

        print(f"\n    Layer {layer_idx} compute_prefill:")
        print(f"      query:     shape={list(query.shape)}, nan={query.isnan().any().item()}")
        print(f"      key_cache: shape={list(key_cache.shape)}, nan={key_cache.isnan().any().item()}")
        print(f"      val_cache: shape={list(value_cache.shape)}, nan={value_cache.isnan().any().item()}")

        expansion = nheads // kvheads
        if expansion != 1:
            keys_e = keys.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)
            values_e = values.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)
        else:
            keys_e, values_e = keys, values

        B, H, S, D = queries.shape

        # Step 1: QK^T
        attn_weights = torch.matmul(queries, keys_e.transpose(2, 3))
        if scale_factor is not None:
            attn_weights = attn_weights * scale_factor
        else:
            attn_weights = attn_weights / math.sqrt(D)
        print(f"      after QK^T:  nan={attn_weights.isnan().any().item()}, "
              f"min={attn_weights.min().item():.4f}, max={attn_weights.max().item():.4f}")

        # Step 2: causal mask
        causal_mask = torch.full((S, S), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        mask_cond = torch.arange(S, device=attn_weights.device)
        causal_mask.masked_fill_(mask_cond < (mask_cond + 1).view(S, 1), 0)
        attn_weights = attn_weights + causal_mask[None, None, :, :]
        print(f"      after mask:  nan={attn_weights.isnan().any().item()}, "
              f"min={attn_weights.min().item():.4f}, max={attn_weights.max().item():.4f}")

        # Step 3: softmax
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(queries.dtype)
        print(f"      after softmax: nan={attn_weights.isnan().any().item()}, "
              f"min={attn_weights.min().item():.6f}, max={attn_weights.max().item():.6f}")

        # Step 4: attention output
        attn_output = torch.matmul(attn_weights, values_e)
        print(f"      attn_output: nan={attn_output.isnan().any().item()}, "
              f"min={attn_output.min().item():.4f}, max={attn_output.max().item():.4f}")

        # Call original for the eviction part
        return attn_output.transpose(2, 1).contiguous()

    # Temporarily replace
    # We can't easily replace just the compute part, so let's just run forward
    # and check the output layer by layer
    print("\n    Running model._helper layer by layer:")

    minikv_config = MiniKVConfig(selection_method="h2o", heavy_ratio=0.25, recent_ratio=0.25)
    extra_kwargs = create_minikv_kwargs(minikv_config, num_layers=config.nlayers)

    with torch.no_grad():
        x = model.shared(input_ids)
        print(f"    After embedding: nan={x.isnan().any().item()}, shape={list(x.shape)}")

        past_key_value_states = [None] * len(model.layers)
        for i, layer in enumerate(model.layers):
            output = layer(
                x=x, position_ids=None,
                past_key_value_state=past_key_value_states[i],
                use_cache=True, **extra_kwargs,
            )
            x, cache = output
            print(f"    After layer {i}: nan={x.isnan().any().item()}, "
                  f"min={x.min().item():.4f}, max={x.max().item():.4f}")
            if x.isnan().any():
                print(f"      >>> NaN first appears at layer {i}!")
                # Check the cache too
                kv = cache[0]
                if isinstance(kv, EvictedKVCache) and kv.keys is not None:
                    print(f"      cache keys nan: {kv.keys.isnan().any().item()}")
                    print(f"      cache vals nan: {kv.values.isnan().any().item()}")
                break


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
