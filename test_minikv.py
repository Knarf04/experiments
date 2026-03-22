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
    """Create a tiny LLaMA with properly initialized random weights."""
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
    model.reset_parameters()
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
    torch.manual_seed(123)  # use same seed as test 1 (seed 99 causes NaN even in baseline)
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
    """Trace through layer 0 sub-operations to find exact NaN source."""
    print("\n    === DETAILED LAYER 0 DIAGNOSIS ===")

    with torch.no_grad():
        x = model.shared(input_ids)
        print(f"\n    embedding: nan={x.isnan().any().item()}, shape={list(x.shape)}")

        layer = model.layers[0]

        # Step 1: LayerNorm
        residual = x
        x_ln = layer.ln(x)
        print(f"    ln(x):     nan={x_ln.isnan().any().item()}, "
              f"min={x_ln.min().item():.6f}, max={x_ln.max().item():.6f}")

        # Step 2: QKV projection
        attn = layer.attn
        q_out, k_out, v_out = attn.in_proj(x_ln, None, None)
        print(f"    q_proj:    nan={q_out.isnan().any().item()}, "
              f"min={q_out.min().item():.6f}, max={q_out.max().item():.6f}")
        print(f"    k_proj:    nan={k_out.isnan().any().item()}, "
              f"min={k_out.min().item():.6f}, max={k_out.max().item():.6f}")
        print(f"    v_proj:    nan={v_out.isnan().any().item()}, "
              f"min={v_out.min().item():.6f}, max={v_out.max().item():.6f}")

        B, S, _ = x_ln.shape
        queries = q_out.view(B, S, attn.nheads, attn.emb_kq_per_head)
        keys = k_out.view(B, S, attn.kvheads, attn.emb_kq_per_head)
        values = v_out.view(B, S, attn.kvheads, attn.emb_v_per_head)
        print(f"    reshape:   q={list(queries.shape)}, k={list(keys.shape)}, v={list(values.shape)}")

        # Step 3: RoPE
        if attn.position_encoder is not None:
            queries_r, keys_r = attn.position_encoder.adjusted_qk(
                queries, keys, None, None, False
            )
            print(f"    rope(q):   nan={queries_r.isnan().any().item()}, "
                  f"min={queries_r.min().item():.6f}, max={queries_r.max().item():.6f}")
            print(f"    rope(k):   nan={keys_r.isnan().any().item()}, "
                  f"min={keys_r.min().item():.6f}, max={keys_r.max().item():.6f}")
        else:
            queries_r, keys_r = queries, keys
            print(f"    no RoPE")

        # Step 4: Transpose (what store_op does)
        keys_t = keys_r.transpose(2, 1)   # (B, kvheads, S, D)
        values_t = values.transpose(2, 1)

        # Step 5: GQA expand
        expansion = attn.nheads // attn.kvheads
        if expansion != 1:
            keys_e = keys_t.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)
            values_e = values_t.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)
        else:
            keys_e, values_e = keys_t, values_t
        print(f"    expand:    keys_e={list(keys_e.shape)}, values_e={list(values_e.shape)}")

        # Step 6: QK^T
        queries_t = queries_r.transpose(2, 1)  # (B, nheads, S, D)
        D = queries_t.shape[-1]
        import math
        attn_weights = torch.matmul(queries_t, keys_e.transpose(2, 3)) / math.sqrt(D)
        print(f"    QK^T/sqrt: nan={attn_weights.isnan().any().item()}, "
              f"min={attn_weights.min().item():.6f}, max={attn_weights.max().item():.6f}, "
              f"shape={list(attn_weights.shape)}")

        # Step 7: Causal mask
        causal_mask = torch.full((S, S), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        mask_cond = torch.arange(S, device=attn_weights.device)
        causal_mask.masked_fill_(mask_cond < (mask_cond + 1).view(S, 1), 0)
        attn_weights_masked = attn_weights + causal_mask[None, None, :, :]
        print(f"    +mask:     nan={attn_weights_masked.isnan().any().item()}, "
              f"min={attn_weights_masked.min().item():.6f}, max={attn_weights_masked.max().item():.6f}")

        # Step 8: Softmax
        attn_probs = nn.functional.softmax(attn_weights_masked, dim=-1, dtype=torch.float32).to(queries_t.dtype)
        print(f"    softmax:   nan={attn_probs.isnan().any().item()}, "
              f"min={attn_probs.min().item():.6f}, max={attn_probs.max().item():.6f}")

        # Step 9: Attention output
        attn_out = torch.matmul(attn_probs, values_e)
        print(f"    AV:        nan={attn_out.isnan().any().item()}, "
              f"min={attn_out.min().item():.6f}, max={attn_out.max().item():.6f}")

        # Step 10: Output projection
        attn_out_flat = attn_out.transpose(2, 1).contiguous().view(B, S, attn.nheads * attn.emb_v_per_head)
        dense_out = attn.dense(attn_out_flat)
        print(f"    dense:     nan={dense_out.isnan().any().item()}, "
              f"min={dense_out.min().item():.6f}, max={dense_out.max().item():.6f}")

        # Step 11: Residual
        x_after_attn = dense_out + residual
        print(f"    +residual: nan={x_after_attn.isnan().any().item()}, "
              f"min={x_after_attn.min().item():.6f}, max={x_after_attn.max().item():.6f}")

        # Step 12: FFN
        residual2 = x_after_attn
        x_ff_ln = layer.ff_ln(x_after_attn)
        print(f"    ff_ln:     nan={x_ff_ln.isnan().any().item()}, "
              f"min={x_ff_ln.min().item():.6f}, max={x_ff_ln.max().item():.6f}")
        x_ff = layer.ff_sub_layer(x_ff_ln)
        print(f"    ff:        nan={x_ff.isnan().any().item()}, "
              f"min={x_ff.min().item():.6f}, max={x_ff.max().item():.6f}")
        x_out = x_ff + residual2
        print(f"    +residual: nan={x_out.isnan().any().item()}, "
              f"min={x_out.min().item():.6f}, max={x_out.max().item():.6f}")

        # Now compare: run same layer through actual SDPA (no MiniKV)
        print(f"\n    --- COMPARISON: Same layer with SDPA (use_cache=False, no MiniKV) ---")
        x2 = model.shared(input_ids)
        residual2 = x2
        x2 = layer.ln(x2)
        x2 = layer.attn(q=x2, position_ids=None, past_key_value_state=None, use_cache=False)
        print(f"    sdpa attn:  nan={x2.isnan().any().item()}, "
              f"min={x2.min().item():.6f}, max={x2.max().item():.6f}")
        x2 = x2 + residual2
        x2_r = x2
        x2 = layer.ff_ln(x2)
        x2 = layer.ff_sub_layer(x2)
        x2 = x2 + x2_r
        print(f"    sdpa out:   nan={x2.isnan().any().item()}, "
              f"min={x2.min().item():.6f}, max={x2.max().item():.6f}")


def test_decode_divergence():
    """Test 6: Check if first decode step produces correct logits.
    Compares baseline (SDPA full cache) vs MiniKV (evicted cache) decode."""
    print("=" * 60)
    print("TEST 6: Decode step comparison (baseline vs MiniKV)")
    print("=" * 60)

    model, config = make_micro_model()
    torch.manual_seed(123)
    input_ids = torch.randint(0, 256, (1, 64))
    seq_len = input_ids.shape[1]

    # --- Baseline: prefill + 1 decode step ---
    print("\n  [Baseline] SDPA prefill + decode")
    with torch.no_grad():
        baseline_out = model(input_ids, use_cache=True)
    baseline_logits_prefill = baseline_out[0]
    baseline_cache = baseline_out[1]
    next_token = baseline_logits_prefill[:, -1:, :].argmax(dim=-1)
    print(f"    Prefill logits nan: {baseline_logits_prefill.isnan().any().item()}")
    print(f"    Next token: {next_token.item()}")
    print(f"    Cache[0] shape: {baseline_cache[0][0].shape}")

    # Decode 1 step
    with torch.no_grad():
        baseline_decode = model(next_token, past_key_value_states=list(baseline_cache), use_cache=True)
    baseline_decode_logits = baseline_decode[0]
    print(f"    Decode logits nan: {baseline_decode_logits.isnan().any().item()}")
    if not baseline_decode_logits.isnan().any():
        baseline_next2 = baseline_decode_logits[:, -1, :].argmax(dim=-1)
        print(f"    2nd generated token: {baseline_next2.item()}")

    # --- MiniKV: prefill + 1 decode step ---
    print("\n  [MiniKV] H2O prefill + decode")
    minikv_config = MiniKVConfig(selection_method="h2o", heavy_ratio=0.25, recent_ratio=0.25)
    extra_kwargs = create_minikv_kwargs(minikv_config, num_layers=config.nlayers)

    with torch.no_grad():
        minikv_out = model(input_ids, use_cache=True, **extra_kwargs)
    minikv_logits_prefill = minikv_out[0]
    minikv_cache = minikv_out[1]
    next_token_mk = minikv_logits_prefill[:, -1:, :].argmax(dim=-1)
    print(f"    Prefill logits nan: {minikv_logits_prefill.isnan().any().item()}")
    print(f"    Next token: {next_token_mk.item()}")

    # Check cache state
    kv0 = minikv_cache[0][0]
    print(f"    Cache[0] type: {type(kv0).__name__}")
    if isinstance(kv0, EvictedKVCache):
        print(f"    Cache[0] physical: {kv0.keys.shape[2]}, logical: {kv0.seq_len_logical}")
        print(f"    Cache[0] keys nan: {kv0.keys.isnan().any().item()}")

    # Prefill logits should match
    prefill_diff = (baseline_logits_prefill - minikv_logits_prefill).abs().max().item()
    print(f"\n    Prefill logit diff: {prefill_diff:.2e}")
    if prefill_diff > 1e-4:
        print(f"    ERROR: Prefill logits should be identical!")

    # Decode 1 step with minikv
    extra_kwargs["minikv_state"]["layer_counter"] = 0
    extra_kwargs["minikv_state"]["is_prefill"] = False
    if "mask" in extra_kwargs:
        del extra_kwargs["mask"]

    with torch.no_grad():
        minikv_decode = model(next_token_mk, past_key_value_states=list(minikv_cache),
                             use_cache=True, **extra_kwargs)
    minikv_decode_logits = minikv_decode[0]
    print(f"    Decode logits nan: {minikv_decode_logits.isnan().any().item()}")
    if not minikv_decode_logits.isnan().any():
        minikv_next2 = minikv_decode_logits[:, -1, :].argmax(dim=-1)
        print(f"    2nd generated token: {minikv_next2.item()}")

    # Compare decode logits
    if not baseline_decode_logits.isnan().any() and not minikv_decode_logits.isnan().any():
        decode_diff = (baseline_decode_logits - minikv_decode_logits).abs().max().item()
        print(f"\n    Decode logit diff: {decode_diff:.2e}")
        if decode_diff < 1e-2:
            print("    (very close — eviction has minimal impact)")
        else:
            print("    (expected: some divergence due to evicted tokens)")

    # Overall
    ok = not minikv_logits_prefill.isnan().any() and not minikv_decode_logits.isnan().any()
    if ok:
        print("\n  PASS: No NaN in MiniKV prefill or decode")
    else:
        print("\n  FAIL: NaN detected")
    print()
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, default="all",
                       choices=["all", "baseline", "rope", "cache", "selection", "prefill", "decode"],
                       help="Which test to run")
    args = parser.parse_args()

    tests = {
        "baseline": test_baseline_vs_minikv,
        "rope": test_rope_positions,
        "cache": test_cache_shapes,
        "selection": test_selection_standalone,
        "prefill": test_minikv_vs_no_eviction_forward,
        "decode": test_decode_divergence,
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
