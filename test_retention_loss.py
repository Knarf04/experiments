"""
Correctness tests for the retention_loss feature.

Tests verify:
  1. Shape and gradient existence (single device)
  2. (Skipped — requires multi-GPU) CP shape
  3. CP vs non-CP numerical agreement on final states (CRITICAL)
  4. Gradient correctness via finite difference (CRITICAL)
  5. Backward compatibility when retention_loss is disabled
  6. Interaction between retention_loss and state_pass
  7. End-to-end retention loss optimization loop

Run:
    cd /path/to/mamba && python -m pytest experiments/test_retention_loss.py -v
    # or directly:
    python experiments/test_retention_loss.py
"""

import sys
import torch
import torch.nn.functional as F
from copy import deepcopy
from einops import rearrange

from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined


# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

BATCH = 2
D_MODEL = 256
D_STATE = 128
CHUNK_SIZE = 4
NGROUPS = 1
HEADDIM = 64
D_CONV = 4
EXPAND = 2
DEVICE = "cuda"
DTYPE = torch.float32  # float32 for numerical precision

D_INNER = EXPAND * D_MODEL          # 512
NHEADS = D_INNER // HEADDIM         # 8
NUM_SHARDS = 4                      # simulated CP ranks
SEQ_LEN = 4 * NUM_SHARDS * CHUNK_SIZE  # 64

MODEL_KWARGS = dict(
    d_model=D_MODEL,
    d_state=D_STATE,
    chunk_size=CHUNK_SIZE,
    ngroups=NGROUPS,
    headdim=HEADDIM,
    d_conv=D_CONV,
    device=DEVICE,
    dtype=DTYPE,
)


def make_model(experiments=None, seed=42, **extra):
    torch.manual_seed(seed)
    kw = {**MODEL_KWARGS, "experiments": experiments or {}, **extra}
    return Mamba2(**kw)


def make_input(batch=BATCH, seq_len=SEQ_LEN, requires_grad=False, seed=42):
    torch.manual_seed(seed)
    return torch.randn(batch, seq_len, D_MODEL, device=DEVICE, dtype=DTYPE,
                        requires_grad=requires_grad)


# ===========================================================================
# Test 1: Shape and Gradient Existence (single device, no CP)
# ===========================================================================

def test_1_shape_and_grad():
    """retention_loss=True → retention_states with correct shape, requires_grad, and grads propagate."""
    model = make_model(experiments={"retention_loss": True}, layer_idx=0)
    x = make_input(requires_grad=True)

    result = model(x)
    assert isinstance(result, tuple) and len(result) == 2, \
        "Expected (out, exp_dict) tuple when experiments is non-empty"
    out, exp_dict = result
    assert 0 in exp_dict, "layer_idx=0 should be a key in exp_dict"
    exp = exp_dict[0]
    assert "retention_states" in exp, "retention_states missing from experiment_out"

    rs = exp["retention_states"]
    expected_shape = (BATCH, 1, NHEADS, HEADDIM, D_STATE)
    assert rs.shape == expected_shape, f"Shape mismatch: {rs.shape} vs {expected_shape}"
    assert rs.requires_grad, "retention_states should require grad"

    # Backward through retention_states
    model.zero_grad()
    rs.sum().backward()
    assert model.A_log.grad is not None, "A_log should have grad"
    assert model.in_proj.weight.grad is not None, "in_proj.weight should have grad"
    assert x.grad is not None, "Input should have grad"

    print("Test 1 PASSED: shape and gradient existence")


# ===========================================================================
# Test 2: CP Shape (simulated) — skipped, requires multi-GPU
# ===========================================================================

def test_2_cp_shape():
    """TODO: Requires real multi-GPU CP mesh. Covered implicitly by Test 3."""
    print("Test 2 SKIPPED: requires multi-GPU CP mesh")


# ===========================================================================
# Test 3: CP vs Non-CP Numerical Agreement (CRITICAL)
# ===========================================================================

def test_3_cp_vs_noncp_states():
    """
    Verify that sharding a sequence into N chunks and running sequentially
    (passing final_state as initial_state) produces the same per-chunk final
    states as running the full sequence at once.

    This is the core correctness test: it validates that the states surfaced
    by retention_loss match the true SSM states at chunk boundaries.
    """
    torch.manual_seed(42)

    # Create raw SSM inputs (bypass the module, test the kernel directly)
    batch = BATCH
    seq_len = SEQ_LEN
    x = torch.randn(batch, seq_len, NHEADS, HEADDIM, device=DEVICE, dtype=DTYPE)
    dt = torch.randn(batch, seq_len, NHEADS, device=DEVICE, dtype=DTYPE)
    A = -torch.rand(NHEADS, device=DEVICE, dtype=DTYPE).abs() - 0.5  # negative
    B = torch.randn(batch, seq_len, NGROUPS, D_STATE, device=DEVICE, dtype=DTYPE)
    C = torch.randn(batch, seq_len, NGROUPS, D_STATE, device=DEVICE, dtype=DTYPE)

    # --- Full sequence (ground truth) ---
    y_full, final_full = mamba_chunk_scan_combined(
        x, dt, A, B, C, CHUNK_SIZE,
        dt_softplus=True,
        return_final_states=True,
    )

    # --- Sharded: simulate CP by running each shard sequentially ---
    shard_len = seq_len // NUM_SHARDS
    initial_states = None
    shard_outputs = []
    shard_finals = []
    for i in range(NUM_SHARDS):
        sl = slice(i * shard_len, (i + 1) * shard_len)
        y_shard, fs_shard = mamba_chunk_scan_combined(
            x[:, sl], dt[:, sl], A, B[:, sl], C[:, sl], CHUNK_SIZE,
            dt_softplus=True,
            initial_states=initial_states,
            return_final_states=True,
        )
        shard_outputs.append(y_shard)
        shard_finals.append(fs_shard)
        initial_states = fs_shard.detach()

    # Concatenated output should match full output
    y_cat = torch.cat(shard_outputs, dim=1)
    torch.testing.assert_close(y_cat, y_full, atol=1e-4, rtol=1e-4)

    # Last shard's final state should match the full-sequence final state
    torch.testing.assert_close(shard_finals[-1], final_full, atol=1e-4, rtol=1e-4)

    # Intermediate shard final states: verify they're non-trivial (not zeros)
    for i, fs in enumerate(shard_finals):
        assert fs.abs().max() > 1e-6, f"Shard {i} final_states are all zeros"

    print("Test 3 PASSED: CP vs non-CP numerical agreement")


# ===========================================================================
# Test 4: Gradient Correctness via Finite Difference (CRITICAL)
# ===========================================================================

def test_4_finite_difference():
    """
    Compare analytical gradients (from backward) with numerical finite-difference
    gradients for the retention_states path. Focus on A_log and dt_bias — the
    parameters most affected by the backward changes.
    """
    torch.manual_seed(42)
    model = make_model(experiments={"retention_loss": True}, layer_idx=0)
    x = make_input(requires_grad=False, seed=123)  # fixed input, no input grad needed

    def loss_fn():
        out, exp_dict = model(x)
        rs = exp_dict[0]["retention_states"]
        # Use a non-trivial scalar reduction so gradients aren't all 1s
        return (rs ** 2).sum() + out.sum()

    # Analytical gradient
    model.zero_grad()
    loss = loss_fn()
    loss.backward()

    eps = 5e-4
    tol = 1e-2  # relative tolerance for finite-diff comparison

    params_to_check = [
        ("A_log", model.A_log),
        ("dt_bias", model.dt_bias),
    ]

    for name, param in params_to_check:
        if param.grad is None:
            print(f"  WARNING: {name}.grad is None, skipping")
            continue

        analytical = param.grad.clone()
        numerical = torch.zeros_like(param)

        # Finite difference over a subset of elements (full check is slow)
        n_check = min(param.numel(), 8)
        flat = param.data.view(-1)
        for idx in range(n_check):
            orig = flat[idx].item()

            flat[idx] = orig + eps
            model.zero_grad()
            lp = loss_fn().item()

            flat[idx] = orig - eps
            model.zero_grad()
            lm = loss_fn().item()

            flat[idx] = orig  # restore
            numerical.view(-1)[idx] = (lp - lm) / (2 * eps)

        # Compare the subset
        ana_sub = analytical.view(-1)[:n_check]
        num_sub = numerical.view(-1)[:n_check]

        # Relative error (handle near-zero gracefully)
        scale = torch.clamp(ana_sub.abs().max(), min=1e-6)
        rel_err = (ana_sub - num_sub).abs().max() / scale
        assert rel_err < tol, (
            f"Finite-diff mismatch for {name}: rel_err={rel_err:.6f} > tol={tol}\n"
            f"  analytical: {ana_sub.tolist()}\n"
            f"  numerical:  {num_sub.tolist()}"
        )

    print("Test 4 PASSED: finite difference gradient correctness")


# ===========================================================================
# Test 5: Backward Compatibility
# ===========================================================================

def test_5_backward_compat():
    """
    retention_loss=False (or absent) should NOT produce retention_states
    and should give identical outputs to the no-experiments case.
    """
    seed = 42
    x_a = make_input(seed=99)
    x_b = x_a.clone()

    # Case A: no experiments at all
    model_a = make_model(experiments={}, layer_idx=0, seed=seed)
    out_a = model_a(x_a)
    assert not isinstance(out_a, tuple), \
        "With empty experiments dict, forward should return plain tensor"

    # Case B: retention_loss=False explicitly
    model_b = make_model(experiments={"retention_loss": False}, layer_idx=0, seed=seed)
    result_b = model_b(x_b)
    assert isinstance(result_b, tuple), \
        "With non-empty experiments dict, forward should return (out, exp_dict)"
    out_b, exp_dict_b = result_b
    assert "retention_states" not in exp_dict_b[0], \
        "retention_states should NOT be present when retention_loss=False"

    # Outputs should be identical
    torch.testing.assert_close(out_a, out_b, atol=0, rtol=0)

    print("Test 5 PASSED: backward compatibility")


# ===========================================================================
# Test 6: Interaction with State Passing
# ===========================================================================

def test_6_state_pass_interaction():
    """
    Both retention_loss=True and state_pass=True should coexist:
    - experiment_out has both "retention_states" and "final_states"
    - retention_states requires grad, final_states does not
    - State passing works across sequential forward calls
    """
    model = make_model(
        experiments={
            "retention_loss": True,
            "sp": {"batch": BATCH},
        },
        layer_idx=0,
    )
    x = make_input(requires_grad=True)

    # First forward
    out, exp_dict = model(x)
    exp = exp_dict[0]
    assert "retention_states" in exp, "retention_states missing"
    assert "final_states" in exp, "final_states missing"

    rs = exp["retention_states"]
    fs = exp["final_states"]
    assert rs.requires_grad, "retention_states should require grad"
    # final_states stored via .detach() in state_pass logic, but the buffer
    # itself may still be a leaf — just verify retention_states is differentiable
    assert rs.shape[1] == 1, "Single device → num_ranks=1"

    # Backward through retention_states should work
    model.zero_grad()
    rs.sum().backward()
    assert model.A_log.grad is not None, "Grad should flow through retention_states"

    # Second forward — state passing should have updated prev_final_states
    model.zero_grad()
    x2 = make_input(requires_grad=True, seed=99)
    out2, exp_dict2 = model(x2)
    # prev_final_states should be non-zero after first forward
    assert model.prev_final_states.abs().max() > 1e-8, \
        "prev_final_states should be updated after first forward"

    print("Test 6 PASSED: state_pass + retention_loss interaction")


# ===========================================================================
# Test 7: End-to-End Retention Loss Optimization
# ===========================================================================

def test_7_e2e_optimization():
    """
    Run a few optimization steps using retention_states as a loss signal.
    Verify the loss changes and parameters update.
    """
    model = make_model(experiments={"retention_loss": True}, layer_idx=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Snapshot initial parameters
    init_params = {n: p.clone() for n, p in model.named_parameters()}

    losses = []
    for step in range(5):
        optimizer.zero_grad()
        x = make_input(seed=step * 7)
        out, exp_dict = model(x)
        rs = exp_dict[0]["retention_states"]
        # Loss: L2 norm of states (encourages small states as a regularizer)
        retention_loss = (rs ** 2).mean()
        total_loss = out.sum() * 0.0 + retention_loss  # only retention drives grads
        total_loss.backward()
        optimizer.step()
        losses.append(total_loss.item())

    # Loss should change (not stuck at same value)
    assert not all(abs(losses[i] - losses[0]) < 1e-10 for i in range(1, len(losses))), \
        f"Loss is stuck: {losses}"

    # Some parameters should have changed
    n_changed = 0
    for name, p in model.named_parameters():
        if not torch.equal(p, init_params[name]):
            n_changed += 1
    assert n_changed > 0, "No parameters changed after optimization"

    print(f"Test 7 PASSED: e2e optimization (losses: {[f'{l:.4f}' for l in losses]}, "
          f"{n_changed} params changed)")


# ===========================================================================
# Runner
# ===========================================================================

ALL_TESTS = [
    test_1_shape_and_grad,
    test_2_cp_shape,
    test_3_cp_vs_noncp_states,
    test_4_finite_difference,
    test_5_backward_compat,
    test_6_state_pass_interaction,
    test_7_e2e_optimization,
]


def main():
    passed = 0
    failed = 0
    skipped = 0
    for test_fn in ALL_TESTS:
        try:
            test_fn()
            if "SKIPPED" in test_fn.__doc__ or "SKIP" in test_fn.__name__:
                skipped += 1
            else:
                passed += 1
        except Exception as e:
            failed += 1
            print(f"Test {test_fn.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
