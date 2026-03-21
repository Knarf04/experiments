"""
Diagnostic v6: Test rmsnorm_fn directly and compare with manual implementation.
Check if mamba_ssm version or norm_before_gate semantics changed.

Usage:
    python experiments/diagnose_nemotronh_v6.py --model <path> --bf16
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def manual_rmsnorm_gated(hidden_states, weight, gate, eps, group_size, norm_before_gate):
    """Manual implementation of gated RMSNorm for comparison."""
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    if norm_before_gate:
        # Norm first, then gate
        groups = hidden_states.unfold(-1, group_size, group_size)
        variance = groups.pow(2).mean(-1, keepdim=True)
        hidden_states = groups / torch.sqrt(variance + eps)
        hidden_states = hidden_states.reshape(hidden_states.shape[:-2] + (-1,))
        hidden_states = weight.to(torch.float32) * hidden_states
        if gate is not None:
            hidden_states = hidden_states * torch.nn.functional.silu(gate.to(torch.float32))
    else:
        # Gate first, then norm
        if gate is not None:
            hidden_states = hidden_states * torch.nn.functional.silu(gate.to(torch.float32))
        groups = hidden_states.unfold(-1, group_size, group_size)
        variance = groups.pow(2).mean(-1, keepdim=True)
        hidden_states = groups / torch.sqrt(variance + eps)
        hidden_states = hidden_states.reshape(hidden_states.shape[:-2] + (-1,))
        hidden_states = weight.to(torch.float32) * hidden_states

    return hidden_states.to(input_dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("NemotronH Diagnostic v6 — RMSNorm Analysis")
    print("=" * 60)

    # Check mamba_ssm version
    import mamba_ssm
    print(f"\n[mamba_ssm version]: {mamba_ssm.__version__}")

    from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn
    import inspect
    sig = inspect.signature(rmsnorm_fn)
    print(f"[rmsnorm_fn signature]: {sig}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=dtype,
        attn_implementation="flash_attention_2",
    )
    model.eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Get layer 0 mamba mixer
    mixer = model.backbone.layers[0].mixer
    norm = mixer.norm

    print(f"\n[Norm config]")
    print(f"  weight shape: {norm.weight.shape}")
    print(f"  weight stats: mean={norm.weight.float().mean():.6f}, std={norm.weight.float().std():.6f}")
    print(f"  weight first5: {norm.weight[:5].float().tolist()}")
    print(f"  eps: {norm.variance_epsilon}")
    print(f"  group_size: {norm.group_size}")
    print(f"  norm_before_gate: False (hardcoded)")

    # Run mamba layer 0 to get scan_output and gate
    test_text = "The capital of France is"
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    emb = model.backbone.embeddings(inputs["input_ids"])
    layer0 = model.backbone.layers[0]
    normed = layer0.norm(emb.to(dtype=layer0.norm.weight.dtype))

    with torch.no_grad():
        projected = mixer.in_proj(normed)
        batch_size, seq_len, _ = normed.shape
        d_mlp = (projected.shape[-1] - mixer.intermediate_size - mixer.conv_dim - mixer.num_heads) // 2
        _, _, gate, hidden_states_B_C, dt = projected.split(
            [d_mlp, d_mlp, mixer.intermediate_size, mixer.conv_dim, mixer.num_heads], dim=-1
        )

        from causal_conv1d import causal_conv1d_fn
        hidden_states_B_C_conv = causal_conv1d_fn(
            x=hidden_states_B_C.transpose(1, 2),
            weight=mixer.conv1d.weight.squeeze(1),
            bias=mixer.conv1d.bias,
            activation=mixer.activation,
        ).transpose(1, 2)

        groups_time_state_size = mixer.n_groups * mixer.ssm_state_size
        hidden_states_split, B, C = torch.split(
            hidden_states_B_C_conv,
            [mixer.intermediate_size, groups_time_state_size, groups_time_state_size],
            dim=-1,
        )

        hidden_states_4d = hidden_states_split.view(batch_size, seq_len, -1, mixer.head_dim)
        A = -torch.exp(mixer.A_log.float())

        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
        scan_output, _ = mamba_chunk_scan_combined(
            hidden_states_4d, dt, A,
            B.view(batch_size, seq_len, mixer.n_groups, -1),
            C.view(batch_size, seq_len, mixer.n_groups, -1),
            chunk_size=mixer.chunk_size, D=mixer.D, z=None, seq_idx=None,
            return_final_states=True, dt_bias=mixer.dt_bias, dt_softplus=True,
        )
        scan_flat = scan_output.view(batch_size, seq_len, -1)

    print(f"\n[Inputs to gated norm]")
    print(f"  scan_output (x): shape={scan_flat.shape}, std={scan_flat.float().std():.6f}")
    print(f"  gate (z):        shape={gate.shape}, std={gate.float().std():.6f}")

    # Test 1: rmsnorm_fn with norm_before_gate=False (current code)
    with torch.no_grad():
        result_nbg_false = rmsnorm_fn(
            x=scan_flat, weight=norm.weight, bias=None, z=gate,
            eps=norm.variance_epsilon, group_size=norm.group_size,
            norm_before_gate=False,
        )
    print(f"\n[rmsnorm_fn norm_before_gate=False]")
    print(f"  output std: {result_nbg_false.float().std():.6f}")

    # Test 2: rmsnorm_fn with norm_before_gate=True
    with torch.no_grad():
        result_nbg_true = rmsnorm_fn(
            x=scan_flat, weight=norm.weight, bias=None, z=gate,
            eps=norm.variance_epsilon, group_size=norm.group_size,
            norm_before_gate=True,
        )
    print(f"\n[rmsnorm_fn norm_before_gate=True]")
    print(f"  output std: {result_nbg_true.float().std():.6f}")

    # Test 3: rmsnorm_fn WITHOUT gating (z=None)
    with torch.no_grad():
        result_no_gate = rmsnorm_fn(
            x=scan_flat, weight=norm.weight, bias=None, z=None,
            eps=norm.variance_epsilon, group_size=norm.group_size,
            norm_before_gate=False,
        )
    print(f"\n[rmsnorm_fn without gating (z=None)]")
    print(f"  output std: {result_no_gate.float().std():.6f}")

    # Test 4: rmsnorm_fn WITHOUT group_size (full hidden dim)
    with torch.no_grad():
        result_no_group = rmsnorm_fn(
            x=scan_flat, weight=norm.weight, bias=None, z=gate,
            eps=norm.variance_epsilon, group_size=scan_flat.shape[-1],
            norm_before_gate=False,
        )
    print(f"\n[rmsnorm_fn full hidden dim (group_size={scan_flat.shape[-1]})]")
    print(f"  output std: {result_no_group.float().std():.6f}")

    # Test 5: Manual implementation for comparison
    with torch.no_grad():
        result_manual_false = manual_rmsnorm_gated(
            scan_flat, norm.weight, gate, norm.variance_epsilon,
            norm.group_size, norm_before_gate=False,
        )
        result_manual_true = manual_rmsnorm_gated(
            scan_flat, norm.weight, gate, norm.variance_epsilon,
            norm.group_size, norm_before_gate=True,
        )
    print(f"\n[Manual norm_before_gate=False]")
    print(f"  output std: {result_manual_false.float().std():.6f}")
    print(f"  matches rmsnorm_fn nbg=False: {torch.allclose(result_nbg_false.float(), result_manual_false.float(), atol=1e-2)}")

    print(f"\n[Manual norm_before_gate=True]")
    print(f"  output std: {result_manual_true.float().std():.6f}")
    print(f"  matches rmsnorm_fn nbg=True: {torch.allclose(result_nbg_true.float(), result_manual_true.float(), atol=1e-2)}")

    # Test 6: Bamba-style norm (no group_size, full hidden dim variance)
    with torch.no_grad():
        x = scan_flat.float()
        g = gate.float()
        x_gated = x * torch.nn.functional.silu(g)
        variance = x_gated.pow(2).mean(-1, keepdim=True)
        x_normed = x_gated * torch.rsqrt(variance + norm.variance_epsilon)
        result_bamba_style = (norm.weight.float() * x_normed).to(dtype)
    print(f"\n[Bamba-style norm (gate first, full-dim variance)]")
    print(f"  output std: {result_bamba_style.float().std():.6f}")

    # Test 7: out_proj on each
    with torch.no_grad():
        out_nbg_false = mixer.out_proj(result_nbg_false)
        out_nbg_true = mixer.out_proj(result_nbg_true)
        out_bamba = mixer.out_proj(result_bamba_style)
    print(f"\n[After out_proj]")
    print(f"  norm_before_gate=False: std={out_nbg_false.float().std():.6f}")
    print(f"  norm_before_gate=True:  std={out_nbg_true.float().std():.6f}")
    print(f"  bamba-style:            std={out_bamba.float().std():.6f}")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
