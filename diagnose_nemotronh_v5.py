"""
Diagnostic v5: Instrument the Mamba mixer to find exactly where the output collapses.
Check fast path availability and step through the computation.

Usage:
    python experiments/diagnose_nemotronh_v5.py --model <path_to_model> --bf16
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=dtype,
        attn_implementation="flash_attention_2",
    )
    model.eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print("=" * 60)
    print("NemotronH Diagnostic v5 — Mamba Mixer Internals")
    print("=" * 60)

    # Check fast path
    from transformers.models.nemotron_h.modeling_nemotron_h import (
        is_fast_path_available,
        selective_state_update,
        mamba_chunk_scan_combined,
    )
    from transformers.utils.import_utils import is_mamba_2_ssm_available, is_causal_conv1d_available
    print(f"\n[Fast Path Check]")
    print(f"  is_mamba_2_ssm_available: {is_mamba_2_ssm_available()}")
    print(f"  is_causal_conv1d_available: {is_causal_conv1d_available()}")
    print(f"  is_fast_path_available: {is_fast_path_available}")
    print(f"  selective_state_update: {selective_state_update}")
    print(f"  mamba_chunk_scan_combined: {mamba_chunk_scan_combined}")

    try:
        from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
        print(f"  causal_conv1d_fn: {causal_conv1d_fn}")
        print(f"  causal_conv1d_update: {causal_conv1d_update}")
    except ImportError as e:
        print(f"  causal_conv1d import FAILED: {e}")

    # Get the first mamba layer
    layer0 = model.backbone.layers[0]
    mixer = layer0.mixer
    print(f"\n[Mamba Layer 0 Config]")
    print(f"  num_heads: {mixer.num_heads}")
    print(f"  head_dim: {mixer.head_dim}")
    print(f"  intermediate_size: {mixer.intermediate_size}")
    print(f"  n_groups: {mixer.n_groups}")
    print(f"  ssm_state_size: {mixer.ssm_state_size}")
    print(f"  conv_dim: {mixer.conv_dim}")
    print(f"  chunk_size: {mixer.chunk_size}")
    print(f"  in_proj weight device: {mixer.in_proj.weight.device}")

    # Prepare input
    test_text = "The capital of France is"
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    emb = model.backbone.embeddings(inputs["input_ids"])
    normed = layer0.norm(emb.to(dtype=layer0.norm.weight.dtype))

    print(f"\n[Step-by-step Mamba Layer 0]")
    print(f"  Input (after norm): shape={normed.shape}, std={normed.float().std():.6f}")

    with torch.no_grad():
        # Step 1: in_proj
        projected = mixer.in_proj(normed)
        print(f"  After in_proj: shape={projected.shape}, std={projected.float().std():.6f}")

        batch_size, seq_len, _ = normed.shape
        d_mlp = (projected.shape[-1] - mixer.intermediate_size - mixer.conv_dim - mixer.num_heads) // 2
        print(f"  d_mlp: {d_mlp}")

        # Step 2: Split
        _, _, gate, hidden_states_B_C, dt = projected.split(
            [d_mlp, d_mlp, mixer.intermediate_size, mixer.conv_dim, mixer.num_heads], dim=-1
        )
        print(f"  gate: shape={gate.shape}, std={gate.float().std():.6f}")
        print(f"  hidden_states_B_C: shape={hidden_states_B_C.shape}, std={hidden_states_B_C.float().std():.6f}")
        print(f"  dt: shape={dt.shape}, std={dt.float().std():.6f}, mean={dt.float().mean():.6f}")

        # Step 3: Conv1d
        from causal_conv1d import causal_conv1d_fn
        hidden_states_B_C_conv = causal_conv1d_fn(
            x=hidden_states_B_C.transpose(1, 2),
            weight=mixer.conv1d.weight.squeeze(1),
            bias=mixer.conv1d.bias,
            activation=mixer.activation,
        ).transpose(1, 2)
        print(f"  After conv1d: shape={hidden_states_B_C_conv.shape}, std={hidden_states_B_C_conv.float().std():.6f}")

        # Step 4: Split into hidden_states, B, C
        groups_time_state_size = mixer.n_groups * mixer.ssm_state_size
        hidden_states_split, B, C = torch.split(
            hidden_states_B_C_conv,
            [mixer.intermediate_size, groups_time_state_size, groups_time_state_size],
            dim=-1,
        )
        print(f"  hidden_states: shape={hidden_states_split.shape}, std={hidden_states_split.float().std():.6f}")
        print(f"  B: shape={B.shape}, std={B.float().std():.6f}")
        print(f"  C: shape={C.shape}, std={C.float().std():.6f}")

        # Step 5: Reshape for SSM
        hidden_states_4d = hidden_states_split.view(batch_size, seq_len, -1, mixer.head_dim)
        A = -torch.exp(mixer.A_log.float())
        print(f"  A: {A[:5].tolist()}")
        print(f"  dt_bias: mean={mixer.dt_bias.float().mean():.4f}, std={mixer.dt_bias.float().std():.4f}")

        # Step 6: SSM scan
        scan_output, ssm_state = mamba_chunk_scan_combined(
            hidden_states_4d,
            dt,
            A,
            B.view(batch_size, seq_len, mixer.n_groups, -1),
            C.view(batch_size, seq_len, mixer.n_groups, -1),
            chunk_size=mixer.chunk_size,
            D=mixer.D,
            z=None,
            seq_idx=None,
            return_final_states=True,
            dt_bias=mixer.dt_bias,
            dt_softplus=True,
        )
        print(f"  After SSM scan: shape={scan_output.shape}, std={scan_output.float().std():.6f}")

        # Step 7: Reshape and norm
        scan_flat = scan_output.view(batch_size, seq_len, -1)
        print(f"  scan_flat: shape={scan_flat.shape}, std={scan_flat.float().std():.6f}")
        normed_out = mixer.norm(scan_flat, gate)
        print(f"  After gated norm: shape={normed_out.shape}, std={normed_out.float().std():.6f}")

        # Step 8: out_proj
        out = mixer.out_proj(normed_out)
        print(f"  After out_proj: shape={out.shape}, std={out.float().std():.6f}")

        # Compare with actual mixer output
        actual_out = mixer(normed, cache_params=None, cache_position=torch.arange(seq_len, device=device))
        print(f"\n  Actual mixer output: std={actual_out.float().std():.6f}")
        print(f"  Manual vs actual match: {torch.allclose(out, actual_out, atol=1e-3)}")

        # Check which path the mixer actually takes
        print(f"\n  Path check: is_fast_path_available={is_fast_path_available}, "
              f"device_is_cuda={'cuda' in mixer.in_proj.weight.device.type}")
        if is_fast_path_available and "cuda" in mixer.in_proj.weight.device.type:
            print(f"  -> Using cuda_kernels_forward")
        else:
            print(f"  -> Using torch_forward (SLOW PATH)")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
