"""
Diagnostic v4: Hook into individual blocks to find where computation diverges.
Compare per-block input/output and identify the first problematic layer.

Usage:
    python experiments/diagnose_nemotronh_v4.py --model <path_to_model> --bf16
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

    print("=" * 60)
    print("NemotronH Diagnostic v4 — Per-Block Analysis")
    print("=" * 60)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=dtype,
        attn_implementation="flash_attention_2",
    )
    model.eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Check what attention implementation is actually set
    print(f"\n[Config]")
    print(f"  config._attn_implementation: {model.config._attn_implementation}")
    print(f"  _attn_implementation_internal: {getattr(model.config, '_attn_implementation_internal', 'N/A')}")

    # Check what ALL_ATTENTION_FUNCTIONS contains
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    print(f"  ALL_ATTENTION_FUNCTIONS keys: {list(ALL_ATTENTION_FUNCTIONS._global_mapping.keys())}")
    fa2_fn = ALL_ATTENTION_FUNCTIONS._global_mapping.get("flash_attention_2")
    print(f"  flash_attention_2 function: {fa2_fn}")

    # Hook into each block to capture inputs/outputs
    block_data = {}

    def make_hook(layer_idx, block_type):
        def hook_fn(module, input, output):
            inp = input[0] if isinstance(input, tuple) else input
            out = output[0] if isinstance(output, tuple) else output
            block_data[layer_idx] = {
                "block_type": block_type,
                "input_mean": inp.float().mean().item(),
                "input_std": inp.float().std().item(),
                "output_mean": out.float().mean().item(),
                "output_std": out.float().std().item(),
                "input_abs_max": inp.float().abs().max().item(),
                "output_abs_max": out.float().abs().max().item(),
                "ratio_std": out.float().std().item() / (inp.float().std().item() + 1e-10),
            }
        return hook_fn

    hooks = []
    for i, layer in enumerate(model.backbone.layers):
        h = layer.mixer.register_forward_hook(make_hook(i, layer.block_type))
        hooks.append(h)

    # Run forward pass
    test_text = "The capital of France is"
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(input_ids=inputs["input_ids"], use_cache=False)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Print per-block analysis
    print(f"\n[Per-Block Mixer Analysis]")
    print(f"{'Layer':>5} {'Type':>10} {'In_std':>10} {'Out_std':>10} {'Ratio':>10} {'Out_absmax':>12} {'Flag':>10}")
    for i in sorted(block_data.keys()):
        d = block_data[i]
        flag = ""
        if d["ratio_std"] > 10:
            flag = "BLOW-UP"
        elif d["ratio_std"] < 0.01:
            flag = "COLLAPSE"
        elif d["output_std"] > 100:
            flag = "LARGE"
        print(f"{i:5d} {d['block_type']:>10} {d['input_std']:10.4f} {d['output_std']:10.4f} "
              f"{d['ratio_std']:10.4f} {d['output_abs_max']:12.4f} {flag:>10}")

    # Also run on the v4.51.0 branch's forward path manually to check
    # Test: what if we bypass the NemotronHBlock and call mixer directly?
    print(f"\n[Direct Mamba Layer 0 Test]")
    layer0 = model.backbone.layers[0]
    emb = model.backbone.embeddings(inputs["input_ids"].to(device))
    normed = layer0.norm(emb.to(dtype=layer0.norm.weight.dtype))
    print(f"  Embedding: mean={emb.float().mean():.6f}, std={emb.float().std():.6f}")
    print(f"  After norm: mean={normed.float().mean():.6f}, std={normed.float().std():.6f}")

    with torch.no_grad():
        mamba_out = layer0.mixer(normed, cache_params=None, cache_position=torch.arange(normed.shape[1], device=device))
    print(f"  Mamba out: mean={mamba_out.float().mean():.6f}, std={mamba_out.float().std():.6f}")
    result = emb + mamba_out  # residual
    print(f"  After residual: mean={result.float().mean():.6f}, std={result.float().std():.6f}")

    # Check the first attention layer
    attn_layers = [i for i, l in enumerate(model.backbone.layers) if l.block_type == "attention"]
    if attn_layers:
        first_attn = attn_layers[0]
        print(f"\n[First Attention Layer ({first_attn}) Test]")
        attn_layer = model.backbone.layers[first_attn]
        # We need the hidden_states that reach this layer
        # Run model up to this layer
        hidden = model.backbone.embeddings(inputs["input_ids"].to(device))
        cache_position = torch.arange(hidden.shape[1], device=device)
        for i in range(first_attn):
            bl = model.backbone.layers[i]
            residual = hidden
            normed = bl.norm(hidden.to(dtype=bl.norm.weight.dtype))
            if bl.residual_in_fp32:
                residual = residual.to(torch.float32)
            with torch.no_grad():
                if bl.block_type == "mamba":
                    out = bl.mixer(normed, cache_params=None, cache_position=cache_position)
                elif bl.block_type == "mlp":
                    out = bl.mixer(normed)
            hidden = residual + out

        print(f"  Input hidden: mean={hidden.float().mean():.6f}, std={hidden.float().std():.6f}")
        residual = hidden
        normed = attn_layer.norm(hidden.to(dtype=attn_layer.norm.weight.dtype))
        print(f"  After norm: mean={normed.float().mean():.6f}, std={normed.float().std():.6f}")

        with torch.no_grad():
            attn_out = attn_layer.mixer(
                normed, attention_mask=None, past_key_values=None,
                cache_position=cache_position,
            )
        attn_hidden = attn_out[0]
        print(f"  Attention out: mean={attn_hidden.float().mean():.6f}, std={attn_hidden.float().std():.6f}")
        result = residual + attn_hidden
        print(f"  After residual: mean={result.float().mean():.6f}, std={result.float().std():.6f}")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
