"""
Diagnostic v7: Check if out_proj.weight is being rescaled after loading.
The ratio sqrt(52) = 7.21 matches the output difference exactly.

Usage:
    python experiments/diagnose_nemotronh_v7.py --model <path> --bf16
"""
import argparse
import json
import math
import os
import torch
from safetensors import safe_open


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    num_hidden_layers = 52  # NemotronH default

    print("=" * 60)
    print("NemotronH Diagnostic v7 — out_proj.weight Rescaling Check")
    print("=" * 60)

    # 1. Read out_proj.weight directly from checkpoint
    index_path = os.path.join(args.model, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    layers_to_check = [0, 2, 4, 6, 25]
    print(f"\n[1] Checkpoint vs Model comparison for out_proj.weight")
    print(f"    Expected rescale factor: 1/sqrt({num_hidden_layers}) = {1/math.sqrt(num_hidden_layers):.6f}")
    print(f"    sqrt({num_hidden_layers}) = {math.sqrt(num_hidden_layers):.4f}")

    ckpt_weights = {}
    for layer_idx in layers_to_check:
        key = f"backbone.layers.{layer_idx}.mixer.out_proj.weight"
        if key not in weight_map:
            print(f"    {key}: NOT in checkpoint")
            continue
        shard = os.path.join(args.model, weight_map[key])
        with safe_open(shard, framework="pt", device="cpu") as f:
            ckpt_weights[layer_idx] = f.get_tensor(key)

    # 2. Load model
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=dtype,
        attn_implementation="flash_attention_2",
    )
    model.eval()

    print(f"\n    rescale_prenorm_residual: {model.config.rescale_prenorm_residual}")

    # 3. Compare
    for layer_idx in layers_to_check:
        if layer_idx not in ckpt_weights:
            continue
        key = f"backbone.layers.{layer_idx}.mixer.out_proj.weight"
        ckpt_w = ckpt_weights[layer_idx].float()
        model_w = model.backbone.layers[layer_idx].mixer.out_proj.weight.float()

        ratio = (ckpt_w.std() / model_w.std()).item()
        cos_sim = torch.nn.functional.cosine_similarity(
            ckpt_w.flatten().unsqueeze(0),
            model_w.flatten().unsqueeze(0),
        ).item()

        # Check if model_w ≈ ckpt_w / sqrt(N)
        scaled_ckpt = ckpt_w / math.sqrt(num_hidden_layers)
        match_scaled = torch.allclose(model_w, scaled_ckpt, atol=1e-3)
        match_exact = torch.allclose(model_w, ckpt_w, atol=1e-3)

        print(f"\n    Layer {layer_idx} out_proj.weight:")
        print(f"      Checkpoint: std={ckpt_w.std():.6f}, first3={ckpt_w.flatten()[:3].tolist()}")
        print(f"      Model:      std={model_w.std():.6f}, first3={model_w.flatten()[:3].tolist()}")
        print(f"      Ratio (ckpt/model): {ratio:.4f} (sqrt(52)={math.sqrt(52):.4f})")
        print(f"      Cosine sim: {cos_sim:.6f}")
        print(f"      Exact match: {match_exact}")
        print(f"      Match after /sqrt(52): {match_scaled}")
        if match_scaled and not match_exact:
            print(f"      *** CONFIRMED: out_proj.weight was divided by sqrt({num_hidden_layers}) after loading! ***")
        elif match_exact:
            print(f"      Weight matches checkpoint exactly.")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
