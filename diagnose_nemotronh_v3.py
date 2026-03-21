"""
Diagnostic v3: Check if checkpoint weights are actually loaded into the model.
Compare checkpoint tensor values directly with model parameter values.

Usage:
    python experiments/diagnose_nemotronh_v3.py --model <path_to_model> --bf16
"""
import argparse
import json
import os
import torch
from safetensors import safe_open


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.bf16 else torch.float16

    print("=" * 60)
    print("NemotronH Diagnostic v3 — Weight Loading Verification")
    print("=" * 60)

    # 1. Read checkpoint index to find shard locations
    index_path = os.path.join(args.model, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index.get("weight_map", {})

    # 2. Spot-check a few key weights directly from checkpoint
    keys_to_check = [
        "lm_head.weight",
        "backbone.embeddings.weight",
        "backbone.layers.0.norm.weight",
        "backbone.layers.0.mixer.in_proj.weight",
        "backbone.layers.4.mixer.q_proj.weight",   # attention layer
        "backbone.layers.25.mixer.in_proj.weight",  # mid-model mamba layer
    ]

    print(f"\n[1] Reading weights directly from checkpoint files...")
    ckpt_stats = {}
    for key in keys_to_check:
        if key not in weight_map:
            print(f"  {key}: NOT in checkpoint")
            continue
        shard_file = os.path.join(args.model, weight_map[key])
        with safe_open(shard_file, framework="pt", device="cpu") as f:
            tensor = f.get_tensor(key)
        stats = {
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "mean": tensor.float().mean().item(),
            "std": tensor.float().std().item(),
            "abs_max": tensor.float().abs().max().item(),
            "first_5": tensor.flatten()[:5].float().tolist(),
        }
        ckpt_stats[key] = stats
        print(f"  {key}: shape={stats['shape']}, dtype={stats['dtype']}, "
              f"mean={stats['mean']:.6f}, std={stats['std']:.4f}, abs_max={stats['abs_max']:.4f}")

    # 3. Load model
    print(f"\n[2] Loading model...")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
    )
    model.eval()

    # 4. Compare checkpoint values with model parameters
    print(f"\n[3] Comparing checkpoint values with loaded model parameters...")
    model_state = model.state_dict()

    all_match = True
    for key in keys_to_check:
        if key not in ckpt_stats:
            continue
        if key not in model_state:
            print(f"  {key}: NOT in model state_dict!")
            all_match = False
            continue

        model_tensor = model_state[key]
        m_mean = model_tensor.float().mean().item()
        m_std = model_tensor.float().std().item()
        m_abs_max = model_tensor.float().abs().max().item()
        m_first5 = model_tensor.flatten()[:5].float().tolist()

        c = ckpt_stats[key]
        match = (
            abs(m_mean - c["mean"]) < 1e-3
            and abs(m_std - c["std"]) < 1e-3
        )
        status = "MATCH" if match else "MISMATCH !!!"
        if not match:
            all_match = False

        print(f"  {key}: [{status}]")
        print(f"    Checkpoint: mean={c['mean']:.6f}, std={c['std']:.4f}, abs_max={c['abs_max']:.4f}, first5={[f'{v:.4f}' for v in c['first_5']]}")
        print(f"    Model:      mean={m_mean:.6f}, std={m_std:.4f}, abs_max={m_abs_max:.4f}, first5={[f'{v:.4f}' for v in m_first5]}")

    if all_match:
        print(f"\n  All checked weights match checkpoint — loading is correct.")
        print(f"  Issue must be in the forward pass computation.")
    else:
        print(f"\n  *** WEIGHTS DO NOT MATCH — loading is broken! ***")

    # 5. Check all model param names vs checkpoint keys
    print(f"\n[4] Key name comparison...")
    model_keys = set(model_state.keys())
    ckpt_keys = set(weight_map.keys())

    missing_in_model = ckpt_keys - model_keys
    missing_in_ckpt = model_keys - ckpt_keys

    if missing_in_model:
        print(f"  Keys in checkpoint but NOT in model ({len(missing_in_model)}):")
        for k in sorted(missing_in_model)[:20]:
            print(f"    {k}")
        if len(missing_in_model) > 20:
            print(f"    ... and {len(missing_in_model) - 20} more")

    if missing_in_ckpt:
        print(f"  Keys in model but NOT in checkpoint ({len(missing_in_ckpt)}):")
        for k in sorted(missing_in_ckpt)[:20]:
            print(f"    {k}")
        if len(missing_in_ckpt) > 20:
            print(f"    ... and {len(missing_in_ckpt) - 20} more")

    if not missing_in_model and not missing_in_ckpt:
        print(f"  All {len(model_keys)} keys match perfectly.")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
