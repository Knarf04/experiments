"""
Check if Bamba's _init_weights overwrites dt_bias, A_log, D after checkpoint loading.

Usage:
    python experiments/diagnose_bamba_weights.py --model <path> --bf16
"""
import argparse
import json
import os
import torch
import math
from safetensors import safe_open


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.bf16 else torch.float16

    print("=" * 60)
    print("Bamba _init_weights Overwrite Check")
    print("=" * 60)

    # 1. Read checkpoint values for dt_bias, A_log, D
    index_path = os.path.join(args.model, "model.safetensors.index.json")
    single_path = os.path.join(args.model, "model.safetensors")

    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        def load_tensor(key):
            shard = os.path.join(args.model, weight_map[key])
            with safe_open(shard, framework="pt", device="cpu") as f:
                return f.get_tensor(key)
        all_keys = set(weight_map.keys())
    elif os.path.exists(single_path):
        _sf = safe_open(single_path, framework="pt", device="cpu")
        all_keys = set(_sf.keys())
        def load_tensor(key):
            with safe_open(single_path, framework="pt", device="cpu") as f:
                return f.get_tensor(key)
    else:
        raise FileNotFoundError("No safetensors checkpoint found")

    layers_to_check = [0, 1, 9, 18]
    params_to_check = ["mamba.dt_bias", "mamba.A_log", "mamba.D"]

    print(f"\n[1] Checkpoint values for dt_bias, A_log, D")
    ckpt_values = {}
    for layer_idx in layers_to_check:
        for param_suffix in params_to_check:
            for prefix in [f"model.layers.{layer_idx}.{param_suffix}",
                          f"backbone.layers.{layer_idx}.{param_suffix}"]:
                if prefix in all_keys:
                    tensor = load_tensor(prefix)
                    ckpt_values[prefix] = tensor
                    print(f"  {prefix}: shape={tensor.shape}, mean={tensor.float().mean():.6f}, "
                          f"std={tensor.float().std():.6f}, first5={tensor.flatten()[:5].float().tolist()}")

    # 2. Check what _init_weights would set them to
    # For BambaMixer with num_heads heads:
    # dt_bias -> fill_(1.0)
    # A_log -> log(arange(1, num_heads+1))
    # D -> fill_(1.0)
    print(f"\n[2] What _init_weights would set:")
    print(f"  dt_bias -> fill_(1.0) = all ones")
    print(f"  D -> fill_(1.0) = all ones")
    # Find num_heads from any A_log
    for key, tensor in ckpt_values.items():
        if "A_log" in key:
            num_heads = tensor.shape[0]
            init_A_log = torch.log(torch.arange(1, num_heads + 1).float())
            print(f"  A_log -> log(1..{num_heads}) first5={init_A_log[:5].tolist()}")
            # Check if checkpoint A_log matches init
            match = torch.allclose(tensor.float(), init_A_log, atol=1e-3)
            print(f"  A_log checkpoint matches init: {match}")
            break

    # 3. Load model and compare
    print(f"\n[3] Loading model and comparing...")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=dtype,
        attn_implementation="flash_attention_2",
    )
    model.eval()

    print(f"\n[4] Checkpoint vs Model comparison")
    model_state = model.state_dict()
    for key, ckpt_tensor in ckpt_values.items():
        if key not in model_state:
            print(f"  {key}: NOT in model")
            continue
        model_tensor = model_state[key].float().cpu()
        ckpt_tensor_f = ckpt_tensor.float()

        match = torch.allclose(model_tensor, ckpt_tensor_f, atol=1e-3)
        cos = torch.nn.functional.cosine_similarity(
            model_tensor.flatten().unsqueeze(0),
            ckpt_tensor_f.flatten().unsqueeze(0),
        ).item()

        status = "MATCH" if match else "OVERWRITTEN!"
        print(f"\n  {key}: [{status}]")
        print(f"    Checkpoint: mean={ckpt_tensor_f.mean():.6f}, std={ckpt_tensor_f.std():.6f}, "
              f"first5={ckpt_tensor_f.flatten()[:5].tolist()}")
        print(f"    Model:      mean={model_tensor.mean():.6f}, std={model_tensor.std():.6f}, "
              f"first5={model_tensor.flatten()[:5].tolist()}")

        if not match:
            # Check if model value matches _init_weights default
            if "dt_bias" in key or ".D" in key:
                is_ones = torch.allclose(model_tensor, torch.ones_like(model_tensor), atol=1e-3)
                print(f"    Model is all-ones (init default): {is_ones}")
            elif "A_log" in key:
                num_heads = model_tensor.shape[0]
                init_val = torch.log(torch.arange(1, num_heads + 1).float())
                is_init = torch.allclose(model_tensor, init_val, atol=1e-3)
                print(f"    Model matches init default log(1..N): {is_init}")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
