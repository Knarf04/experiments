"""
Diagnostic v8: Trace exactly WHEN _init_weights runs and whether the
rescale_prenorm_residual block executes after checkpoint loading.

Usage:
    python experiments/diagnose_nemotronh_v8.py --model <path> --bf16
"""
import argparse
import traceback
import math
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.bf16 else torch.float16

    print("=" * 60)
    print("NemotronH Diagnostic v8 — _init_weights Tracing")
    print("=" * 60)

    # Monkey-patch _init_weights to trace calls
    from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHPreTrainedModel
    import torch.nn as nn

    original_init_weights = NemotronHPreTrainedModel._init_weights
    call_count = [0]
    rescale_count = [0]

    def traced_init_weights(self, module):
        call_count[0] += 1
        # Check if this call will hit the rescale block
        if self.config.rescale_prenorm_residual:
            for name, p in module.named_parameters():
                if name in ["out_proj.weight"]:
                    is_meta = p.device.type == "meta"
                    rescale_count[0] += 1
                    print(f"\n  [RESCALE #{rescale_count[0]}] _init_weights called for {type(module).__name__}")
                    print(f"    out_proj.weight device={p.device}, is_meta={is_meta}")
                    print(f"    out_proj.weight std BEFORE: {p.float().std().item() if not is_meta else 'N/A (meta)'}")
                    print(f"    module._is_hf_initialized: {getattr(module, '_is_hf_initialized', 'NOT SET')}")
                    # Print short stack trace
                    tb = traceback.format_stack()
                    # Show last 5 frames
                    for line in tb[-6:-1]:
                        print(f"    {line.strip()}")

        # Call original
        original_init_weights(self, module)

        # Check after
        if self.config.rescale_prenorm_residual:
            for name, p in module.named_parameters():
                if name in ["out_proj.weight"]:
                    is_meta = p.device.type == "meta"
                    print(f"    out_proj.weight std AFTER:  {p.float().std().item() if not is_meta else 'N/A (meta)'}")

    NemotronHPreTrainedModel._init_weights = traced_init_weights

    # Also trace _initialize_weights
    from transformers.modeling_utils import PreTrainedModel
    original_initialize_weights_fn = PreTrainedModel._initialize_weights

    def traced_initialize_weights(self, module):
        if hasattr(module, 'out_proj') and isinstance(getattr(module, 'out_proj', None), nn.Linear):
            hf_init = getattr(module, '_is_hf_initialized', 'NOT SET')
            print(f"  [_initialize_weights] module={type(module).__name__}, _is_hf_initialized={hf_init}")
        return original_initialize_weights_fn(self, module)

    PreTrainedModel._initialize_weights = traced_initialize_weights

    # Now load the model
    from transformers import AutoModelForCausalLM
    print(f"\n[Loading model...]")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=dtype,
        attn_implementation="flash_attention_2",
    )
    model.eval()

    print(f"\n[Summary]")
    print(f"  _init_weights total calls: {call_count[0]}")
    print(f"  rescale block triggered: {rescale_count[0]} times")

    # Final check
    import json, os
    from safetensors import safe_open
    index_path = os.path.join(args.model, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]
    key = "backbone.layers.0.mixer.out_proj.weight"
    shard = os.path.join(args.model, weight_map[key])
    with safe_open(shard, framework="pt", device="cpu") as f:
        ckpt_w = f.get_tensor(key)

    model_w = model.backbone.layers[0].mixer.out_proj.weight
    ratio = (ckpt_w.float().std() / model_w.float().std()).item()
    print(f"\n  Final out_proj.weight ratio (ckpt/model): {ratio:.4f}")
    print(f"  sqrt(52) = {math.sqrt(52):.4f}")
    if abs(ratio - math.sqrt(52)) < 0.01:
        print(f"  *** STILL RESCALED after loading ***")
    elif abs(ratio - 1.0) < 0.01:
        print(f"  *** FIXED — matches checkpoint ***")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
