"""
Diagnostic for Bamba model — check weight loading, forward pass, and logits.

Usage:
    python experiments/diagnose_bamba.py --model <path_to_bamba_model> --bf16
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
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("Bamba Diagnostic")
    print("=" * 60)

    # 1. Config
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    print(f"\n[Config]")
    print(f"  model_type:          {config.model_type}")
    print(f"  tie_word_embeddings: {config.tie_word_embeddings}")
    print(f"  num_hidden_layers:   {config.num_hidden_layers}")
    print(f"  attn_layer_indices:  {getattr(config, 'attn_layer_indices', 'N/A')}")
    print(f"  rope_parameters:     {getattr(config, 'rope_parameters', 'N/A')}")
    print(f"  partial_rotary_factor: {getattr(config, 'partial_rotary_factor', 'N/A')}")

    # 2. Load model
    print(f"\n[Loading model...]")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=dtype,
        attn_implementation="flash_attention_2",
    )
    model.eval().to(device)

    # 3. Weight tying check
    print(f"\n[Weight Tying]")
    twk = getattr(model, "_tied_weights_keys", None)
    print(f"  _tied_weights_keys: {twk}")

    backbone = getattr(model, "model", getattr(model, "backbone", None))
    if backbone is not None:
        emb = getattr(backbone, "embed_tokens", getattr(backbone, "embeddings", None))
        if emb is not None:
            emb_weight = emb.weight
            lm_weight = model.lm_head.weight
            same_data = emb_weight.data_ptr() == lm_weight.data_ptr()
            cos_sim = torch.nn.functional.cosine_similarity(
                emb_weight.float().flatten().unsqueeze(0),
                lm_weight.float().flatten().unsqueeze(0),
            ).item()
            print(f"  same data_ptr:     {same_data}")
            print(f"  cosine similarity: {cos_sim:.6f}")

    # 4. Spot-check key weights against checkpoint
    print(f"\n[Weight Loading Check]")
    index_path = os.path.join(args.model, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]

        model_state = model.state_dict()
        keys_to_check = [k for k in list(weight_map.keys())[:5]]
        # Also check out_proj if it exists
        out_proj_keys = [k for k in weight_map if "out_proj.weight" in k]
        if out_proj_keys:
            keys_to_check.extend(out_proj_keys[:2])

        for key in keys_to_check:
            if key not in model_state:
                print(f"  {key}: NOT in model state_dict!")
                continue
            shard = os.path.join(args.model, weight_map[key])
            with safe_open(shard, framework="pt", device="cpu") as f:
                ckpt_w = f.get_tensor(key)
            model_w = model_state[key].float().cpu()
            ckpt_w = ckpt_w.float()
            ratio = (ckpt_w.std() / model_w.std()).item() if model_w.std() > 0 else float('inf')
            match = torch.allclose(model_w, ckpt_w, atol=1e-3)
            status = "MATCH" if match else f"MISMATCH (ratio={ratio:.4f})"
            print(f"  {key}: [{status}]")

    # 5. Forward pass
    print(f"\n[Forward Pass]")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    test_text = "The capital of France is"
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    print(f"  Input: '{test_text}'")

    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=False)

    logits = outputs.logits
    print(f"  Logits: mean={logits.float().mean():.4f}, std={logits.float().std():.4f}")

    last_logits = logits[0, -1, :]
    probs = torch.softmax(last_logits.float(), dim=-1)
    top5_probs, top5_ids = probs.topk(5)
    print(f"\n  Top-5 predictions:")
    for prob, tok_id in zip(top5_probs, top5_ids):
        token_str = tokenizer.decode([tok_id.item()])
        print(f"    '{token_str}' (id={tok_id.item()}): {prob.item():.4f}")

    # 6. Per-block analysis
    print(f"\n[Per-Block Mixer Analysis]")
    block_data = {}

    def make_hook(layer_idx, layer_type):
        def hook_fn(module, input, output):
            try:
                inp = input[0] if isinstance(input, (tuple, list)) and len(input) > 0 else input
                if isinstance(inp, (tuple, list)):
                    inp = inp[0] if len(inp) > 0 else None
                out = output[0] if isinstance(output, (tuple, list)) and len(output) > 0 else output
                if isinstance(out, (tuple, list)):
                    out = out[0] if len(out) > 0 else None
                if inp is not None and out is not None and isinstance(inp, torch.Tensor) and isinstance(out, torch.Tensor):
                    block_data[layer_idx] = {
                        "type": layer_type,
                        "in_std": inp.float().std().item(),
                        "out_std": out.float().std().item(),
                    }
            except Exception:
                pass
        return hook_fn

    hooks = []
    layers = backbone.layers
    for i, layer in enumerate(layers):
        lt = getattr(layer, "layer_type", getattr(layer, "block_type", "?"))
        mixer = getattr(layer, "mamba", getattr(layer, "self_attn", getattr(layer, "mixer", None)))
        if mixer is not None:
            hooks.append(mixer.register_forward_hook(make_hook(i, lt)))

    with torch.no_grad():
        outputs2 = model(input_ids=input_ids, use_cache=False)

    for h in hooks:
        h.remove()

    print(f"  {'Layer':>5} {'Type':>10} {'In_std':>10} {'Out_std':>10} {'Ratio':>10}")
    for i in sorted(block_data.keys())[:20]:
        d = block_data[i]
        ratio = d["out_std"] / (d["in_std"] + 1e-10)
        print(f"  {i:5d} {d['type']:>10} {d['in_std']:10.4f} {d['out_std']:10.4f} {ratio:10.4f}")
    if len(block_data) > 20:
        print(f"  ... ({len(block_data)} layers total)")

    # 7. Perplexity
    print(f"\n[Quick Perplexity]")
    test_text2 = "The quick brown fox jumps over the lazy dog. In a world where technology advances rapidly, we must consider the implications of artificial intelligence on society."
    inputs2 = tokenizer(test_text2, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs3 = model(input_ids=inputs2["input_ids"], use_cache=False)
    shift_logits = outputs3.logits[:, :-1, :].contiguous().float()
    shift_labels = inputs2["input_ids"][:, 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )
    ppl = torch.exp(loss).item()
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Perplexity: {ppl:.2f}")
    if ppl > 1000:
        print(f"  *** WARNING: Perplexity extremely high ***")
    elif ppl > 100:
        print(f"  *** WARNING: Perplexity high ***")
    else:
        print(f"  Perplexity looks reasonable")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
