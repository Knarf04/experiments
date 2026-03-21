"""
Diagnostic: Compare outputs between v4.51.0_backup and main branches.
Run this on BOTH branches with the same model to find where they diverge.

Usage:
    # On main branch:
    python experiments/diagnose_compare_branches.py --model <path> --bf16 --save main_outputs.pt

    # On v4.51.0_backup branch:
    python experiments/diagnose_compare_branches.py --model <path> --bf16 --save v451_outputs.pt

    # Compare:
    python experiments/diagnose_compare_branches.py --compare main_outputs.pt v451_outputs.pt
"""
import argparse
import torch
import sys


def run_model(args):
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=dtype,
        attn_implementation="flash_attention_2",
    )
    model.eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    test_text = "The capital of France is"
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    results = {"input_ids": input_ids.cpu()}

    # Capture per-layer hidden states
    layer_outputs = {}
    def make_block_hook(idx):
        def hook(module, inp, out):
            inp_tensor = inp[0] if isinstance(inp, tuple) else inp
            out_tensor = out[0] if isinstance(out, tuple) else out
            layer_outputs[idx] = {
                "input": inp_tensor.cpu().float(),
                "output": out_tensor.cpu().float(),
            }
        return hook

    hooks = []
    backbone = model.backbone if hasattr(model, "backbone") else model.model
    for i, layer in enumerate(backbone.layers):
        hooks.append(layer.register_forward_hook(make_block_hook(i)))

    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=False)

    for h in hooks:
        h.remove()

    # Save key outputs
    results["logits"] = outputs.logits.cpu().float()
    results["logits_stats"] = {
        "mean": outputs.logits.float().mean().item(),
        "std": outputs.logits.float().std().item(),
    }

    # Save per-layer stats (full tensors too large)
    for i, data in layer_outputs.items():
        results[f"layer_{i}_input_mean"] = data["input"].mean().item()
        results[f"layer_{i}_input_std"] = data["input"].std().item()
        results[f"layer_{i}_output_mean"] = data["output"].mean().item()
        results[f"layer_{i}_output_std"] = data["output"].std().item()
        # Save first few values for exact comparison
        results[f"layer_{i}_output_first10"] = data["output"][0, 0, :10].tolist()

    # Also save embedding output and final norm output
    emb = backbone.embeddings(input_ids) if hasattr(backbone, "embeddings") else backbone.embed_tokens(input_ids)
    results["embedding_first10"] = emb[0, 0, :10].cpu().float().tolist()
    results["embedding_std"] = emb.float().std().item()

    n_layers = len(backbone.layers)
    results["num_layers"] = n_layers
    results["final_logits_first10"] = outputs.logits[0, -1, :10].cpu().float().tolist()

    # Top-5 predictions
    probs = torch.softmax(outputs.logits[0, -1].float(), dim=-1)
    top5_probs, top5_ids = probs.topk(5)
    results["top5"] = [(tokenizer.decode([tid.item()]), tid.item(), p.item())
                       for tid, p in zip(top5_ids, top5_probs)]

    torch.save(results, args.save)
    print(f"Saved to {args.save}")
    print(f"Logits: mean={results['logits_stats']['mean']:.4f}, std={results['logits_stats']['std']:.4f}")
    print(f"Top-5: {results['top5']}")
    for i in range(min(5, n_layers)):
        print(f"  Layer {i}: in_std={results[f'layer_{i}_input_std']:.6f}, "
              f"out_std={results[f'layer_{i}_output_std']:.6f}, "
              f"first3_out={results[f'layer_{i}_output_first10'][:3]}")


def compare(args):
    a = torch.load(args.compare[0], weights_only=False)
    b = torch.load(args.compare[1], weights_only=False)

    print(f"Comparing {args.compare[0]} vs {args.compare[1]}")
    print(f"\nEmbedding:")
    print(f"  A: std={a['embedding_std']:.6f}, first10={[f'{v:.4f}' for v in a['embedding_first10']]}")
    print(f"  B: std={b['embedding_std']:.6f}, first10={[f'{v:.4f}' for v in b['embedding_first10']]}")

    n = min(a["num_layers"], b["num_layers"])
    print(f"\nPer-layer comparison:")
    first_diverge = None
    for i in range(n):
        a_out = a[f"layer_{i}_output_first10"]
        b_out = b[f"layer_{i}_output_first10"]
        match = all(abs(x - y) < 1e-3 for x, y in zip(a_out[:3], b_out[:3]))
        a_std = a[f"layer_{i}_output_std"]
        b_std = b[f"layer_{i}_output_std"]
        status = "OK" if match else "DIVERGED"
        if not match and first_diverge is None:
            first_diverge = i
        print(f"  Layer {i:3d}: A_std={a_std:.6f} B_std={b_std:.6f} | "
              f"A_first3={[f'{v:.4f}' for v in a_out[:3]]} "
              f"B_first3={[f'{v:.4f}' for v in b_out[:3]]} [{status}]")

    print(f"\nLogits:")
    print(f"  A: mean={a['logits_stats']['mean']:.4f}, std={a['logits_stats']['std']:.4f}")
    print(f"  B: mean={b['logits_stats']['mean']:.4f}, std={b['logits_stats']['std']:.4f}")
    print(f"  A first10: {[f'{v:.4f}' for v in a['final_logits_first10']]}")
    print(f"  B first10: {[f'{v:.4f}' for v in b['final_logits_first10']]}")
    print(f"\nTop-5:")
    print(f"  A: {a['top5']}")
    print(f"  B: {b['top5']}")
    if first_diverge is not None:
        print(f"\n*** First divergence at layer {first_diverge} ***")
    else:
        print(f"\n*** No divergence in layer outputs — issue must be in post-processing ***")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--save", type=str)
    parser.add_argument("--compare", nargs=2, type=str)
    args = parser.parse_args()

    if args.compare:
        compare(args)
    elif args.model and args.save:
        run_model(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
