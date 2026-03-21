"""
Diagnostic script for NemotronH merge inconsistency.
Checks weight tying, forward pass sanity, and compares with v4.51.0 behavior.

Usage:
    python experiments/diagnose_nemotronh.py --model <path_to_model>
"""
import argparse
import torch
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("NemotronH Diagnostic")
    print("=" * 60)

    # 1. Load config and check key settings
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    print(f"\n[Config]")
    print(f"  model_type:          {config.model_type}")
    print(f"  tie_word_embeddings: {config.tie_word_embeddings}")
    print(f"  num_hidden_layers:   {config.num_hidden_layers}")
    print(f"  hybrid_override_pattern: {config.hybrid_override_pattern[:30]}...")
    print(f"  _attn_implementation (internal): {getattr(config, '_attn_implementation_internal', 'N/A')}")

    # 2. Load model
    print(f"\n[Loading model...]")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
    )
    model.eval()
    model.to(device)

    # 3. Check _tied_weights_keys
    print(f"\n[Weight Tying]")
    twk = getattr(model, "_tied_weights_keys", None)
    print(f"  _tied_weights_keys type: {type(twk).__name__}")
    print(f"  _tied_weights_keys value: {twk}")
    print(f"  isinstance(dict): {isinstance(twk, dict)}")

    # 4. Check if lm_head and embedding share the same tensor
    backbone = getattr(model, "backbone", getattr(model, "model", None))
    if backbone is not None:
        emb = backbone.embeddings if hasattr(backbone, "embeddings") else getattr(backbone, "embed_tokens", None)
        if emb is not None:
            emb_weight = emb.weight
            lm_weight = model.lm_head.weight
            same_data = emb_weight.data_ptr() == lm_weight.data_ptr()
            cos_sim = torch.nn.functional.cosine_similarity(
                emb_weight.float().flatten().unsqueeze(0),
                lm_weight.float().flatten().unsqueeze(0),
            ).item()
            print(f"  embedding shape:   {emb_weight.shape}")
            print(f"  lm_head shape:     {lm_weight.shape}")
            print(f"  same data_ptr:     {same_data}")
            print(f"  cosine similarity: {cos_sim:.6f}")
            if not same_data and cos_sim < 0.9:
                print(f"  *** WARNING: lm_head.weight is NOT tied and NOT similar to embeddings! ***")
                print(f"  *** This likely means lm_head is randomly initialized -> garbage output ***")
            elif not same_data and cos_sim > 0.99:
                print(f"  Weights are not tied (separate tensors) but have same values (loaded from checkpoint)")
        else:
            print(f"  Could not find embedding layer")
    else:
        print(f"  Could not find backbone")

    # 5. Check NemotronHBlock structure
    print(f"\n[Block Structure]")
    if backbone is not None:
        for i, layer in enumerate(backbone.layers[:5]):
            bt = getattr(layer, "block_type", "?")
            mixer_type = type(layer.mixer).__name__
            print(f"  Layer {i}: block_type={bt}, mixer={mixer_type}")
        print(f"  ... ({len(backbone.layers)} layers total)")

    # 6. Forward pass sanity check
    print(f"\n[Forward Pass Sanity Check]")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    test_text = "The capital of France is"
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    print(f"  Input: '{test_text}'")
    print(f"  Token IDs: {input_ids[0].tolist()}")

    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=False)

    logits = outputs.logits
    print(f"  Output logits shape: {logits.shape}")
    print(f"  Logits stats: mean={logits.float().mean():.4f}, std={logits.float().std():.4f}, min={logits.float().min():.4f}, max={logits.float().max():.4f}")

    # Check if logits look reasonable
    last_logits = logits[0, -1, :]
    probs = torch.softmax(last_logits.float(), dim=-1)
    top5_probs, top5_ids = probs.topk(5)
    print(f"\n  Top-5 predictions for next token:")
    for prob, tok_id in zip(top5_probs, top5_ids):
        token_str = tokenizer.decode([tok_id.item()])
        print(f"    '{token_str}' (id={tok_id.item()}): {prob.item():.4f}")

    # 7. Check intermediate hidden states
    print(f"\n[Hidden State Check]")
    with torch.no_grad():
        outputs2 = model(input_ids=input_ids, use_cache=False, output_hidden_states=True)

    if outputs2.hidden_states is not None:
        for i, hs in enumerate(outputs2.hidden_states):
            if i % 10 == 0 or i == len(outputs2.hidden_states) - 1:
                hs_float = hs.float()
                print(f"  Layer {i:3d}: mean={hs_float.mean():.6f}, std={hs_float.std():.4f}, "
                      f"min={hs_float.min():.4f}, max={hs_float.max():.4f}, "
                      f"has_nan={hs_float.isnan().any().item()}, has_inf={hs_float.isinf().any().item()}")
    else:
        print(f"  hidden_states is None (output_hidden_states may not be working)")

    # 8. Perplexity on a short sequence
    print(f"\n[Quick Perplexity Check]")
    test_text2 = "The quick brown fox jumps over the lazy dog. In a world where technology advances rapidly, we must consider the implications of artificial intelligence on society."
    inputs2 = tokenizer(test_text2, return_tensors="pt").to(device)
    input_ids2 = inputs2["input_ids"]
    with torch.no_grad():
        outputs3 = model(input_ids=input_ids2, use_cache=False)

    shift_logits = outputs3.logits[:, :-1, :].contiguous().float()
    shift_labels = input_ids2[:, 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )
    ppl = torch.exp(loss).item()
    print(f"  Sequence length: {input_ids2.shape[1]}")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Perplexity: {ppl:.2f}")
    if ppl > 1000:
        print(f"  *** WARNING: Perplexity is extremely high — model output is likely garbage ***")
    elif ppl > 100:
        print(f"  *** WARNING: Perplexity is high — something may be wrong ***")
    else:
        print(f"  Perplexity looks reasonable")

    print(f"\n{'=' * 60}")
    print("Diagnostic complete.")


if __name__ == "__main__":
    main()
