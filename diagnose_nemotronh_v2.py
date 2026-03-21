"""
Diagnostic v2: Check what keys the checkpoint actually has for lm_head,
and test if manually tying weights fixes the output.

Usage:
    python experiments/diagnose_nemotronh_v2.py --model <path_to_model> --bf16
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
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("NemotronH Diagnostic v2 — Checkpoint & Weight Tying")
    print("=" * 60)

    # 1. Check checkpoint index for lm_head.weight
    print(f"\n[1] Checking checkpoint for lm_head.weight...")
    index_path = os.path.join(args.model, "model.safetensors.index.json")
    single_path = os.path.join(args.model, "model.safetensors")

    lm_head_in_checkpoint = False
    lm_head_mapped_to = None

    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        if "lm_head.weight" in weight_map:
            lm_head_in_checkpoint = True
            lm_head_mapped_to = weight_map["lm_head.weight"]
            print(f"  lm_head.weight IS in checkpoint index -> {lm_head_mapped_to}")
        else:
            print(f"  lm_head.weight NOT in checkpoint index")
            # Check if it's shared with embedding
            if "backbone.embeddings.weight" in weight_map:
                print(f"  backbone.embeddings.weight -> {weight_map['backbone.embeddings.weight']}")

        # Print all keys containing 'lm_head' or 'embed'
        print(f"\n  Keys matching 'lm_head' or 'embed':")
        for key in sorted(weight_map.keys()):
            if 'lm_head' in key.lower() or 'embed' in key.lower():
                print(f"    {key} -> {weight_map[key]}")

    elif os.path.exists(single_path):
        with safe_open(single_path, framework="pt") as f:
            keys = list(f.keys())
        lm_head_in_checkpoint = "lm_head.weight" in keys
        print(f"  lm_head.weight in single safetensors: {lm_head_in_checkpoint}")
    else:
        # Check for pytorch .bin format
        bin_index = os.path.join(args.model, "pytorch_model.bin.index.json")
        if os.path.exists(bin_index):
            with open(bin_index) as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})
            lm_head_in_checkpoint = "lm_head.weight" in weight_map
            print(f"  lm_head.weight in bin index: {lm_head_in_checkpoint}")
        else:
            print(f"  Could not find checkpoint index file")

    # 2. Load model and check loading warnings
    print(f"\n[2] Loading model with detailed missing/unexpected key info...")
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    print(f"  tie_word_embeddings = {config.tie_word_embeddings}")

    # Force tie_word_embeddings=True if checkpoint doesn't have lm_head.weight
    if not lm_head_in_checkpoint:
        print(f"\n  *** lm_head.weight is NOT in checkpoint ***")
        print(f"  *** The checkpoint relies on weight tying! ***")
        print(f"  *** But config has tie_word_embeddings={config.tie_word_embeddings} ***")
        if not config.tie_word_embeddings:
            print(f"  *** MISMATCH: config says no tying, but checkpoint needs it ***")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
    )
    model.eval()

    # Check lm_head state
    emb_w = model.backbone.embeddings.weight
    lm_w = model.lm_head.weight
    cos = torch.nn.functional.cosine_similarity(
        emb_w.float().flatten().unsqueeze(0),
        lm_w.float().flatten().unsqueeze(0),
    ).item()
    print(f"\n  lm_head vs embeddings cosine sim: {cos:.6f}")
    print(f"  lm_head weight stats: mean={lm_w.float().mean():.6f}, std={lm_w.float().std():.6f}")
    print(f"  embeddings weight stats: mean={emb_w.float().mean():.6f}, std={emb_w.float().std():.6f}")

    # 3. Test: manually tie weights and re-run
    print(f"\n[3] Manually tying lm_head.weight = embeddings.weight and re-testing...")
    model.lm_head.weight = model.backbone.embeddings.weight
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    test_text = "The capital of France is"
    inputs = tokenizer(test_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(input_ids=inputs["input_ids"], use_cache=False)

    logits = outputs.logits
    last_logits = logits[0, -1, :]
    probs = torch.softmax(last_logits.float(), dim=-1)
    top5_probs, top5_ids = probs.topk(5)
    print(f"\n  Top-5 predictions (after manual tying):")
    for prob, tok_id in zip(top5_probs, top5_ids):
        token_str = tokenizer.decode([tok_id.item()])
        print(f"    '{token_str}' (id={tok_id.item()}): {prob.item():.4f}")

    # Quick PPL check
    test_text2 = "The quick brown fox jumps over the lazy dog. In a world where technology advances rapidly, we must consider the implications of artificial intelligence on society."
    inputs2 = tokenizer(test_text2, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs2 = model(input_ids=inputs2["input_ids"], use_cache=False)
    shift_logits = outputs2.logits[:, :-1, :].contiguous().float()
    shift_labels = inputs2["input_ids"][:, 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )
    ppl = torch.exp(loss).item()
    print(f"\n  Perplexity after manual tying: {ppl:.2f}")
    if ppl < 100:
        print(f"  *** FIXED! Manual weight tying resolved the issue ***")
        print(f"  *** Root cause: checkpoint doesn't have lm_head.weight, ***")
        print(f"  *** needs tie_word_embeddings=True or _tied_weights_keys fix ***")
    else:
        print(f"  Manual tying did NOT fix it — issue is elsewhere")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
