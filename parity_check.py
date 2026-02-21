"""
Parity check between two HuggingFace Bamba models.

Usage:
    python parity_check.py --model-a /path/to/model_a --model-b /path/to/model_b
    python parity_check.py --model-a /path/to/model_a --model-b /path/to/model_b --prompt "Hello world"
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


DEFAULT_PROMPTS = [
    "The capital city of France is",
    "In a world where AI is everywhere,",
    "The quick brown fox",
]


def load_model(path, device):
    print(f"  Loading model from {path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()
    model.to(device)
    return model


def get_logits(model, input_ids):
    with torch.no_grad():
        out = model(input_ids=input_ids)
    return out.logits  # (batch, seq_len, vocab)


def check_parity(logits_a, logits_b, label=""):
    prefix = f"[{label}] " if label else ""
    diff = (logits_a.float() - logits_b.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # Top-1 token agreement at each position
    tokens_a = logits_a.argmax(dim=-1)
    tokens_b = logits_b.argmax(dim=-1)
    agree = (tokens_a == tokens_b).float().mean().item()

    print(f"{prefix}max |logit diff|  : {max_diff:.6f}")
    print(f"{prefix}mean |logit diff| : {mean_diff:.6f}")
    print(f"{prefix}top-1 token agree : {agree * 100:.1f}%")
    return max_diff, mean_diff, agree


def run_greedy(model, tokenizer, input_ids, max_new_tokens=20):
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", required=True, help="Path to first HF model")
    parser.add_argument("--model-b", required=True, help="Path to second HF model")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to use. If not set, runs a small default set.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS

    print(f"Device: {device}\n")

    print("=== Loading models ===")
    model_a = load_model(args.model_a, device)
    model_b = load_model(args.model_b, device)

    # Use tokenizer from model_a; both models should share the same vocabulary
    tokenizer = AutoTokenizer.from_pretrained(args.model_a)

    print("\n=== Config diff ===")
    rp_a = getattr(model_a.config, "rope_parameters", None)
    rp_b = getattr(model_b.config, "rope_parameters", None)
    print(f"  model-a rope_parameters : {rp_a}")
    print(f"  model-b rope_parameters : {rp_b}")

    print("\n=== Logit parity ===")
    all_max_diffs = []
    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        logits_a = get_logits(model_a, input_ids)
        logits_b = get_logits(model_b, input_ids)
        print(f'\nPrompt: "{prompt}"')
        max_diff, mean_diff, agree = check_parity(logits_a, logits_b)
        all_max_diffs.append(max_diff)

    print(f"\nOverall max |logit diff| across all prompts: {max(all_max_diffs):.6f}")

    print("\n=== Greedy generation ===")
    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        out_a = run_greedy(model_a, tokenizer, input_ids, args.max_new_tokens)
        out_b = run_greedy(model_b, tokenizer, input_ids, args.max_new_tokens)
        match = "MATCH" if out_a == out_b else "DIFFER"
        print(f'\nPrompt : "{prompt}"')
        print(f"  model-a : {out_a}")
        print(f"  model-b : {out_b}")
        print(f"  => {match}")


if __name__ == "__main__":
    main()
