import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SAMPLE_PATH = "subseq_lambada.txt"


def compute_ppl(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    with open(SAMPLE_PATH, 'r', encoding='utf-8') as f:
        text = f.read()
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

    max_amount_of_windows = 10
    lengths = [10_000, 8_000, 2_000]

    ppls = []
    seq_len = input_ids.size(1)
    for window_size in lengths:
        if seq_len < window_size:
            print(f'skipping {window_size//1000}k: seq_len={seq_len//1000}k < window_size')
            continue

        stride = max((seq_len - window_size) // max_amount_of_windows, 10)
        nlls = []
        for begin_loc in range(0, seq_len - window_size, stride):
            end_loc = begin_loc + window_size
            chunk = input_ids[:, begin_loc:end_loc]
            target_ids = chunk[:, -100:].clone()

            with torch.no_grad():
                outputs = model(chunk)
                logits = outputs.logits[:, -101:-1, :]  # (1, 100, vocab)
                loss = torch.nn.CrossEntropyLoss()(
                    logits.squeeze(0), target_ids.squeeze(0)
                )
            nlls.append(loss)

        ppl = torch.exp(torch.stack(nlls).mean()).item()
        print(f'{window_size // 1000}k ppl: {ppl:.2f}')
        ppls.append(ppl)

    if ppls:
        avg_ppl = sum(ppls) / len(ppls)
        print(f'average ppl: {avg_ppl:.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name or path")
    args = parser.parse_args()
    compute_ppl(args.model)
