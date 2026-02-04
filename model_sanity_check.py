import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


model_path = "/gpfs/goon/models/granite-4-lite/run-20250607-phase3-annealing-with-fim-revised-mix/hf"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

prompt = "The capital city of USA is"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=False,   # greedy
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

print(tokenizer.decode(out[0], skip_special_tokens=True))
