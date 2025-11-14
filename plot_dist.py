import json
import numpy as np

def read_jsonl(filepath: str) -> list:
    """
    Read a JSONL file and return a list of Python dicts.
    """
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if len(records) == 10:
                break
    return records

path = "/gpfs/hshen/mmd/mamba2.jsonl"

records = read_jsonl(path)
print(f"Loaded {len(records)} records via json module")

model_type = "mamba2"
seq_len = 2048
attn_layers = {
    "bamba": [9, 18, 27],
    "nemotron-h": [7, 18, 29, 40],
    "mamba2": [],
}

print("dt shape:", np.array(records[0]["dt"]).shape)
print("forget shape: ", np.array(records[0]["forget"]).shape)

dt_dict = {}
forget_dict = {}
count = {}
for rec in records:
    layer_idx = rec['layer_idx']
    if layer_idx in attn_layers[model_type]:
        continue
    dt = np.array(rec['dt'])
    forget = np.array(rec['forget'])
    if seq_len not in dt_dict.keys():
        dt_dict[seq_len] = {}
        forget_dict[seq_len] = {}
        count[seq_len] = {}
    if layer_idx not in dt_dict[seq_len].keys():
        dt_dict[seq_len][layer_idx] = dt
        forget_dict[seq_len] = forget
        count[seq_len][layer_idx] = 1
    else:
        dt_dict[seq_len][layer_idx] += dt
        forget_dict[seq_len] += forget
        count[seq_len][layer_idx] += 1
        
for seq_len in forget_dict.keys():
    for layer_idx in forget_dict[seq_len].keys():
        forget_dict[seq_len][layer_idx] /= count[seq_len][layer_idx]

print(forget_dict)
