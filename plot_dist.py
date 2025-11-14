import json
import numpy as np
import matplotlib.pyplot as plt

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
            if len(records) == 1000:
                break
    return records

# load pre-computed ERF
with open("/gpfs/hshen/upload/mamba2_erf.json", "r") as f:
    erf_json = json.load(f)

erf = {}
all_erf = []
for sl in erf_json.keys():
    seq_len = int(sl)
    erf[seq_len] = {}
    for li in erf_json[sl].keys():
        layer_idx = int(li)
        erf[seq_len][layer_idx] = np.array(erf_json[sl][li])
        all_erf += erf_json[sl][li]
all_erf = np.array(all_erf)
cutoff = np.quantile(all_erf, 0.8)
print("Top 20% ERF cutoff:", cutoff)

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

# print("dt shape:", np.array(records[0]["dt"]).shape)
# print("forget shape: ", np.array(records[0]["forget"]).shape)
# (batch, seqlen, nheads)

# dt_dict = {}
forget_dict = {}
count = {}
for rec in records:
    layer_idx = rec['layer_idx']
    if layer_idx in attn_layers[model_type]:
        continue

    # dt = np.array(rec['dt']).transpose(2, 0, 1)
    # dt = dt.reshape(dt.shape[0], -1)

    forget = np.array(rec['forget'])
    if seq_len not in forget_dict.keys():
        # dt_dict[seq_len] = {}
        forget_dict[seq_len] = {}
        count[seq_len] = {}
    if layer_idx not in forget_dict[seq_len].keys():
        # dt_dict[seq_len][layer_idx] = dt
        forget_dict[seq_len][layer_idx] = forget
        count[seq_len][layer_idx] = 1
    else:
        # dt_dict[seq_len][layer_idx] = np.concatenate((dt_dict[seq_len][layer_idx], dt), axis=1)
        forget_dict[seq_len][layer_idx] += forget
        count[seq_len][layer_idx] += 1

for seq_len in forget_dict.keys():
    for layer_idx in forget_dict[seq_len].keys():
        forget_dict[seq_len][layer_idx] /= count[seq_len][layer_idx]
        forget_dict[seq_len][layer_idx] = np.mean(forget_dict[seq_len][layer_idx], axis=(0, 1))

print(forget_dict[2048][0].shape)
# print(dt_dict[2048][0].shape)

keys = forget_dict[seq_len].keys()  # d1 and d2 have the same keys

x_vals = [erf[seq_len][k] for k in keys]
y_vals = [forget_dict[seq_len][k] for k in keys]

plt.scatter(x_vals, y_vals)
plt.xlabel("ERF")
plt.ylabel("Average forget gate")
plt.savefig("/gpfs/hshen/plots/mamba2_forget_dist.png", dpi=600)
plt.show()