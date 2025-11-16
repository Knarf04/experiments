import json
import numpy as np
import matplotlib.pyplot as plt

model_type = "mamba2"
seq_len = 2048   # target sequence length for plotting

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
            if len(records) == 10000:
                break
    return records

# load pre-computed ERF
with open(f"/gpfs/hshen/upload/{model_type}_erf.json", "r") as f:
    erf_json = json.load(f)

erf = {}
all_erf = []
for sl in erf_json.keys():
    sl_int = int(sl)
    erf[sl_int] = {}
    for li in erf_json[sl].keys():
        layer_idx = int(li)
        erf[sl_int][layer_idx] = np.array(erf_json[sl][li])
        all_erf += erf_json[sl][li]

all_erf = np.array(all_erf)
cutoff = np.quantile(all_erf, 0.8)
print("Top 20% ERF cutoff:", cutoff)

path = f"/gpfs/hshen/mmd/{model_type}.jsonl"

records = read_jsonl(path)
print(f"Loaded {len(records)} records via json module")


attn_layers = {
    "bamba2": [9, 18, 27],
    "nemotronh": [7, 18, 29, 40],
    "mamba2": [],
}

# (batch, seqlen, nheads)

dt_dict = {}
forget_dict = {}
count = {}

# per-head accumulators for forget-gate statistics
forget_sum_head = {}
forget_sumsq_head = {}
forget_n_head = {}

# per-head accumulators for dt (Δt) statistics
dt_sum_head = {}
dt_sumsq_head = {}
dt_n_head = {}

for rec in records:
    layer_idx = rec['layer_idx']
    if layer_idx in attn_layers[model_type]:
        continue

    dt = np.array(rec['dt'])       # (B, T, H)
    forget = np.array(rec['forget'])

    # ---- existing per-layer averaging (for means) ----
    if seq_len not in forget_dict.keys():
        dt_dict[seq_len] = {}
        forget_dict[seq_len] = {}
        count[seq_len] = {}
    if layer_idx not in forget_dict[seq_len].keys():
        dt_dict[seq_len][layer_idx] = dt
        forget_dict[seq_len][layer_idx] = forget
        count[seq_len][layer_idx] = 1
    else:
        dt_dict[seq_len][layer_idx] += dt
        forget_dict[seq_len][layer_idx] += forget
        count[seq_len][layer_idx] += 1

    # ---- per-head statistics for forget (mean + variance) ----
    if seq_len not in forget_sum_head:
        forget_sum_head[seq_len] = {}
        forget_sumsq_head[seq_len] = {}
        forget_n_head[seq_len] = {}

    if layer_idx not in forget_sum_head[seq_len]:
        B, T, H = forget.shape
        forget_sum_head[seq_len][layer_idx] = np.zeros(H, dtype=np.float64)
        forget_sumsq_head[seq_len][layer_idx] = np.zeros(H, dtype=np.float64)
        forget_n_head[seq_len][layer_idx] = 0

    f_flat = forget.reshape(-1, forget.shape[-1])  # (B*T, H)
    forget_sum_head[seq_len][layer_idx]   += f_flat.sum(axis=0)
    forget_sumsq_head[seq_len][layer_idx] += (f_flat ** 2).sum(axis=0)
    forget_n_head[seq_len][layer_idx]     += f_flat.shape[0]

    # ---- per-head statistics for dt (Δt) mean + variance ----
    if seq_len not in dt_sum_head:
        dt_sum_head[seq_len] = {}
        dt_sumsq_head[seq_len] = {}
        dt_n_head[seq_len] = {}

    if layer_idx not in dt_sum_head[seq_len]:
        B, T, H = dt.shape
        dt_sum_head[seq_len][layer_idx] = np.zeros(H, dtype=np.float64)
        dt_sumsq_head[seq_len][layer_idx] = np.zeros(H, dtype=np.float64)
        dt_n_head[seq_len][layer_idx] = 0

    d_flat = dt.reshape(-1, dt.shape[-1])  # (B*T, H)
    dt_sum_head[seq_len][layer_idx]   += d_flat.sum(axis=0)
    dt_sumsq_head[seq_len][layer_idx] += (d_flat ** 2).sum(axis=0)
    dt_n_head[seq_len][layer_idx]     += d_flat.shape[0]

# ---- compute per-head means for forget and dt, and per-head variances for forget and dt ----

forget_mean_dict = {}
forget_var_dict = {}
dt_mean_dict = {}
dt_var_dict = {}

# dt_mean: as before, from accumulated sums over full tensors
for sl in dt_dict.keys():
    dt_mean_dict[sl] = {}
    for layer_idx in dt_dict[sl].keys():
        dt_avg = dt_dict[sl][layer_idx] / count[sl][layer_idx]  # (B, T, H) avg over records
        dt_mean_dict[sl][layer_idx] = np.mean(dt_avg, axis=(0, 1))  # per-head mean Δt

# forget mean & variance: from per-head aggregates
for sl in forget_sum_head.keys():
    forget_mean_dict[sl] = {}
    forget_var_dict[sl] = {}
    for layer_idx in forget_sum_head[sl].keys():
        s  = forget_sum_head[sl][layer_idx]
        s2 = forget_sumsq_head[sl][layer_idx]
        n  = forget_n_head[sl][layer_idx]

        mean = s / n
        var  = s2 / n - mean**2  # population variance

        forget_mean_dict[sl][layer_idx] = mean   # (H,)
        forget_var_dict[sl][layer_idx]  = var    # (H,)

# dt variance (and optionally mean) from per-head aggregates
for sl in dt_sum_head.keys():
    if sl not in dt_var_dict:
        dt_var_dict[sl] = {}
    for layer_idx in dt_sum_head[sl].keys():
        s  = dt_sum_head[sl][layer_idx]
        s2 = dt_sumsq_head[sl][layer_idx]
        n  = dt_n_head[sl][layer_idx]

        mean = s / n
        var  = s2 / n - mean**2

        # overwrite dt_mean_dict with per-head mean from per-head stats (optional but consistent)
        dt_mean_dict[sl][layer_idx] = mean
        dt_var_dict[sl][layer_idx]  = var

# ---------------- PLOTTING ----------------

keys = forget_mean_dict[seq_len].keys()

# log-ERF vs average forget gate
x_vals = np.array([np.log(erf[seq_len][k]) for k in keys])          # log-ERF
y_vals = np.array([forget_mean_dict[seq_len][k] for k in keys])     # per-head mean forget

cutoff = np.log(np.quantile(all_erf, 0.8))
mask = x_vals >= cutoff   # Top 20%

plt.figure()

# Bottom 80%
plt.scatter(
    x_vals[~mask],
    y_vals[~mask],
    c="blue",
    marker='.',
    s=10,
    label="Bottom 80%",
)

# Top 20%
plt.scatter(
    x_vals[mask],
    y_vals[mask],
    c="red",
    marker='.',
    s=10,
    label="Top 20%",
)

plt.axvline(cutoff, linestyle="--", linewidth=1)

plt.xlabel("log-ERF", fontsize=14, fontweight='bold')
plt.ylabel("Average forget gate", fontsize=14, fontweight='bold')
plt.legend()

plt.savefig(f"/gpfs/hshen/plots/{model_type}_forget_dist.png", dpi=600)
plt.show()

# Histograms of per-head mean forget gate (bottom 80% vs top 20%)
plt.figure()

bins = np.linspace(y_vals.min(), y_vals.max(), 61)
plt.hist(
    y_vals[~mask],
    bins=bins,
    alpha=0.6,
    label="Bottom 80%",
)
plt.hist(
    y_vals[mask],
    bins=bins,
    alpha=0.6,
    label="Top 20%",
)

plt.xlabel("Average forget gate", fontsize=14, fontweight='bold')
plt.ylabel("Count", fontsize=14, fontweight='bold')
plt.legend()

plt.savefig(f"/gpfs/hshen/plots/{model_type}_forget_hist.png", dpi=600)
plt.show()

# Histogram of per-head variance of forget gate (bottom 80% vs top 20%)
plt.figure()

var_vals_forget = np.array([forget_var_dict[seq_len][k] for k in keys])

bins = np.linspace(var_vals_forget.min(), var_vals_forget.max(), 61)
plt.hist(
    var_vals_forget[~mask],
    bins=bins,
    alpha=0.6,
    label="Bottom 80%",
)
plt.hist(
    var_vals_forget[mask],
    bins=bins,
    alpha=0.6,
    label="Top 20%",
)

plt.xlabel("Variance of forget gate", fontsize=14, fontweight='bold')
plt.ylabel("Count", fontsize=14, fontweight='bold')
plt.legend()

plt.savefig(f"/gpfs/hshen/plots/{model_type}_forget_var_hist.png", dpi=600)
plt.show()

# log-ERF vs Δt (mean per head)
x_vals = np.array([np.log(erf[seq_len][k]) for k in keys])
y_vals = np.array([dt_mean_dict[seq_len][k] for k in keys])   # per-head mean Δt

cutoff = np.log(np.quantile(all_erf, 0.8))
mask = x_vals >= cutoff   # Top 20%

plt.figure()

# Bottom 80%
plt.scatter(
    x_vals[~mask],
    y_vals[~mask],
    c="blue",
    marker='.',
    s=10,
    label="Bottom 80%",
)

# Top 20%
plt.scatter(
    x_vals[mask],
    y_vals[mask],
    c="red",
    marker='.',
    s=10,
    label="Top 20%",
)

plt.axvline(cutoff, linestyle="--", linewidth=1)

plt.xlabel("log-ERF", fontsize=14, fontweight='bold')
plt.ylabel(r"$\Delta_t$", fontsize=14, fontweight='bold')
plt.legend()

plt.savefig(f"/gpfs/hshen/plots/{model_type}_dt_dist.png", dpi=600)
plt.show()

# Histogram of per-head mean Δt, binned between 0 and 0.6 for 30 bins
plt.figure()

bins = np.linspace(0.0, 0.6, 61)  # 30 bins between 0 and 0.6
plt.hist(
    y_vals[~mask],
    bins=bins,
    alpha=0.6,
    label="Bottom 80%",
)
plt.hist(
    y_vals[mask],
    bins=bins,
    alpha=0.6,
    label="Top 20%",
)

plt.xlabel(r"$\Delta_t$", fontsize=14, fontweight='bold')
plt.ylabel("Count", fontsize=14, fontweight='bold')
plt.legend()

plt.savefig(f"/gpfs/hshen/plots/{model_type}_dt_hist.png", dpi=600)
plt.show()

# NEW: histogram of per-head variance of Δt (bottom 80% vs top 20%)
plt.figure()

var_vals_dt = np.array([dt_var_dict[seq_len][k] for k in keys])

# bins = np.linspace(var_vals_dt.min(), var_vals_dt.max(), 61)
bins = np.linspace(0.0, 0.05, 61)
plt.hist(
    var_vals_dt[~mask],
    bins=bins,
    alpha=0.6,
    label="Bottom 80%",
)
plt.hist(
    var_vals_dt[mask],
    bins=bins,
    alpha=0.6,
    label="Top 20%",
)

plt.xlabel(r"Variance of $\Delta_t$", fontsize=14, fontweight='bold')
plt.ylabel("Count", fontsize=14, fontweight='bold')
plt.legend()

plt.savefig(f"/gpfs/hshen/plots/{model_type}_dt_var_hist.png", dpi=600)
plt.show()


# ---------------- PLOTTING ----------------

keys = forget_mean_dict[seq_len].keys()

# log-ERF vs average forget gate
x_vals = np.array([np.log(erf[seq_len][k]) for k in keys])          # log-ERF
y_vals = np.array([forget_mean_dict[seq_len][k] for k in keys])     # per-head mean forget

xy = x_vals * y_vals
xy = xy/np.max(xy) * 1.24

cutoff = np.log(np.quantile(all_erf, 0.8))
mask = x_vals >= cutoff   # Top 20%

bins = np.linspace(xy.min(), xy.max(), 61)
plt.hist(
    xy[~mask],
    bins=bins,
    alpha=0.6,
    label="Bottom 80%",
)
plt.hist(
    xy[mask],
    bins=bins,
    alpha=0.6,
    label="Top 20%",
)

plt.xlabel("Average state magnitude growth per step", fontsize=14, fontweight='bold')
plt.ylabel("Count", fontsize=14, fontweight='bold')
plt.legend()

plt.savefig(f"/gpfs/hshen/plots/{model_type}_state_hist.png", dpi=600)
plt.show()

