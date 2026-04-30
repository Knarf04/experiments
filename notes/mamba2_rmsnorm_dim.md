# Mamba-2 in-block RMSNorm: normalization dimension

Source: [`mamba/mamba_ssm/modules/mamba2.py`](../../mamba/mamba_ssm/modules/mamba2.py) and the gated-norm kernel in [`mamba/mamba_ssm/ops/triton/layernorm_gated.py`](../../mamba/mamba_ssm/ops/triton/layernorm_gated.py).

## Construction

[`mamba2.py:147-150`](../../mamba/mamba_ssm/modules/mamba2.py#L147-L150):

```python
self.norm = RMSNormGated(
    self.d_ssm, eps=1e-5,
    norm_before_gate=self.norm_before_gate,
    group_size=self.d_ssm // ngroups,
    **factory_kwargs,
)
```

- `hidden_size = d_ssm = nheads * headdim` → elementwise affine `weight` is per-channel over the full SSM feature axis.
- `group_size = d_ssm // ngroups` → statistics are computed within groups of this many channels.
- Subtle: this uses the constructor arg `ngroups`, **not** `self.ngroups = ngroups // world_size`. With TP (`world_size > 1`) those diverge; on a single GPU they're the same.

## Input shape at the norm call

[`mamba2.py:410-412`](../../mamba/mamba_ssm/modules/mamba2.py#L410-L412):

```python
y = rearrange(y, "b l h p -> b l (h p)")   # (B, L, nheads*headdim)
if self.rmsnorm:
    y = self.norm(y, z)
```

Heads are flattened back into one feature axis before the norm — i.e. the norm sees `(B, L, d_ssm)`, not `(B, L, H, head_dim)`.

## What "group" means in the kernel

Reference impl mirroring the Triton kernel — [`layernorm_gated.py:28-34`](../../mamba/mamba_ssm/ops/triton/layernorm_gated.py#L28-L34):

```python
x_group = rearrange(x, "... (g d) -> ... g d", d=group_size)
rstd    = 1 / torch.sqrt((x_group.square()).mean(dim=-1, keepdim=True) + eps)
out     = rearrange(x_group * rstd, "... g d -> ... (g d)") * weight
```

Mean-square is taken over the last `d = group_size` elements; there are `g = ngroups` groups along the feature dim.

## Normalization granularity vs. `ngroups`

With `d_ssm = nheads * headdim` and `group_size = d_ssm // ngroups`:

| `ngroups`        | `group_size`            | Effective norm                                         |
|------------------|-------------------------|--------------------------------------------------------|
| `1` (default)    | `nheads * headdim`      | **Joint** RMS over all heads (one denominator)         |
| `nheads`         | `headdim`               | **Per-head** RMS (each head normalized over `head_dim`)|
| general `g`      | `(nheads/g) * headdim`  | RMS shared across a contiguous block of `nheads/g` heads |

The grouping matches Mamba-2's GVA grouping for `B`/`C`.

## Gating

`RMSNormGated` is gated with `z`. With `norm_before_gate=False` (the Mamba-2 default), the order in the reference impl ([`layernorm_gated.py:18-39`](../../mamba/mamba_ssm/ops/triton/layernorm_gated.py#L18-L39)) is:

1. `x = x * F.silu(z)` (gate first, since `norm_before_gate` is False — applied to `x` *before* computing rstd)
2. compute group RMS, scale, apply per-channel `weight`

With `norm_before_gate=True`: norm first, then `out *= F.silu(z)`.

## Takeaway

The in-block RMSNorm normalizes over **`d_ssm // ngroups` contiguous channels**, not per-head by default. It's per-head only when `ngroups == nheads`, joint across all heads when `ngroups == 1` (the default), and otherwise spans `nheads/ngroups` heads at a time. The per-channel `weight` (size `d_ssm`) is applied after the in-group rstd.

When inspecting per-head magnitudes (Bamba / Nemotron-H / Zamba2 plots), check the configured `ngroups` for that model: if `ngroups < nheads`, heads in the same group share an RMS denominator, so their post-norm magnitudes are not independent.

## Per-model `ngroups`

### Nemotron-H

From [`transformers/src/transformers/models/nemotron_h/configuration_nemotron_h.py`](../../transformers/src/transformers/models/nemotron_h/configuration_nemotron_h.py) defaults:

- `mamba_num_heads = 128`
- `mamba_head_dim = 64`
- `mamba_n_groups = 8`

So `d_ssm = 128 * 64 = 8192` and `group_size = 8192 / 8 = 1024 = 16 * 64` → **each group spans 16 heads × `head_dim`**.

Implication for the plotted diagnostics:

- 16 heads share an RMS denominator. Their relative magnitudes within a group are normalized (cross-head pattern *within* a block of 16 is suppressed).
- The relative scale *across* the 8 groups is preserved.
- Per-head magnitude diagnostics within a group are not independent post-norm; the cross-head/cross-group cos-sim direction diagnostic is the relevant one for within-group structure, while raw magnitude is meaningful between groups.
