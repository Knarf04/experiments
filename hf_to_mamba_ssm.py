"""Convert HuggingFace checkpoints (NemotronH / GraniteMoeHybrid) to mamba_ssm's MambaLMHeadModel.

Usage:
    python hf_to_mamba_ssm.py --model_dir /path/to/hf_model --output_dir /path/to/output
    python hf_to_mamba_ssm.py --model_dir /path/to/hf_model --output_dir /path/to/output --check_parity
    python hf_to_mamba_ssm.py --model_dir /path/to/hf_model --output_dir /path/to/output --model_type nemotron_h

Supported model types:
    - nemotron_h: NemotronHForCausalLM (hybrid Mamba2/Attention/MLP)
    - granitemoehybrid: GraniteMoeHybridForCausalLM with MoE turned off
"""

import argparse

import torch
import torch.nn.functional as F

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig


# ─── Utilities ───────────────────────────────────────────────────────────────


def check_parity(hf_model, mamba_model, tokenizer, device="cuda", dtype=torch.bfloat16):
    """Compare outputs of HF model and mamba_ssm model on the same input.

    Note: requires GPU — mamba_ssm's Mamba2 uses CUDA kernels for forward pass.
    """
    hf_model.to(device=device, dtype=dtype).eval()
    mamba_model.to(device=device, dtype=dtype).eval()

    input_ids = tokenizer(
        "The quick brown fox jumps over the lazy dog", return_tensors="pt"
    )["input_ids"].to(device)

    with torch.no_grad():
        hf_logits = hf_model(input_ids).logits.float()
        mamba_logits = mamba_model(input_ids).logits.float()

    # Trim to common vocab size (mamba_ssm may have padded vocab)
    min_vocab = min(hf_logits.shape[-1], mamba_logits.shape[-1])
    hf_logits = hf_logits[..., :min_vocab]
    mamba_logits = mamba_logits[..., :min_vocab]

    max_diff = (hf_logits - mamba_logits).abs().max().item()
    mean_diff = (hf_logits - mamba_logits).abs().mean().item()
    cos_sim = F.cosine_similarity(
        hf_logits.flatten().unsqueeze(0),
        mamba_logits.flatten().unsqueeze(0),
    ).item()

    print(f"  Max diff:  {max_diff:.2e}")
    print(f"  Mean diff: {mean_diff:.2e}")
    print(f"  Cos sim:   {cos_sim:.6f}")
    return max_diff, mean_diff


def combine_qkv(q_weight, k_weight, v_weight, q_bias=None, k_bias=None, v_bias=None):
    """Combine separate Q, K, V projection weights into mamba_ssm MHA's in_proj format.

    mamba_ssm MHA splits in_proj as:
        q, kv = split([num_heads*head_dim, 2*num_kv_heads*head_dim])
        kv = rearrange("... (two hkv d) -> ... two hkv d", two=2)

    The rearrange means KV is laid out as [K_all_heads, V_all_heads].
    So in_proj = cat([Q, K, V], dim=0).
    """
    weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
    bias = None
    if q_bias is not None:
        bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
    return weight, bias


# ─── NemotronH ───────────────────────────────────────────────────────────────


def parse_nemotronh_pattern(pattern):
    """Parse NemotronH hybrid_override_pattern into mamba_ssm blocks.

    Pattern chars: M=Mamba2, *=Attention, -=MLP.
    Each mixer (M or *) optionally pairs with the following MLP (-).

    Returns:
        list of dicts: {"type": "mamba"|"attention", "mixer_idx": int, "mlp_idx": int|None}
    """
    blocks = []
    i = 0
    while i < len(pattern):
        char = pattern[i]
        if char in ("M", "*"):
            mixer_idx = i
            mlp_idx = None
            if i + 1 < len(pattern) and pattern[i + 1] == "-":
                mlp_idx = i + 1
                i += 2
            else:
                i += 1
            block_type = "mamba" if char == "M" else "attention"
            blocks.append(
                {"type": block_type, "mixer_idx": mixer_idx, "mlp_idx": mlp_idx}
            )
        elif char == "-":
            raise ValueError(
                f"Unexpected standalone MLP at position {i} in pattern: {pattern}"
            )
        else:
            raise ValueError(
                f"Unknown character '{char}' at position {i} in pattern: {pattern}"
            )
    return blocks


def nemotronh_config_to_mamba(hf_config):
    """Convert NemotronH HF config to MambaConfig + block mapping."""
    blocks = parse_nemotronh_pattern(hf_config.hybrid_override_pattern)
    attn_layer_idx = [i for i, b in enumerate(blocks) if b["type"] == "attention"]

    d_intermediate_per_layer = [
        hf_config.intermediate_size if b["mlp_idx"] is not None else 0
        for b in blocks
    ]

    mamba_config = MambaConfig(
        d_model=hf_config.hidden_size,
        n_layer=len(blocks),
        d_intermediate=d_intermediate_per_layer,
        vocab_size=hf_config.vocab_size,
        ssm_cfg=dict(
            layer="Mamba2",
            d_state=hf_config.ssm_state_size,
            d_conv=hf_config.conv_kernel,
            expand=hf_config.mamba_expand,
            headdim=hf_config.mamba_head_dim,
            ngroups=hf_config.n_groups,
            rmsnorm=True,
            norm_before_gate=False,
            bias=hf_config.use_bias,
            conv_bias=hf_config.use_conv_bias,
            chunk_size=hf_config.chunk_size,
        ),
        attn_layer_idx=attn_layer_idx,
        attn_cfg=dict(
            num_heads=hf_config.num_attention_heads,
            num_heads_kv=hf_config.num_key_value_heads,
            head_dim=hf_config.attention_head_dim,
            out_proj_bias=hf_config.attention_bias,
            qkv_proj_bias=hf_config.attention_bias,
            causal=True,
            rotary_emb_dim=0,  # NemotronH doesn't use RoPE
        ),
        rms_norm=True,
        norm_epsilon=getattr(hf_config, "layer_norm_epsilon", 1e-5),
        residual_in_fp32=getattr(hf_config, "residual_in_fp32", False),
        fused_add_norm=True,
        pad_vocab_size_multiple=1,
        tie_embeddings=hf_config.tie_word_embeddings,
        mlp_cfg=dict(type="simple", activation="relu2"),
    )
    return mamba_config, blocks


def convert_nemotronh_weights(hf_state_dict, blocks, hf_config):
    """Convert NemotronH state dict to mamba_ssm format."""
    new_sd = {}

    # Embedding
    new_sd["backbone.embedding.weight"] = hf_state_dict[
        "backbone.embeddings.weight"
    ].clone()

    # Final norm
    new_sd["backbone.norm_f.weight"] = hf_state_dict["backbone.norm_f.weight"].clone()

    # LM head
    if "lm_head.weight" in hf_state_dict:
        new_sd["lm_head.weight"] = hf_state_dict["lm_head.weight"].clone()

    for block_idx, block in enumerate(blocks):
        mixer_idx = block["mixer_idx"]
        mlp_idx = block["mlp_idx"]
        hf_prefix = f"backbone.layers.{mixer_idx}"
        mamba_prefix = f"backbone.layers.{block_idx}"

        # Norm (from mixer layer)
        new_sd[f"{mamba_prefix}.norm.weight"] = hf_state_dict[
            f"{hf_prefix}.norm.weight"
        ].clone()

        if block["type"] == "mamba":
            # Mamba2 mixer — direct copy
            for param in [
                "in_proj.weight",
                "conv1d.weight",
                "conv1d.bias",
                "dt_bias",
                "A_log",
                "D",
                "norm.weight",
                "out_proj.weight",
            ]:
                hf_key = f"{hf_prefix}.mixer.{param}"
                if hf_key in hf_state_dict:
                    new_sd[f"{mamba_prefix}.mixer.{param}"] = hf_state_dict[
                        hf_key
                    ].clone()

            # Optional biases (in_proj, out_proj)
            for param in ["in_proj.bias", "out_proj.bias"]:
                hf_key = f"{hf_prefix}.mixer.{param}"
                if hf_key in hf_state_dict:
                    new_sd[f"{mamba_prefix}.mixer.{param}"] = hf_state_dict[
                        hf_key
                    ].clone()

        elif block["type"] == "attention":
            # Attention — combine Q/K/V into in_proj
            q_w = hf_state_dict[f"{hf_prefix}.mixer.q_proj.weight"]
            k_w = hf_state_dict[f"{hf_prefix}.mixer.k_proj.weight"]
            v_w = hf_state_dict[f"{hf_prefix}.mixer.v_proj.weight"]

            q_b = hf_state_dict.get(f"{hf_prefix}.mixer.q_proj.bias")
            k_b = hf_state_dict.get(f"{hf_prefix}.mixer.k_proj.bias")
            v_b = hf_state_dict.get(f"{hf_prefix}.mixer.v_proj.bias")

            in_proj_w, in_proj_b = combine_qkv(q_w, k_w, v_w, q_b, k_b, v_b)
            new_sd[f"{mamba_prefix}.mixer.in_proj.weight"] = in_proj_w
            if in_proj_b is not None:
                new_sd[f"{mamba_prefix}.mixer.in_proj.bias"] = in_proj_b

            # Output projection: o_proj → out_proj
            new_sd[f"{mamba_prefix}.mixer.out_proj.weight"] = hf_state_dict[
                f"{hf_prefix}.mixer.o_proj.weight"
            ].clone()
            o_bias_key = f"{hf_prefix}.mixer.o_proj.bias"
            if o_bias_key in hf_state_dict:
                new_sd[f"{mamba_prefix}.mixer.out_proj.bias"] = hf_state_dict[
                    o_bias_key
                ].clone()

        # MLP (if this block has one)
        if mlp_idx is not None:
            mlp_hf_prefix = f"backbone.layers.{mlp_idx}"

            # MLP norm
            new_sd[f"{mamba_prefix}.norm2.weight"] = hf_state_dict[
                f"{mlp_hf_prefix}.norm.weight"
            ].clone()

            # MLP weights: NemotronH uses up_proj/down_proj via .mixer
            new_sd[f"{mamba_prefix}.mlp.fc1.weight"] = hf_state_dict[
                f"{mlp_hf_prefix}.mixer.up_proj.weight"
            ].clone()
            new_sd[f"{mamba_prefix}.mlp.fc2.weight"] = hf_state_dict[
                f"{mlp_hf_prefix}.mixer.down_proj.weight"
            ].clone()

            # Optional biases
            for fc, proj in [("fc1", "up_proj"), ("fc2", "down_proj")]:
                hf_key = f"{mlp_hf_prefix}.mixer.{proj}.bias"
                if hf_key in hf_state_dict:
                    new_sd[f"{mamba_prefix}.mlp.{fc}.bias"] = hf_state_dict[
                        hf_key
                    ].clone()

    return new_sd


# ─── GraniteMoeHybrid ────────────────────────────────────────────────────────


def granite_config_to_mamba(hf_config):
    """Convert GraniteMoeHybrid HF config to MambaConfig."""
    layer_types = hf_config.layers_block_type  # list of "mamba" / "attention"
    attn_layer_idx = [i for i, t in enumerate(layer_types) if t == "attention"]

    # Compute headdim
    mamba_intermediate = hf_config.mamba_expand * hf_config.hidden_size
    if hf_config.mamba_d_head == "auto":
        headdim = mamba_intermediate // hf_config.mamba_n_heads
    else:
        headdim = hf_config.mamba_d_head

    # Get rope_theta from config
    rope_theta = getattr(hf_config, "rope_theta", None)
    if rope_theta is None:
        rope_params = getattr(hf_config, "rope_parameters", None)
        if rope_params and isinstance(rope_params, dict):
            rope_theta = rope_params.get("rope_theta", 10000.0)
        else:
            rope_theta = 10000.0

    head_dim = hf_config.hidden_size // hf_config.num_attention_heads

    # Always untie embeddings so we can fold different scalings
    mamba_config = MambaConfig(
        d_model=hf_config.hidden_size,
        n_layer=hf_config.num_hidden_layers,
        d_intermediate=hf_config.shared_intermediate_size,
        vocab_size=hf_config.vocab_size,
        ssm_cfg=dict(
            layer="Mamba2",
            d_state=hf_config.mamba_d_state,
            d_conv=hf_config.mamba_d_conv,
            expand=hf_config.mamba_expand,
            headdim=headdim,
            ngroups=hf_config.mamba_n_groups,
            rmsnorm=True,
            norm_before_gate=False,
            bias=hf_config.mamba_proj_bias,
            conv_bias=hf_config.mamba_conv_bias,
            chunk_size=hf_config.mamba_chunk_size,
        ),
        attn_layer_idx=attn_layer_idx,
        attn_cfg=dict(
            num_heads=hf_config.num_attention_heads,
            num_heads_kv=hf_config.num_key_value_heads,
            head_dim=head_dim,
            out_proj_bias=hf_config.attention_bias,
            qkv_proj_bias=hf_config.attention_bias,
            causal=True,
            softmax_scale=hf_config.attention_multiplier,
            rotary_emb_dim=head_dim,
            rotary_emb_base=rope_theta,
        ),
        rms_norm=True,
        norm_epsilon=getattr(hf_config, "rms_norm_eps", 1e-6),
        residual_in_fp32=True,
        fused_add_norm=True,
        pad_vocab_size_multiple=1,
        tie_embeddings=False,
        mlp_cfg=dict(type="gated", multiple_of=1),
    )
    return mamba_config


def convert_granite_weights(hf_state_dict, hf_config):
    """Convert GraniteMoeHybrid state dict to mamba_ssm format.

    Folds scaling factors into weights:
    - embedding_multiplier → embedding weights
    - residual_multiplier → out_proj and fc2 weights
    - logits_scaling → lm_head weights (divided)
    """
    new_sd = {}
    layer_types = hf_config.layers_block_type

    emb_mult = getattr(hf_config, "embedding_multiplier", 1.0)
    res_mult = getattr(hf_config, "residual_multiplier", 1.0)
    logits_scale = getattr(hf_config, "logits_scaling", 1.0)

    # Embedding (scaled by embedding_multiplier)
    emb_weight = hf_state_dict["model.embed_tokens.weight"]
    new_sd["backbone.embedding.weight"] = emb_weight * emb_mult

    # Final norm
    new_sd["backbone.norm_f.weight"] = hf_state_dict["model.norm.weight"].clone()

    # LM head (divide by logits_scaling).
    # Use raw emb_weight (not scaled by emb_mult) — lm_head only gets logits scaling.
    if "lm_head.weight" in hf_state_dict:
        new_sd["lm_head.weight"] = hf_state_dict["lm_head.weight"] / logits_scale
    else:
        # Tied weights in HF — derive from raw embedding weight
        new_sd["lm_head.weight"] = emb_weight / logits_scale

    for i, layer_type in enumerate(layer_types):
        hf_prefix = f"model.layers.{i}"
        mamba_prefix = f"backbone.layers.{i}"

        # Input layernorm → norm
        new_sd[f"{mamba_prefix}.norm.weight"] = hf_state_dict[
            f"{hf_prefix}.input_layernorm.weight"
        ].clone()

        if layer_type == "mamba":
            # Mamba2 mixer — direct copy, scale out_proj by residual_multiplier
            for param in [
                "in_proj.weight",
                "conv1d.weight",
                "conv1d.bias",
                "dt_bias",
                "A_log",
                "D",
                "norm.weight",
            ]:
                hf_key = f"{hf_prefix}.mamba.{param}"
                if hf_key in hf_state_dict:
                    new_sd[f"{mamba_prefix}.mixer.{param}"] = hf_state_dict[
                        hf_key
                    ].clone()

            # out_proj scaled by residual_multiplier
            out_key = f"{hf_prefix}.mamba.out_proj.weight"
            if out_key in hf_state_dict:
                new_sd[f"{mamba_prefix}.mixer.out_proj.weight"] = (
                    hf_state_dict[out_key] * res_mult
                )

            # Optional biases
            for param in ["in_proj.bias", "out_proj.bias"]:
                hf_key = f"{hf_prefix}.mamba.{param}"
                if hf_key in hf_state_dict:
                    w = hf_state_dict[hf_key].clone()
                    if param == "out_proj.bias":
                        w = w * res_mult
                    new_sd[f"{mamba_prefix}.mixer.{param}"] = w

        elif layer_type == "attention":
            # Attention — combine Q/K/V, scale o_proj by residual_multiplier
            q_w = hf_state_dict[f"{hf_prefix}.self_attn.q_proj.weight"]
            k_w = hf_state_dict[f"{hf_prefix}.self_attn.k_proj.weight"]
            v_w = hf_state_dict[f"{hf_prefix}.self_attn.v_proj.weight"]

            q_b = hf_state_dict.get(f"{hf_prefix}.self_attn.q_proj.bias")
            k_b = hf_state_dict.get(f"{hf_prefix}.self_attn.k_proj.bias")
            v_b = hf_state_dict.get(f"{hf_prefix}.self_attn.v_proj.bias")

            in_proj_w, in_proj_b = combine_qkv(q_w, k_w, v_w, q_b, k_b, v_b)
            new_sd[f"{mamba_prefix}.mixer.in_proj.weight"] = in_proj_w
            if in_proj_b is not None:
                new_sd[f"{mamba_prefix}.mixer.in_proj.bias"] = in_proj_b

            # o_proj → out_proj, scaled by residual_multiplier
            new_sd[f"{mamba_prefix}.mixer.out_proj.weight"] = (
                hf_state_dict[f"{hf_prefix}.self_attn.o_proj.weight"] * res_mult
            )
            o_bias_key = f"{hf_prefix}.self_attn.o_proj.bias"
            if o_bias_key in hf_state_dict:
                new_sd[f"{mamba_prefix}.mixer.out_proj.bias"] = (
                    hf_state_dict[o_bias_key] * res_mult
                )

        # MLP (every GraniteMoeHybrid layer has a shared_mlp)
        new_sd[f"{mamba_prefix}.norm2.weight"] = hf_state_dict[
            f"{hf_prefix}.post_attention_layernorm.weight"
        ].clone()

        # shared_mlp.input_linear → mlp.fc1  (gated, 2x intermediate)
        new_sd[f"{mamba_prefix}.mlp.fc1.weight"] = hf_state_dict[
            f"{hf_prefix}.shared_mlp.input_linear.weight"
        ].clone()

        # shared_mlp.output_linear → mlp.fc2, scaled by residual_multiplier
        new_sd[f"{mamba_prefix}.mlp.fc2.weight"] = (
            hf_state_dict[f"{hf_prefix}.shared_mlp.output_linear.weight"] * res_mult
        )

    return new_sd


# ─── Main Conversion Functions ───────────────────────────────────────────────


def convert_nemotronh(model_dir, output_dir=None, check=False, device="cpu", dtype=torch.float32):
    """Full NemotronH → MambaLMHeadModel conversion."""
    print(f"Loading NemotronH config from {model_dir}...")
    hf_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)

    print("Building MambaConfig...")
    mamba_config, blocks = nemotronh_config_to_mamba(hf_config)
    print(f"  {len(blocks)} blocks, "
          f"{sum(1 for b in blocks if b['type'] == 'mamba')} mamba, "
          f"{sum(1 for b in blocks if b['type'] == 'attention')} attention, "
          f"{sum(1 for b in blocks if b['mlp_idx'] is not None)} with MLP")

    print(f"Loading HF model from {model_dir}...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=dtype, device_map=device, trust_remote_code=True
    )
    hf_sd = hf_model.state_dict()

    print("Converting weights...")
    mamba_sd = convert_nemotronh_weights(hf_sd, blocks, hf_config)

    print("Building MambaLMHeadModel...")
    mamba_model = MambaLMHeadModel(mamba_config, device=device, dtype=dtype)

    result = mamba_model.load_state_dict(mamba_sd, strict=False)
    _report_load_result(result, mamba_config)

    if check:
        print("\nRunning parity check...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        check_parity(hf_model, mamba_model, tokenizer, device=device, dtype=dtype)

    if output_dir:
        _save_model(mamba_model, model_dir, output_dir)

    return mamba_model


def convert_granitemoehybrid(model_dir, output_dir=None, check=False, device="cpu", dtype=torch.float32):
    """Full GraniteMoeHybrid → MambaLMHeadModel conversion."""
    print(f"Loading GraniteMoeHybrid config from {model_dir}...")
    hf_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)

    layer_types = hf_config.layers_block_type
    print(f"  {len(layer_types)} layers, "
          f"{sum(1 for t in layer_types if t == 'mamba')} mamba, "
          f"{sum(1 for t in layer_types if t == 'attention')} attention")

    emb_mult = getattr(hf_config, "embedding_multiplier", 1.0)
    res_mult = getattr(hf_config, "residual_multiplier", 1.0)
    logits_scale = getattr(hf_config, "logits_scaling", 1.0)
    if emb_mult != 1.0 or res_mult != 1.0 or logits_scale != 1.0:
        print(f"  Folding scaling factors: emb={emb_mult}, res={res_mult}, logits={logits_scale}")

    print("Building MambaConfig...")
    mamba_config = granite_config_to_mamba(hf_config)

    print(f"Loading HF model from {model_dir}...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=dtype, device_map=device, trust_remote_code=True
    )
    hf_sd = hf_model.state_dict()

    print("Converting weights...")
    mamba_sd = convert_granite_weights(hf_sd, hf_config)

    print("Building MambaLMHeadModel...")
    mamba_model = MambaLMHeadModel(mamba_config, device=device, dtype=dtype)

    result = mamba_model.load_state_dict(mamba_sd, strict=False)
    _report_load_result(result, mamba_config)

    if check:
        print("\nRunning parity check...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        check_parity(hf_model, mamba_model, tokenizer, device=device, dtype=dtype)

    if output_dir:
        _save_model(mamba_model, model_dir, output_dir)

    return mamba_model


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _report_load_result(result, mamba_config):
    """Report missing/unexpected keys from load_state_dict."""
    if result.unexpected_keys:
        print(f"  WARNING: unexpected keys: {result.unexpected_keys}")
    if result.missing_keys:
        missing = [
            k for k in result.missing_keys
            if not (k == "lm_head.weight" and mamba_config.tie_embeddings)
        ]
        if missing:
            print(f"  WARNING: missing keys: {missing}")
        else:
            print("  All weights loaded successfully (lm_head tied to embedding).")
    else:
        print("  All weights loaded successfully.")


def _save_model(mamba_model, model_dir, output_dir):
    """Save converted model and tokenizer."""
    print(f"\nSaving to {output_dir}...")
    mamba_model.save_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)
    print("Done!")


# ─── Auto-detection ──────────────────────────────────────────────────────────


MODEL_TYPE_MAP = {
    "nemotron_h": convert_nemotronh,
    "granitemoehybrid": convert_granitemoehybrid,
}

HF_MODEL_TYPE_MAP = {
    "nemotron_h": "nemotron_h",
    "granitemoehybrid": "granitemoehybrid",
}


def detect_model_type(model_dir):
    """Auto-detect model type from HF config."""
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model_type = getattr(config, "model_type", None)
    if model_type in HF_MODEL_TYPE_MAP:
        return HF_MODEL_TYPE_MAP[model_type]
    raise ValueError(
        f"Unknown model type: {model_type}. "
        f"Supported: {list(HF_MODEL_TYPE_MAP.keys())}"
    )


# ─── CLI ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace checkpoints to mamba_ssm MambaLMHeadModel"
    )
    parser.add_argument(
        "--model_dir", required=True,
        help="Path to HuggingFace model directory",
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Path to save converted model (omit for dry-run)",
    )
    parser.add_argument(
        "--model_type",
        choices=["nemotron_h", "granitemoehybrid", "auto"],
        default="auto",
        help="Model type (default: auto-detect from config)",
    )
    parser.add_argument(
        "--check_parity", action="store_true",
        help="Run parity check after conversion (requires GPU)",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device for model loading (default: cpu)",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float32", "bfloat16", "float16"],
        help="Data type (default: bfloat16)",
    )

    args = parser.parse_args()

    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    dtype = dtype_map[args.dtype]

    if args.model_type == "auto":
        model_type = detect_model_type(args.model_dir)
        print(f"Auto-detected model type: {model_type}")
    else:
        model_type = args.model_type

    converter = MODEL_TYPE_MAP[model_type]
    converter(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        check=args.check_parity,
        device=args.device,
        dtype=dtype,
    )


if __name__ == "__main__":
    main()
