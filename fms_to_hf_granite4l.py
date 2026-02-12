"""Convert mamba_ssm / fms-fsdp checkpoint to GraniteMoeHybrid HuggingFace format.

Reverse of the GraniteMoeHybrid path in hf_to_mamba_ssm.py.
Un-folds scaling factors (embedding_multiplier, residual_multiplier, logits_scaling)
and reverses the GatedMLP gate/value swap.

Usage:
    python fms_to_hf_granite4l.py --src-dir /path/to/consolidated.00.pth --model-variant granite_4_lite --model-dir /path/to/output --tokenizer /path/to/tokenizer
    python fms_to_hf_granite4l.py --src-dir /path/to/consolidated.00.pth --model-variant granite_4_lite --model-dir /path/to/output --tokenizer /path/to/tokenizer --embedding-multiplier 12.0 --residual-multiplier 0.22 --logits-scaling 13.0
"""

import argparse
import json
import os

import torch

from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import load_file, save_file

from transformers import AutoTokenizer
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME

from mamba_ssm.models.config_mamba import MambaConfig
from transformers.models.granitemoehybrid import GraniteMoeHybridConfig

from fms_fsdp.utils.config_utils import get_model_config


# ─── Utilities ───────────────────────────────────────────────────────────────


def split_qkv(in_proj_weight, num_heads, num_heads_kv, head_dim, in_proj_bias=None):
    """Split mamba_ssm MHA's combined in_proj back into separate Q, K, V."""
    q_dim = num_heads * head_dim
    kv_dim = num_heads_kv * head_dim
    q_w, k_w, v_w = in_proj_weight.split([q_dim, kv_dim, kv_dim], dim=0)
    q_b, k_b, v_b = None, None, None
    if in_proj_bias is not None:
        q_b, k_b, v_b = in_proj_bias.split([q_dim, kv_dim, kv_dim], dim=0)
    return q_w, k_w, v_w, q_b, k_b, v_b


# ─── HF Config Construction ──────────────────────────────────────────────────


def convert_ssm_config_to_hf_config(mamba_config, emb_mult=1.0, res_mult=1.0,
                                     logits_scale=1.0, **kwargs):
    """Convert MambaConfig to GraniteMoeHybridConfig."""
    ssm = mamba_config.ssm_cfg
    attn = mamba_config.attn_cfg
    attn_set = set(mamba_config.attn_layer_idx or [])

    layer_types = [
        "attention" if i in attn_set else "mamba"
        for i in range(mamba_config.n_layer)
    ]

    hf_config = GraniteMoeHybridConfig(
        vocab_size=mamba_config.vocab_size,
        hidden_size=mamba_config.d_model,
        num_hidden_layers=mamba_config.n_layer,
        num_attention_heads=attn.get("num_heads", 32),
        num_key_value_heads=attn.get("num_heads_kv", 8),
        attention_bias=attn.get("qkv_proj_bias", False),
        shared_intermediate_size=mamba_config.d_intermediate,
        layer_types=layer_types,
        rms_norm_eps=mamba_config.norm_epsilon,
        tie_word_embeddings=mamba_config.tie_embeddings,
        embedding_multiplier=emb_mult,
        residual_multiplier=res_mult,
        logits_scaling=logits_scale,
        attention_multiplier=attn.get("softmax_scale", 1.0),
        mamba_n_heads=ssm.get("expand", 2) * mamba_config.d_model // ssm.get("headdim", 64),
        mamba_n_groups=ssm.get("ngroups", 1),
        mamba_d_state=ssm.get("d_state", 128),
        mamba_d_head=ssm.get("headdim", 64),
        mamba_d_conv=ssm.get("d_conv", 4),
        mamba_expand=ssm.get("expand", 2),
        mamba_chunk_size=ssm.get("chunk_size", 256),
        mamba_conv_bias=ssm.get("conv_bias", True),
        mamba_proj_bias=ssm.get("bias", False),
        **kwargs,
    )
    hf_config.architectures = ["GraniteMoeHybridForCausalLM"]
    return hf_config


# ─── Weight Conversion ───────────────────────────────────────────────────────


def convert_state_dict(mamba_sd, mamba_config, emb_mult=1.0, res_mult=1.0, logits_scale=1.0):
    """Convert mamba_ssm state dict to GraniteMoeHybrid HF format.

    Reverse of convert_granite_weights in hf_to_mamba_ssm.py.
    Un-folds scaling factors that were baked into weights.
    """
    hf_sd = {}
    attn_set = set(mamba_config.attn_layer_idx or [])
    attn_cfg = mamba_config.attn_cfg

    num_heads = attn_cfg["num_heads"]
    num_heads_kv = attn_cfg["num_heads_kv"]
    head_dim = attn_cfg.get("head_dim", mamba_config.d_model // num_heads)

    # Embedding: un-fold embedding_multiplier
    hf_sd["model.embed_tokens.weight"] = mamba_sd["backbone.embedding.weight"] / emb_mult

    # Final norm
    hf_sd["model.norm.weight"] = mamba_sd["backbone.norm_f.weight"]

    # LM head: un-fold logits_scaling
    if "lm_head.weight" in mamba_sd:
        hf_sd["lm_head.weight"] = mamba_sd["lm_head.weight"] * logits_scale

    for i in range(mamba_config.n_layer):
        layer_type = "attention" if i in attn_set else "mamba"
        mamba_prefix = f"backbone.layers.{i}"
        hf_prefix = f"model.layers.{i}"

        # Input layernorm
        hf_sd[f"{hf_prefix}.input_layernorm.weight"] = mamba_sd[f"{mamba_prefix}.norm.weight"]

        if layer_type == "mamba":
            for param in [
                "in_proj.weight", "conv1d.weight", "conv1d.bias",
                "dt_bias", "A_log", "D", "norm.weight",
                "in_proj.bias",
            ]:
                mamba_key = f"{mamba_prefix}.mixer.{param}"
                if mamba_key in mamba_sd:
                    hf_sd[f"{hf_prefix}.mamba.{param}"] = mamba_sd[mamba_key]

            # out_proj: un-fold residual_multiplier
            out_key = f"{mamba_prefix}.mixer.out_proj.weight"
            if out_key in mamba_sd:
                hf_sd[f"{hf_prefix}.mamba.out_proj.weight"] = mamba_sd[out_key] / res_mult

            out_bias_key = f"{mamba_prefix}.mixer.out_proj.bias"
            if out_bias_key in mamba_sd:
                hf_sd[f"{hf_prefix}.mamba.out_proj.bias"] = mamba_sd[out_bias_key] / res_mult

        elif layer_type == "attention":
            in_proj_w = mamba_sd[f"{mamba_prefix}.mixer.in_proj.weight"]
            in_proj_b = mamba_sd.get(f"{mamba_prefix}.mixer.in_proj.bias")

            q_w, k_w, v_w, q_b, k_b, v_b = split_qkv(
                in_proj_w, num_heads, num_heads_kv, head_dim, in_proj_b
            )

            hf_sd[f"{hf_prefix}.self_attn.q_proj.weight"] = q_w
            hf_sd[f"{hf_prefix}.self_attn.k_proj.weight"] = k_w
            hf_sd[f"{hf_prefix}.self_attn.v_proj.weight"] = v_w
            if q_b is not None:
                hf_sd[f"{hf_prefix}.self_attn.q_proj.bias"] = q_b
                hf_sd[f"{hf_prefix}.self_attn.k_proj.bias"] = k_b
                hf_sd[f"{hf_prefix}.self_attn.v_proj.bias"] = v_b

            # o_proj: un-fold residual_multiplier
            hf_sd[f"{hf_prefix}.self_attn.o_proj.weight"] = (
                mamba_sd[f"{mamba_prefix}.mixer.out_proj.weight"] / res_mult
            )
            o_bias_key = f"{mamba_prefix}.mixer.out_proj.bias"
            if o_bias_key in mamba_sd:
                hf_sd[f"{hf_prefix}.self_attn.o_proj.bias"] = mamba_sd[o_bias_key] / res_mult

        # MLP norm
        hf_sd[f"{hf_prefix}.post_attention_layernorm.weight"] = mamba_sd[
            f"{mamba_prefix}.norm2.weight"
        ]

        # MLP: un-swap gate/value halves
        # Forward: gate_w, value_w = hf.chunk(2); fc1 = cat([value_w, gate_w])
        # Reverse: value_w, gate_w = fc1.chunk(2); input_linear = cat([gate_w, value_w])
        fc1_w = mamba_sd[f"{mamba_prefix}.mlp.fc1.weight"]
        value_w, gate_w = fc1_w.chunk(2, dim=0)
        hf_sd[f"{hf_prefix}.shared_mlp.input_linear.weight"] = torch.cat(
            [gate_w, value_w], dim=0
        )

        # output_linear: un-fold residual_multiplier
        hf_sd[f"{hf_prefix}.shared_mlp.output_linear.weight"] = (
            mamba_sd[f"{mamba_prefix}.mlp.fc2.weight"] / res_mult
        )

    return hf_sd


# ─── Verification ────────────────────────────────────────────────────────────


def verify_conversion(fms_sd, hf_sd, mamba_config, emb_mult=1.0, res_mult=1.0, logits_scale=1.0):
    """Verify the conversion by checking each FMS key maps correctly to HF keys.

    Applies the same mapping logic used in convert_state_dict and checks
    that each resulting tensor matches what's in hf_sd.
    """
    all_match = True
    matched_hf_keys = set()
    attn_set = set(mamba_config.attn_layer_idx or [])
    attn_cfg = mamba_config.attn_cfg
    num_heads = attn_cfg["num_heads"]
    num_heads_kv = attn_cfg["num_heads_kv"]
    head_dim = attn_cfg.get("head_dim", mamba_config.d_model // num_heads)

    def check(hf_key, expected):
        nonlocal all_match
        matched_hf_keys.add(hf_key)
        if hf_key not in hf_sd:
            print(f"MISSING  {hf_key}")
            all_match = False
        elif expected.shape != hf_sd[hf_key].shape:
            print(f"SHAPE MISMATCH  {hf_key}: {expected.shape} vs {hf_sd[hf_key].shape}")
            all_match = False
        elif not torch.equal(expected, hf_sd[hf_key]):
            max_diff = (expected.float() - hf_sd[hf_key].float()).abs().max().item()
            print(f"VALUE MISMATCH  {hf_key}: max abs diff = {max_diff}")
            all_match = False
        else:
            print(f"OK  {hf_key}")

    # Global keys
    check("model.embed_tokens.weight", fms_sd["backbone.embedding.weight"] / emb_mult)
    check("model.norm.weight", fms_sd["backbone.norm_f.weight"])
    if "lm_head.weight" in fms_sd:
        check("lm_head.weight", fms_sd["lm_head.weight"] * logits_scale)

    for i in range(mamba_config.n_layer):
        layer_type = "attention" if i in attn_set else "mamba"
        mp = f"backbone.layers.{i}"
        hp = f"model.layers.{i}"

        check(f"{hp}.input_layernorm.weight", fms_sd[f"{mp}.norm.weight"])

        if layer_type == "mamba":
            for param in [
                "in_proj.weight", "conv1d.weight", "conv1d.bias",
                "dt_bias", "A_log", "D", "norm.weight", "in_proj.bias",
            ]:
                mamba_key = f"{mp}.mixer.{param}"
                if mamba_key in fms_sd:
                    check(f"{hp}.mamba.{param}", fms_sd[mamba_key])

            out_key = f"{mp}.mixer.out_proj.weight"
            if out_key in fms_sd:
                check(f"{hp}.mamba.out_proj.weight", fms_sd[out_key] / res_mult)
            out_bias = f"{mp}.mixer.out_proj.bias"
            if out_bias in fms_sd:
                check(f"{hp}.mamba.out_proj.bias", fms_sd[out_bias] / res_mult)

        elif layer_type == "attention":
            in_proj_w = fms_sd[f"{mp}.mixer.in_proj.weight"]
            in_proj_b = fms_sd.get(f"{mp}.mixer.in_proj.bias")
            q_w, k_w, v_w, q_b, k_b, v_b = split_qkv(
                in_proj_w, num_heads, num_heads_kv, head_dim, in_proj_b
            )
            check(f"{hp}.self_attn.q_proj.weight", q_w)
            check(f"{hp}.self_attn.k_proj.weight", k_w)
            check(f"{hp}.self_attn.v_proj.weight", v_w)
            if q_b is not None:
                check(f"{hp}.self_attn.q_proj.bias", q_b)
                check(f"{hp}.self_attn.k_proj.bias", k_b)
                check(f"{hp}.self_attn.v_proj.bias", v_b)

            check(f"{hp}.self_attn.o_proj.weight",
                  fms_sd[f"{mp}.mixer.out_proj.weight"] / res_mult)
            o_bias = f"{mp}.mixer.out_proj.bias"
            if o_bias in fms_sd:
                check(f"{hp}.self_attn.o_proj.bias", fms_sd[o_bias] / res_mult)

        # MLP
        check(f"{hp}.post_attention_layernorm.weight", fms_sd[f"{mp}.norm2.weight"])

        fc1_w = fms_sd[f"{mp}.mlp.fc1.weight"]
        value_w, gate_w = fc1_w.chunk(2, dim=0)
        check(f"{hp}.shared_mlp.input_linear.weight",
              torch.cat([gate_w, value_w], dim=0))
        check(f"{hp}.shared_mlp.output_linear.weight",
              fms_sd[f"{mp}.mlp.fc2.weight"] / res_mult)

    extra_hf = set(hf_sd.keys()) - matched_hf_keys
    if extra_hf:
        print(f"\nEXTRA HF KEYS not mapped from FMS ({len(extra_hf)}):")
        for ek in sorted(extra_hf):
            print(f"  {ek}")
        all_match = False

    print(f"\n{'ALL ENTRIES MATCH' if all_match else 'MISMATCHES FOUND'}")
    return all_match


# ─── Save ────────────────────────────────────────────────────────────────────


def save_single_safetensor(state_dict, save_directory, metadata):
    save_file(state_dict, os.path.join(save_directory, SAFE_WEIGHTS_NAME), metadata)


def save_sharded_safetensors(state_dict, save_directory, metadata, max_shard_size="5GB"):
    filename_pattern = SAFE_WEIGHTS_NAME.replace(".bin", "{suffix}.bin").replace(
        ".safetensors", "{suffix}.safetensors"
    )
    state_dict_split = split_torch_state_dict_into_shards(
        state_dict, filename_pattern=filename_pattern, max_shard_size=max_shard_size
    )
    index = {
        "metadata": state_dict_split.metadata,
        "weight_map": state_dict_split.tensor_to_filename,
    }
    with open(os.path.join(save_directory, SAFE_WEIGHTS_INDEX_NAME), "w", encoding="utf-8") as f:
        f.write(json.dumps(index, indent=2, sort_keys=True) + "\n")

    for shard_file, tensors in state_dict_split.filename_to_tensors.items():
        shard = {tensor: state_dict[tensor].contiguous() for tensor in tensors}
        save_file(shard, os.path.join(save_directory, shard_file), metadata=metadata)


# ─── Main ────────────────────────────────────────────────────────────────────


def fms_to_hf(model_variant, load_path, save_path, tokenizer_path,
              precision="fp32", emb_mult=1.0, res_mult=1.0, logits_scale=1.0,
              save_model=True):
    print("Loading mamba_ssm config from model variant...")
    config_data = get_model_config(model_variant)
    mamba_config = MambaConfig(**config_data)

    attn_set = set(mamba_config.attn_layer_idx or [])
    n_mamba = sum(1 for i in range(mamba_config.n_layer) if i not in attn_set)
    n_attn = len(attn_set)
    print(f"  {mamba_config.n_layer} layers, {n_mamba} mamba, {n_attn} attention")

    if emb_mult != 1.0 or res_mult != 1.0 or logits_scale != 1.0:
        print(f"  Un-folding scaling factors: emb={emb_mult}, res={res_mult}, logits={logits_scale}")

    print("Constructing HF config...")
    token_ids = {}
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    for key in ["bos_token_id", "eos_token_id", "pad_token_id"]:
        tid = getattr(tokenizer, key, None)
        if tid is not None:
            token_ids[key] = tid

    hf_config = convert_ssm_config_to_hf_config(
        mamba_config, emb_mult=emb_mult, res_mult=res_mult,
        logits_scale=logits_scale, **token_ids,
    )
    hf_config.save_pretrained(save_path)

    print("Copying tokenizer...")
    tokenizer.save_pretrained(save_path)

    print(f"Loading state dict from {load_path}...")
    mamba_sd = torch.load(load_path, map_location="cpu").get("model_state")

    print("Converting state dict to GraniteMoeHybrid HF format...")
    hf_sd = convert_state_dict(
        mamba_sd, mamba_config,
        emb_mult=emb_mult, res_mult=res_mult, logits_scale=logits_scale,
    )

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[precision]

    save_file_fn = None
    if isinstance(save_model, bool) and save_model:
        save_file_fn = save_single_safetensor
    elif isinstance(save_model, str) and save_model == "sharded":
        save_file_fn = save_sharded_safetensors

    if save_file_fn:
        save_file_fn(
            {k: v.to(dtype) for k, v in hf_sd.items()},
            save_path,
            metadata={"format": "pt"},
        )

    print(f"Saved to {save_path}")
    return mamba_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert mamba_ssm checkpoint to GraniteMoeHybrid HuggingFace format"
    )
    parser.add_argument('--src-dir', type=str, required=True,
                        help="Path to consolidated.00.pth")
    parser.add_argument('--model-variant', type=str, required=True,
                        help="Mamba model variant (e.g. granite_4_lite)")
    parser.add_argument('--model-dir', type=str, required=True,
                        help="Saving directory")
    parser.add_argument('--model-name', type=str, default='',
                        help="Model name (appended to model-dir)")
    parser.add_argument('--tokenizer', type=str, required=True,
                        help="Tokenizer name or path")
    parser.add_argument('--precision', type=str, default='fp32',
                        choices=['fp32', 'bf16', 'fp16'])
    parser.add_argument('--embedding-multiplier', type=float, default=1.0,
                        help="Embedding scaling factor to un-fold (default: 1.0 = no scaling)")
    parser.add_argument('--residual-multiplier', type=float, default=1.0,
                        help="Residual scaling factor to un-fold (default: 1.0 = no scaling)")
    parser.add_argument('--logits-scaling', type=float, default=1.0,
                        help="Logits scaling factor to un-fold (default: 1.0 = no scaling)")

    args = parser.parse_args()

    save_path = os.path.join(args.model_dir, args.model_name) if args.model_name else args.model_dir
    os.makedirs(save_path, exist_ok=True)

    mamba_config = fms_to_hf(
        model_variant=args.model_variant,
        load_path=args.src_dir,
        save_path=save_path,
        tokenizer_path=args.tokenizer,
        precision=args.precision,
        emb_mult=args.embedding_multiplier,
        res_mult=args.residual_multiplier,
        logits_scale=args.logits_scaling,
    )

    # Verify conversion
    print("\nVerifying conversion...")
    fms_sd = torch.load(args.src_dir, map_location="cpu").get("model_state")
    hf_sd = load_file(os.path.join(save_path, SAFE_WEIGHTS_NAME))
    verify_conversion(
        fms_sd, hf_sd, mamba_config,
        emb_mult=args.embedding_multiplier,
        res_mult=args.residual_multiplier,
        logits_scale=args.logits_scaling,
    )
