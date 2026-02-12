"""Convert mamba_ssm / fms-fsdp checkpoint to NemotronH HuggingFace format.

Reverse of the NemotronH path in hf_to_mamba_ssm.py.

Usage:
    python fms_to_hf_nemotronh.py --src-dir /path/to/consolidated.00.pth --model-variant nemotron_h_8b --model-dir /path/to/output --tokenizer /path/to/tokenizer
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
from transformers.models.nemotron_h import NemotronHConfig

from fms_fsdp.utils.config_utils import get_model_config


# ─── Utilities ───────────────────────────────────────────────────────────────


def reconstruct_blocks(mamba_config):
    """Reconstruct the NemotronH block→HF layer mapping from MambaConfig.

    Each mamba_ssm block maps to 1 or 2 NemotronH layers:
      - 1 mixer layer (M or *) always present
      - 1 MLP layer (-) if d_intermediate > 0

    Returns list of dicts with: type, mixer_idx, mlp_idx (HF layer indices).
    """
    attn_set = set(mamba_config.attn_layer_idx or [])
    d_inter = mamba_config.d_intermediate

    blocks = []
    hf_layer_idx = 0
    for block_idx in range(mamba_config.n_layer):
        block_type = "attention" if block_idx in attn_set else "mamba"
        if isinstance(d_inter, list):
            has_mlp = d_inter[block_idx] > 0
        else:
            has_mlp = d_inter > 0

        mixer_idx = hf_layer_idx
        mlp_idx = hf_layer_idx + 1 if has_mlp else None
        blocks.append({"type": block_type, "mixer_idx": mixer_idx, "mlp_idx": mlp_idx})
        hf_layer_idx += 1 + (1 if has_mlp else 0)

    return blocks


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


def reconstruct_pattern(blocks):
    """Reconstruct the NemotronH hybrid_override_pattern from blocks."""
    chars = []
    for block in blocks:
        chars.append("*" if block["type"] == "attention" else "M")
        if block["mlp_idx"] is not None:
            chars.append("-")
    return "".join(chars)


def convert_ssm_config_to_hf_config(mamba_config, blocks, **kwargs):
    """Convert MambaConfig to NemotronHConfig."""
    ssm = mamba_config.ssm_cfg
    attn = mamba_config.attn_cfg
    pattern = reconstruct_pattern(blocks)

    expand = ssm.get("expand", 2)
    headdim = ssm.get("headdim", 64)
    mamba_num_heads = expand * mamba_config.d_model // headdim

    hf_config = NemotronHConfig(
        vocab_size=mamba_config.vocab_size,
        hidden_size=mamba_config.d_model,
        intermediate_size=max(d for d in (mamba_config.d_intermediate
                                          if isinstance(mamba_config.d_intermediate, list)
                                          else [mamba_config.d_intermediate])
                              if d > 0),
        num_hidden_layers=len(pattern),
        hybrid_override_pattern=pattern,
        num_attention_heads=attn.get("num_heads", 32),
        attention_head_dim=attn.get("head_dim", 128),
        num_key_value_heads=attn.get("num_heads_kv", 8),
        attention_bias=attn.get("qkv_proj_bias", False),
        mlp_hidden_act="relu2",
        use_bias=ssm.get("bias", False),
        layer_norm_epsilon=mamba_config.norm_epsilon,
        residual_in_fp32=mamba_config.residual_in_fp32,
        tie_word_embeddings=mamba_config.tie_embeddings,
        ssm_state_size=ssm.get("d_state", 128),
        mamba_num_heads=mamba_num_heads,
        mamba_n_groups=ssm.get("ngroups", 8),
        mamba_head_dim=headdim,
        mamba_d_conv=ssm.get("d_conv", 4),
        mamba_expand=expand,
        mamba_chunk_size=ssm.get("chunk_size", 256),
        mamba_conv_bias=ssm.get("conv_bias", True),
        mamba_proj_bias=ssm.get("bias", False),
        **kwargs,
    )
    hf_config.architectures = ["NemotronHForCausalLM"]
    return hf_config


# ─── Weight Conversion ───────────────────────────────────────────────────────


def convert_state_dict(mamba_sd, blocks, attn_cfg):
    """Convert mamba_ssm state dict to NemotronH HF format.

    Reverse of convert_nemotronh_weights in hf_to_mamba_ssm.py.
    """
    hf_sd = {}

    num_heads = attn_cfg["num_heads"]
    num_heads_kv = attn_cfg["num_heads_kv"]
    head_dim = attn_cfg.get("head_dim", 128)

    # Embedding: backbone.embedding → backbone.embeddings
    hf_sd["backbone.embeddings.weight"] = mamba_sd["backbone.embedding.weight"]

    # Final norm
    hf_sd["backbone.norm_f.weight"] = mamba_sd["backbone.norm_f.weight"]

    # LM head
    if "lm_head.weight" in mamba_sd:
        hf_sd["lm_head.weight"] = mamba_sd["lm_head.weight"]

    for block_idx, block in enumerate(blocks):
        mixer_idx = block["mixer_idx"]
        mlp_idx = block["mlp_idx"]
        mamba_prefix = f"backbone.layers.{block_idx}"
        hf_prefix = f"backbone.layers.{mixer_idx}"

        # Norm
        hf_sd[f"{hf_prefix}.norm.weight"] = mamba_sd[f"{mamba_prefix}.norm.weight"]

        if block["type"] == "mamba":
            for param in [
                "in_proj.weight", "conv1d.weight", "conv1d.bias",
                "dt_bias", "A_log", "D", "norm.weight", "out_proj.weight",
                "in_proj.bias", "out_proj.bias",
            ]:
                mamba_key = f"{mamba_prefix}.mixer.{param}"
                if mamba_key in mamba_sd:
                    hf_sd[f"{hf_prefix}.mixer.{param}"] = mamba_sd[mamba_key]

        elif block["type"] == "attention":
            in_proj_w = mamba_sd[f"{mamba_prefix}.mixer.in_proj.weight"]
            in_proj_b = mamba_sd.get(f"{mamba_prefix}.mixer.in_proj.bias")

            q_w, k_w, v_w, q_b, k_b, v_b = split_qkv(
                in_proj_w, num_heads, num_heads_kv, head_dim, in_proj_b
            )

            hf_sd[f"{hf_prefix}.mixer.q_proj.weight"] = q_w
            hf_sd[f"{hf_prefix}.mixer.k_proj.weight"] = k_w
            hf_sd[f"{hf_prefix}.mixer.v_proj.weight"] = v_w
            if q_b is not None:
                hf_sd[f"{hf_prefix}.mixer.q_proj.bias"] = q_b
                hf_sd[f"{hf_prefix}.mixer.k_proj.bias"] = k_b
                hf_sd[f"{hf_prefix}.mixer.v_proj.bias"] = v_b

            hf_sd[f"{hf_prefix}.mixer.o_proj.weight"] = mamba_sd[
                f"{mamba_prefix}.mixer.out_proj.weight"
            ]
            o_bias_key = f"{mamba_prefix}.mixer.out_proj.bias"
            if o_bias_key in mamba_sd:
                hf_sd[f"{hf_prefix}.mixer.o_proj.bias"] = mamba_sd[o_bias_key]

        # MLP
        if mlp_idx is not None:
            mlp_hf_prefix = f"backbone.layers.{mlp_idx}"
            hf_sd[f"{mlp_hf_prefix}.norm.weight"] = mamba_sd[f"{mamba_prefix}.norm2.weight"]
            hf_sd[f"{mlp_hf_prefix}.mixer.up_proj.weight"] = mamba_sd[f"{mamba_prefix}.mlp.fc1.weight"]
            hf_sd[f"{mlp_hf_prefix}.mixer.down_proj.weight"] = mamba_sd[f"{mamba_prefix}.mlp.fc2.weight"]

            for fc, proj in [("fc1", "up_proj"), ("fc2", "down_proj")]:
                mamba_key = f"{mamba_prefix}.mlp.{fc}.bias"
                if mamba_key in mamba_sd:
                    hf_sd[f"{mlp_hf_prefix}.mixer.{proj}.bias"] = mamba_sd[mamba_key]

    return hf_sd


# ─── Verification ────────────────────────────────────────────────────────────


def verify_conversion(fms_sd, hf_sd, blocks, attn_cfg):
    """Verify the conversion by checking each FMS key maps correctly to HF keys.

    Applies the same mapping logic used in convert_state_dict and checks
    that each resulting tensor matches what's in hf_sd.
    """
    all_match = True
    matched_hf_keys = set()

    num_heads = attn_cfg["num_heads"]
    num_heads_kv = attn_cfg["num_heads_kv"]
    head_dim = attn_cfg.get("head_dim", 128)

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
    check("backbone.embeddings.weight", fms_sd["backbone.embedding.weight"])
    check("backbone.norm_f.weight", fms_sd["backbone.norm_f.weight"])
    if "lm_head.weight" in fms_sd:
        check("lm_head.weight", fms_sd["lm_head.weight"])

    for block_idx, block in enumerate(blocks):
        mixer_idx = block["mixer_idx"]
        mlp_idx = block["mlp_idx"]
        mamba_prefix = f"backbone.layers.{block_idx}"
        hf_prefix = f"backbone.layers.{mixer_idx}"

        check(f"{hf_prefix}.norm.weight", fms_sd[f"{mamba_prefix}.norm.weight"])

        if block["type"] == "mamba":
            for param in [
                "in_proj.weight", "conv1d.weight", "conv1d.bias",
                "dt_bias", "A_log", "D", "norm.weight", "out_proj.weight",
                "in_proj.bias", "out_proj.bias",
            ]:
                mamba_key = f"{mamba_prefix}.mixer.{param}"
                if mamba_key in fms_sd:
                    check(f"{hf_prefix}.mixer.{param}", fms_sd[mamba_key])

        elif block["type"] == "attention":
            in_proj_w = fms_sd[f"{mamba_prefix}.mixer.in_proj.weight"]
            in_proj_b = fms_sd.get(f"{mamba_prefix}.mixer.in_proj.bias")
            q_w, k_w, v_w, q_b, k_b, v_b = split_qkv(
                in_proj_w, num_heads, num_heads_kv, head_dim, in_proj_b
            )
            check(f"{hf_prefix}.mixer.q_proj.weight", q_w)
            check(f"{hf_prefix}.mixer.k_proj.weight", k_w)
            check(f"{hf_prefix}.mixer.v_proj.weight", v_w)
            if q_b is not None:
                check(f"{hf_prefix}.mixer.q_proj.bias", q_b)
                check(f"{hf_prefix}.mixer.k_proj.bias", k_b)
                check(f"{hf_prefix}.mixer.v_proj.bias", v_b)

            check(f"{hf_prefix}.mixer.o_proj.weight",
                  fms_sd[f"{mamba_prefix}.mixer.out_proj.weight"])
            o_bias = f"{mamba_prefix}.mixer.out_proj.bias"
            if o_bias in fms_sd:
                check(f"{hf_prefix}.mixer.o_proj.bias", fms_sd[o_bias])

        if mlp_idx is not None:
            mlp_hf = f"backbone.layers.{mlp_idx}"
            check(f"{mlp_hf}.norm.weight", fms_sd[f"{mamba_prefix}.norm2.weight"])
            check(f"{mlp_hf}.mixer.up_proj.weight", fms_sd[f"{mamba_prefix}.mlp.fc1.weight"])
            check(f"{mlp_hf}.mixer.down_proj.weight", fms_sd[f"{mamba_prefix}.mlp.fc2.weight"])
            for fc, proj in [("fc1", "up_proj"), ("fc2", "down_proj")]:
                mamba_key = f"{mamba_prefix}.mlp.{fc}.bias"
                if mamba_key in fms_sd:
                    check(f"{mlp_hf}.mixer.{proj}.bias", fms_sd[mamba_key])

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
              precision="fp32", save_model=True):
    print("Loading mamba_ssm config from model variant...")
    config_data = get_model_config(model_variant)
    mamba_config = MambaConfig(**config_data)

    blocks = reconstruct_blocks(mamba_config)
    print(f"  {len(blocks)} blocks, "
          f"{sum(1 for b in blocks if b['type'] == 'mamba')} mamba, "
          f"{sum(1 for b in blocks if b['type'] == 'attention')} attention, "
          f"{sum(1 for b in blocks if b['mlp_idx'] is not None)} with MLP")

    print("Constructing HF config...")
    token_ids = {}
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    for key in ["bos_token_id", "eos_token_id", "pad_token_id"]:
        tid = getattr(tokenizer, key, None)
        if tid is not None:
            token_ids[key] = tid

    hf_config = convert_ssm_config_to_hf_config(mamba_config, blocks, **token_ids)
    hf_config.save_pretrained(save_path)

    print("Copying tokenizer...")
    tokenizer.save_pretrained(save_path)

    print(f"Loading state dict from {load_path}...")
    mamba_sd = torch.load(load_path, map_location="cpu").get("model_state")

    print("Converting state dict to NemotronH HF format...")
    hf_sd = convert_state_dict(mamba_sd, blocks, mamba_config.attn_cfg)

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
    return blocks, mamba_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert mamba_ssm checkpoint to NemotronH HuggingFace format"
    )
    parser.add_argument('--src-dir', type=str, required=True,
                        help="Path to consolidated.00.pth")
    parser.add_argument('--model-variant', type=str, required=True,
                        help="Mamba model variant (e.g. nemotron_h_8b)")
    parser.add_argument('--model-dir', type=str, required=True,
                        help="Saving directory")
    parser.add_argument('--model-name', type=str, default='',
                        help="Model name (appended to model-dir)")
    parser.add_argument('--tokenizer', type=str, required=True,
                        help="Tokenizer name or path")
    parser.add_argument('--precision', type=str, default='fp32',
                        choices=['fp32', 'bf16', 'fp16'])

    args = parser.parse_args()

    save_path = os.path.join(args.model_dir, args.model_name) if args.model_name else args.model_dir
    os.makedirs(save_path, exist_ok=True)

    blocks, mamba_config = fms_to_hf(
        model_variant=args.model_variant,
        load_path=args.src_dir,
        save_path=save_path,
        tokenizer_path=args.tokenizer,
        precision=args.precision,
    )

    # Verify conversion
    print("\nVerifying conversion...")
    fms_sd = torch.load(args.src_dir, map_location="cpu").get("model_state")
    hf_sd = load_file(os.path.join(save_path, SAFE_WEIGHTS_NAME))
    verify_conversion(fms_sd, hf_sd, blocks, mamba_config.attn_cfg)
