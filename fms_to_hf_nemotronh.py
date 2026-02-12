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
from safetensors.torch import save_file

from transformers import AutoTokenizer
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME

from mamba_ssm.models.config_mamba import MambaConfig

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

    print("Copying tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
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

    fms_to_hf(
        model_variant=args.model_variant,
        load_path=args.src_dir,
        save_path=save_path,
        tokenizer_path=args.tokenizer,
        precision=args.precision,
    )
