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
from safetensors.torch import save_file

from transformers import AutoTokenizer
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME

from mamba_ssm.models.config_mamba import MambaConfig

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

    print("Copying tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
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

    fms_to_hf(
        model_variant=args.model_variant,
        load_path=args.src_dir,
        save_path=save_path,
        tokenizer_path=args.tokenizer,
        precision=args.precision,
        emb_mult=args.embedding_multiplier,
        res_mult=args.residual_multiplier,
        logits_scale=args.logits_scaling,
    )
