# coding=utf-8
# Copyright 2024 IBM and the HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0

"""
Direct checkpoint converter: loads an FMS/mamba_ssm checkpoint state dict,
converts keys and config, and saves directly as HuggingFace safetensors.
No intermediate pytorch_model.bin or model.save_pretrained.

IMPORTANT: Does NOT change CLI args (kept exactly as you requested).
"""

import argparse
import json
import re
import os
from typing import Dict, Optional, Union

import torch
from torch.distributed._shard.checkpoint import FileSystemReader, load_state_dict

from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import save_file

from transformers import AutoTokenizer
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME
from transformers.models.bamba import BambaConfig

from fms_fsdp.utils.config_utils import get_model_config


# ---------------------------
# Key conversion: mamba_ssm -> HF Bamba
# ---------------------------

def convert_state_dict_from_mamba_ssm(original_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    state_dict: Dict[str, torch.Tensor] = {}

    for orig_k, param in original_sd.items():
        k = orig_k.replace("backbone", "model")
        k = k.replace("embedding", "embed_tokens")
        k = k.replace("mixer", "mamba")
        k = k.replace("norm_f", "final_layernorm")

        k = re.sub(r"(\d+)\.norm\.", r"\1.input_layernorm.", k)
        k = re.sub(r"(\d+)\.norm2\.", r"\1.pre_ff_layernorm.", k)

        k = k.replace("mlp.fc2", "feed_forward.down_proj")

        if "mlp.fc1" in k:
            param, param2 = torch.chunk(param, 2, dim=0)
            state_dict[k.replace("mlp.fc1", "feed_forward.gate_proj")] = param2
            k = k.replace("mlp.fc1", "feed_forward.up_proj")

        # Decide mamba vs attention based on whether conv1d exists in original_sd
        if ("in_proj" in k and orig_k.replace("in_proj", "conv1d") in original_sd) or (
            "out_proj" in k and orig_k.replace("out_proj", "conv1d") in original_sd
        ):
            # mamba layer — keep as-is
            pass
        else:
            # attention layer
            k = k.replace("mamba.out_proj", "self_attn.o_proj")
            if "mamba.in_proj" in k:
                m, n = param.shape
                d = (m - n) // 2
                q, kk, vv = torch.split(param, [n, d, d], dim=0)
                state_dict[k.replace("mamba.in_proj", "self_attn.k_proj")] = kk
                state_dict[k.replace("mamba.in_proj", "self_attn.v_proj")] = vv
                param = q
                k = k.replace("mamba.in_proj", "self_attn.q_proj")

        state_dict[k] = param

    return state_dict


# ---------------------------
# Config conversion: FMS/mamba_ssm config dict -> HF BambaConfig
# ---------------------------

def convert_ssm_config_to_hf_config(config_ssm: Dict, **kwargs) -> BambaConfig:
    hf_config = BambaConfig(**kwargs)
    hf_config.architectures = ["BambaForCausalLM"]

    hf_config.hidden_size = config_ssm["d_model"]
    hf_config.intermediate_size = config_ssm["d_intermediate"]
    hf_config.mamba_n_heads = (hf_config.hidden_size * hf_config.mamba_expand) // hf_config.mamba_d_head
    hf_config.num_hidden_layers = config_ssm["n_layer"]
    hf_config.tie_word_embeddings = config_ssm["tie_embeddings"]

    if config_ssm["ssm_cfg"].get("layer") != "Mamba2":
        raise ValueError("Conversion script only supports Mamba2")

    attn_cfg = config_ssm.get("attn_cfg")
    if attn_cfg:
        assert attn_cfg["causal"], "Only support causal attention."
        assert not attn_cfg["qkv_proj_bias"], "Only support no qkv bias."
        assert not attn_cfg["out_proj_bias"], "Only support no out bias."
        hf_config.attn_rotary_emb = attn_cfg["rotary_emb_dim"]
        hf_config.num_attention_heads = attn_cfg["num_heads"]
        hf_config.num_key_value_heads = attn_cfg["num_heads_kv"]
        hf_config.rope_parameters = {
            "rope_theta": float(attn_cfg.get("rotary_emb_base", 10000.0)),
            "rope_type": "default",
        }

    attention_layer_indices = config_ssm.get("attn_layer_idx")
    if attention_layer_indices is not None:
        hf_config.attn_layer_indices = attention_layer_indices

    vocab_size = config_ssm["vocab_size"]
    pad_mult = config_ssm["pad_vocab_size_multiple"]
    if (vocab_size % pad_mult) != 0:
        vocab_size += pad_mult - (vocab_size % pad_mult)
    hf_config.vocab_size = vocab_size

    return hf_config


# ---------------------------
# Save safetensors (single or sharded)
# ---------------------------

def save_safetensors(
    state_dict: Dict[str, torch.Tensor],
    save_directory: str,
    metadata: Dict,
    sharded: bool = True,
    max_shard_size: Union[int, str] = "5GB",
) -> None:
    os.makedirs(save_directory, exist_ok=True)

    if not sharded:
        save_file(state_dict, os.path.join(save_directory, SAFE_WEIGHTS_NAME), metadata=metadata)
        return

    filename_pattern = SAFE_WEIGHTS_NAME.replace(".bin", "{suffix}.bin").replace(
        ".safetensors", "{suffix}.safetensors"
    )
    split = split_torch_state_dict_into_shards(
        state_dict, filename_pattern=filename_pattern, max_shard_size=max_shard_size
    )

    # Always write an index when sharded (HF expects it)
    index = {"metadata": split.metadata, "weight_map": split.tensor_to_filename}
    with open(os.path.join(save_directory, SAFE_WEIGHTS_INDEX_NAME), "w", encoding="utf-8") as f:
        f.write(json.dumps(index, indent=2, sort_keys=True) + "\n")

    for shard_file, tensors in split.filename_to_tensors.items():
        shard = {t: state_dict[t].contiguous() for t in tensors}
        save_file(shard, os.path.join(save_directory, shard_file), metadata=metadata)


# ---------------------------
# Load FMS checkpoint -> raw state_dict (backbone.* keys)
# ---------------------------

def load_raw_state_dict(load_path: str) -> Dict[str, torch.Tensor]:
    """
    Loads a raw model state dict from:
      - a single file checkpoint (expects dict with 'model_state' or raw dict)
      - an FSDP sharded checkpoint directory (uses load_state_dict)

    Key point: for sharded checkpoints, load_state_dict requires an *output dict*
    to populate, but we do NOT need to instantiate a model; we can pass an empty dict.
    """
    if os.path.isfile(load_path):
        print(f"Loading checkpoint file: {load_path}")
        ckpt = torch.load(load_path, map_location="cpu", weights_only=True)
        raw = ckpt.get("model_state", ckpt)
        if not isinstance(raw, dict):
            raise ValueError("File checkpoint did not contain a dict model_state.")
        return raw

    print(f"Loading FSDP sharded checkpoint dir: {load_path}")
    state = {"model_state": {}}
    load_state_dict(state_dict=state, storage_reader=FileSystemReader(load_path), no_dist=True)
    raw = state["model_state"]
    if not isinstance(raw, dict):
        raise ValueError("Directory checkpoint load did not produce a dict model_state.")
    return raw


def apply_upi_masks_inplace(raw_sd: Dict[str, torch.Tensor], model_variant: str, upi_path: str) -> None:
    if "upi" not in model_variant:
        model_variant_upi = model_variant + "_upi"
    else:
        model_variant_upi = model_variant

    cfg = get_model_config(model_variant_upi)
    upi_mask_dict = torch.load(upi_path, map_location="cpu")

    n_layer = cfg["n_layer"]
    attn_idx = set(cfg.get("attn_layer_idx", []))

    print("Applying UPI masks...")
    for i in range(n_layer):
        if i in attn_idx:
            continue
        key = f"backbone.layers.{i}.mixer.upi_mask"
        mask = upi_mask_dict[i].to(torch.bfloat16)
        if "upi" not in model_variant:
            raw_sd[key] = mask
        else:
            raw_sd[key] = raw_sd[key] * mask


# ---------------------------
# End-to-end: FMS -> HF
# ---------------------------

def convert_and_save(
    model_variant: str,
    load_path: str,
    save_path: str,
    tokenizer_name_or_path: str,
    upi_path: Optional[str] = None,
    precision: str = "fp32",
    sharded: bool = True,
) -> None:
    os.makedirs(save_path, exist_ok=True)

    # 1) Load raw backbone.* state dict directly from FMS checkpoint
    raw_sd = load_raw_state_dict(load_path)

    # 2) Optional: modify raw state dict in-place for UPI
    if upi_path:
        apply_upi_masks_inplace(raw_sd, model_variant, upi_path)

    # 3) Build HF config from get_model_config(model_variant) (no config.json I/O)
    cfg = get_model_config(model_variant)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    token_ids = {}
    for k in ["bos_token_id", "eos_token_id", "pad_token_id"]:
        v = getattr(tokenizer, k, None)
        if v is not None:
            token_ids[k] = int(v)

    unsettables = {
        "mamba_d_head": 64,
        "mamba_d_state": 128,
        "mamba_n_groups": 1,
        "rms_norm_eps": 1e-5,
    }

    hf_config = convert_ssm_config_to_hf_config(cfg, **token_ids, **unsettables)
    hf_config.save_pretrained(save_path)

    # 4) Convert keys to HF naming
    print("Converting state dict keys to HuggingFace format...")
    hf_sd = convert_state_dict_from_mamba_ssm(raw_sd)
    del raw_sd

    # 5) Cast dtype + save safetensors
    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    if precision not in dtype_map:
        raise ValueError(f"Unknown precision: {precision}")
    dtype = dtype_map[precision]

    hf_sd = {k: v.to(dtype) for k, v in hf_sd.items()}

    print(f"Saving safetensors to: {save_path}")
    save_safetensors(hf_sd, save_path, metadata={"format": "pt"}, sharded=sharded)

    # 6) Save tokenizer
    tokenizer.save_pretrained(save_path)

    print(f"Done — HF model saved at {save_path}")


# ---------------------------
# CLI (DO NOT CHANGE THE ARGS)
# ---------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", type=str, required=True,
                        help="Model source directory or .pth file")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Model name (used as subdirectory under --model-dir)")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Root saving directory")
    parser.add_argument("--model-variant", type=str, required=True,
                        help="Mamba model type (e.g. mamba_9.8b)")
    parser.add_argument("--upi-path", type=str, default=None,
                        help="Path to a UPI scaling mask, if applicable")
    parser.add_argument("--precision", type=str, default="fp32",
                        choices=["fp32", "bf16", "fp16"],
                        help="Output precision (default: fp32)")
    args = parser.parse_args()

    DEST_DIR = os.path.join(args.model_dir, args.model_name)
    TOKENIZER_DIR = "/datasets/tokenizers/llama3"

    convert_and_save(
        model_variant=args.model_variant,
        load_path=args.src_dir,
        save_path=DEST_DIR,
        tokenizer_name_or_path=TOKENIZER_DIR,
        upi_path=args.upi_path,
        precision=args.precision,
    )
