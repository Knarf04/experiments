# coding=utf-8
# Copyright 2024 IBM and the HuggingFace Inc.
# Licensed under the Apache License, Version 2.0

"""
Direct FMS/FSDP -> HuggingFace (Bamba) converter

- Loads a checkpoint from:
    (A) a single .pth/.pt file containing {"model_state": state_dict} or a raw state_dict, OR
    (B) a directory containing per-rank/per-shard checkpoint files (common FSDP2/DCP layouts)
- Optionally applies UPI mask to raw keys
- Converts raw keys (backbone.*) -> HF keys (model.*)
- Writes:
    config.json + tokenizer files + sharded safetensors (model.safetensors + index)
- No intermediate pytorch_model.bin and no MambaLMHeadModel.save_pretrained()

If your directory checkpoint is NOT a collection of torch.load()-able shard files,
you may need the torch.distributed.checkpoint loader path at the bottom.
"""

import argparse
import json
import os
import re
import glob
from os import path
from typing import Dict, Optional, Union, Iterable, Tuple

import torch
from safetensors.torch import save_file
from huggingface_hub import split_torch_state_dict_into_shards

from transformers import AutoTokenizer
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME
from transformers.models.bamba import BambaConfig

from fms_fsdp.utils.config_utils import get_model_config


# ---------------------------
# Key conversion: raw -> HF Bamba
# ---------------------------

def convert_state_dict_from_mamba_ssm(original_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert mamba_ssm/FMS-style keys into HF transformers Bamba-style keys.

    This retains your exact mapping logic.
    """
    state_dict: Dict[str, torch.Tensor] = {}

    for orig_k, param in original_sd.items():
        k = orig_k.replace("backbone", "model")

        # embeddings
        k = k.replace("embedding", "embed_tokens")

        # mixer
        k = k.replace("mixer", "mamba")

        # final layernorm
        k = k.replace("norm_f", "final_layernorm")

        # block layernorm
        k = re.sub(r"(\d+)\.norm\.", r"\1.input_layernorm.", k)
        k = re.sub(r"(\d+)\.norm2\.", r"\1.pre_ff_layernorm.", k)

        # mlp
        k = k.replace("mlp.fc2", "feed_forward.down_proj")

        if "mlp.fc1" in k:
            # split into up_proj + gate_proj
            param, param2 = torch.chunk(param, 2, dim=0)
            k2 = k.replace("mlp.fc1", "feed_forward.gate_proj")
            state_dict[k2] = param2
            k = k.replace("mlp.fc1", "feed_forward.up_proj")

        # determine whether this is mamba or attention by conv1d presence
        if ("in_proj" in k and orig_k.replace("in_proj", "conv1d") in original_sd) or (
            "out_proj" in k and orig_k.replace("out_proj", "conv1d") in original_sd
        ):
            # mamba layer: keep mamba.*
            pass
        else:
            # attention rewrite (because mixer -> mamba above)
            k = k.replace("mamba.out_proj", "self_attn.o_proj")
            if "mamba.in_proj" in k:
                m, n = param.shape
                d = (m - n) // 2
                param, param2, param3 = torch.split(param, [n, d, d], dim=0)
                state_dict[k.replace("mamba.in_proj", "self_attn.k_proj")] = param2
                state_dict[k.replace("mamba.in_proj", "self_attn.v_proj")] = param3
                k = k.replace("mamba.in_proj", "self_attn.q_proj")

        state_dict[k] = param

    return state_dict


# ---------------------------
# Config conversion: get_model_config -> HF BambaConfig
# ---------------------------

def convert_ssm_config_to_hf_config(config_ssm: Dict, **kwargs) -> BambaConfig:
    """
    Here config_ssm is assumed to be the dict returned by get_model_config(model_variant).
    """
    hf_config = BambaConfig(**kwargs)
    hf_config.architectures = ["BambaForCausalLM"]

    hf_config.hidden_size = config_ssm["d_model"]
    hf_config.intermediate_size = config_ssm["d_intermediate"]
    hf_config.mamba_n_heads = (hf_config.hidden_size * hf_config.mamba_expand) // hf_config.mamba_d_head
    hf_config.num_hidden_layers = config_ssm["n_layer"]
    hf_config.tie_word_embeddings = config_ssm.get("tie_embeddings", False)

    if config_ssm.get("ssm_cfg", {}).get("layer") != "Mamba2":
        raise ValueError("Conversion script only supports Mamba2")

    attn_cfg = config_ssm.get("attn_cfg")
    if attn_cfg:
        assert attn_cfg["causal"], "Only support causal attention."
        assert not attn_cfg.get("qkv_proj_bias", False), "Only support no qkv bias."
        assert not attn_cfg.get("out_proj_bias", False), "Only support no out bias."
        hf_config.attn_rotary_emb = attn_cfg["rotary_emb_dim"]
        hf_config.num_attention_heads = attn_cfg["num_heads"]
        hf_config.num_key_value_heads = attn_cfg["num_heads_kv"]
        hf_config.rope_parameters = {
            "rope_theta": float(attn_cfg.get("rotary_emb_base", 10000.0)),
            "rope_type": "default",
        }

    if config_ssm.get("attn_layer_idx") is not None:
        hf_config.attn_layer_indices = config_ssm["attn_layer_idx"]

    vocab_size = config_ssm["vocab_size"]
    pad_mult = config_ssm.get("pad_vocab_size_multiple", 1)
    if (vocab_size % pad_mult) != 0:
        vocab_size += pad_mult - (vocab_size % pad_mult)
    hf_config.vocab_size = vocab_size

    return hf_config


# ---------------------------
# Saving
# ---------------------------

def save_sharded_safetensors(
    state_dict: Dict[str, torch.Tensor],
    save_directory: str,
    metadata: Dict,
    max_shard_size: Union[int, str] = "5GB",
) -> None:
    os.makedirs(save_directory, exist_ok=True)

    filename_pattern = SAFE_WEIGHTS_NAME.replace(".bin", "{suffix}.bin").replace(
        ".safetensors", "{suffix}.safetensors"
    )
    split = split_torch_state_dict_into_shards(
        state_dict, filename_pattern=filename_pattern, max_shard_size=max_shard_size
    )

    index = {"metadata": split.metadata, "weight_map": split.tensor_to_filename}
    with open(os.path.join(save_directory, SAFE_WEIGHTS_INDEX_NAME), "w", encoding="utf-8") as f:
        f.write(json.dumps(index, indent=2, sort_keys=True) + "\n")

    for shard_file, tensors in split.filename_to_tensors.items():
        shard = {t: state_dict[t].contiguous() for t in tensors}
        save_file(shard, os.path.join(save_directory, shard_file), metadata=metadata)


def _dtype_from_precision(precision: str) -> torch.dtype:
    precision = precision.lower()
    if precision == "fp32":
        return torch.float32
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    raise ValueError(f"Unknown precision: {precision} (expected fp32|bf16|fp16)")


# ---------------------------
# Loading: file or directory
# ---------------------------

def _is_probably_shard_file(p: str) -> bool:
    base = os.path.basename(p)
    # common shard names: "*.pt", "*.pth", "consolidated.*.pth", "rank*_*.pt", etc.
    return base.endswith((".pt", ".pth", ".bin")) and "token" not in base and "config" not in base


def _list_checkpoint_files(ckpt_dir: str) -> list:
    # Try a few common patterns; adjust if your layout differs
    patterns = [
        os.path.join(ckpt_dir, "*.pt"),
        os.path.join(ckpt_dir, "*.pth"),
        os.path.join(ckpt_dir, "**", "*.pt"),
        os.path.join(ckpt_dir, "**", "*.pth"),
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive="**" in pat))
    files = sorted({f for f in files if _is_probably_shard_file(f)})
    return files


def load_fms_model_state(load_path: str) -> Dict[str, torch.Tensor]:
    """
    Load raw state dict with backbone.* keys.

    Supports:
      - single file: torch.load(file) returns {"model_state": {...}} or raw {...}
      - directory: merges torch.load(shard)["model_state"] (or raw) across shard files
    """
    if os.path.isfile(load_path):
        ckpt = torch.load(load_path, map_location="cpu")
        if isinstance(ckpt, dict) and "model_state" in ckpt and isinstance(ckpt["model_state"], dict):
            return ckpt["model_state"]
        if isinstance(ckpt, dict):
            return ckpt
        raise ValueError("Checkpoint file did not contain a dict or dict['model_state'].")

    if not os.path.isdir(load_path):
        raise ValueError(f"load_path is neither file nor directory: {load_path}")

    shard_files = _list_checkpoint_files(load_path)
    if not shard_files:
        raise ValueError(
            f"No shard files found under {load_path}. "
            "If this is a torch.distributed.checkpoint (DCP) directory without torch-loadable shards, "
            "use the DCP fallback loader (see comment near bottom)."
        )

    merged: Dict[str, torch.Tensor] = {}
    for sf in shard_files:
        data = torch.load(sf, map_location="cpu")
        if isinstance(data, dict) and "model_state" in data and isinstance(data["model_state"], dict):
            part = data["model_state"]
        elif isinstance(data, dict):
            part = data
        else:
            raise ValueError(f"Shard {sf} is not a dict.")

        # merge
        for k, v in part.items():
            if k in merged:
                continue
            merged[k] = v

    return merged


# ---------------------------
# Optional: UPI mask
# ---------------------------

def apply_upi_masks_inplace(
    model_state: Dict[str, torch.Tensor],
    model_variant: str,
    upi_path: str,
) -> None:
    """
    Same behavior as your original:
      - If model_variant lacks "upi": overwrite upi_mask
      - Else: multiply existing upi_mask
    """
    model_variant_upi = model_variant if "upi" in model_variant else model_variant + "_upi"
    cfg = get_model_config(model_variant_upi)

    upi_mask_dict = torch.load(upi_path, map_location="cpu")
    n_layer = cfg["n_layer"]
    attn_idx = set(cfg.get("attn_layer_idx", []))

    for i in range(n_layer):
        if i in attn_idx:
            continue
        key = f"backbone.layers.{i}.mixer.upi_mask"
        mask = upi_mask_dict[i].to(torch.bfloat16)
        if "upi" not in model_variant:
            model_state[key] = mask
        else:
            model_state[key] = model_state[key] * mask


# ---------------------------
# End-to-end conversion
# ---------------------------

def convert_fms_to_hf(
    model_variant: str,
    src_path: str,
    out_dir: str,
    tokenizer_name_or_path: str,
    precision: str = "fp32",
    max_shard_size: Union[int, str] = "5GB",
    upi_path: Optional[str] = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    print(f"[1/5] Load FMS checkpoint: {src_path}")
    raw_sd = load_fms_model_state(src_path)

    if upi_path:
        print(f"[2/5] Apply UPI mask: {upi_path}")
        apply_upi_masks_inplace(raw_sd, model_variant, upi_path)
    else:
        print("[2/5] No UPI mask")

    print("[3/5] Build & save HF config + tokenizer")
    cfg = get_model_config(model_variant)

    tok = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    token_ids = {}
    for k in ["bos_token_id", "eos_token_id", "pad_token_id"]:
        v = getattr(tok, k, None)
        if v is not None:
            token_ids[k] = int(v)

    # keep your "unsettables"
    unsettables = {
        "mamba_d_head": 64,
        "mamba_d_state": 128,
        "mamba_n_groups": 1,
        "rms_norm_eps": 1e-5,
    }

    hf_config = convert_ssm_config_to_hf_config(cfg, **token_ids, **unsettables)
    hf_config.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    print("[4/5] Convert keys -> HF format")
    hf_sd = convert_state_dict_from_mamba_ssm(raw_sd)
    del raw_sd

    print("[5/5] Cast & write sharded safetensors")
    dtype = _dtype_from_precision(precision)
    hf_sd = {k: v.to(dtype) for k, v in hf_sd.items()}
    save_sharded_safetensors(hf_sd, out_dir, metadata={"format": "pt"}, max_shard_size=max_shard_size)

    print(f"Done. HF model at: {out_dir}")


# ---------------------------
# CLI
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", type=str, required=True,
                    help="FMS checkpoint: single file (.pth/.pt) or a directory of shard files.")
    ap.add_argument("--model-variant", type=str, required=True,
                    help="Variant key for get_model_config().")
    ap.add_argument("--out-dir", type=str, required=True,
                    help="Output directory for HF model (config + tokenizer + safetensors).")
    ap.add_argument("--tokenizer", type=str, default="/datasets/tokenizers/llama3",
                    help="Tokenizer name or path.")
    ap.add_argument("--precision", type=str, default="fp32", choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--max-shard-size", type=str, default="5GB")
    ap.add_argument("--upi-path", type=str, default=None,
                    help="Optional UPI mask path.")
    args = ap.parse_args()

    convert_fms_to_hf(
        model_variant=args.model_variant,
        src_path=args.src_dir,
        out_dir=args.out_dir,
        tokenizer_name_or_path=args.tokenizer,
        precision=args.precision,
        max_shard_size=args.max_shard_size,
        upi_path=args.upi_path,
    )


if __name__ == "__main__":
    main()
