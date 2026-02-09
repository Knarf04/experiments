# coding=utf-8
# Copyright 2024 IBM and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Direct checkpoint converter: loads an FMS/mamba_ssm checkpoint state dict,
converts keys and config, and saves directly as HuggingFace safetensors.
No intermediate pytorch_model.bin or model instantiation.
"""

import argparse
import json
import re
import os
from os import path
from typing import Dict, Optional, Union

import torch
from torch.distributed._shard.checkpoint import FileSystemReader, load_state_dict

from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import save_file

from transformers import AutoTokenizer
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME
from transformers.models.bamba import BambaConfig

from fms_fsdp.utils.config_utils import get_model_config


def convert_state_dict_from_mamba_ssm(original_sd: Dict) -> Dict[str, torch.Tensor]:
    state_dict = {}

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
            k2 = k.replace("mlp.fc1", "feed_forward.gate_proj")
            state_dict[k2] = param2
            k = k.replace("mlp.fc1", "feed_forward.up_proj")

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
                param, param2, param3 = torch.split(param, [n, d, d], dim=0)
                state_dict[k.replace("mamba.in_proj", "self_attn.k_proj")] = param2
                state_dict[k.replace("mamba.in_proj", "self_attn.v_proj")] = param3
                k = k.replace("mamba.in_proj", "self_attn.q_proj")

        state_dict[k] = param

    return state_dict


def convert_ssm_config_to_hf_config(
    config_ssm: Dict,
    **kwargs,
) -> BambaConfig:
    """Convert a config from mamba_ssm to a BambaConfig."""
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
    if attention_layer_indices:
        hf_config.attn_layer_indices = attention_layer_indices

    vocab_size = config_ssm["vocab_size"]
    pad_vocab_size_multiple = config_ssm["pad_vocab_size_multiple"]
    if (vocab_size % pad_vocab_size_multiple) != 0:
        vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
    hf_config.vocab_size = vocab_size

    return hf_config


def save_safetensors(
    state_dict: Dict,
    save_directory: str,
    metadata: Dict,
    sharded: bool = True,
    max_shard_size: Union[int, str] = "5GB",
):
    """Save state dict as safetensors, optionally sharded."""
    os.makedirs(save_directory, exist_ok=True)

    if not sharded:
        save_file(state_dict, os.path.join(save_directory, SAFE_WEIGHTS_NAME), metadata)
        return

    filename_pattern = SAFE_WEIGHTS_NAME.replace(".bin", "{suffix}.bin").replace(
        ".safetensors", "{suffix}.safetensors"
    )
    state_dict_split = split_torch_state_dict_into_shards(
        state_dict, filename_pattern=filename_pattern, max_shard_size=max_shard_size
    )

    if state_dict_split.is_sharded:
        index = {
            "metadata": state_dict_split.metadata,
            "weight_map": state_dict_split.tensor_to_filename,
        }
        with open(os.path.join(save_directory, SAFE_WEIGHTS_INDEX_NAME), "w", encoding="utf-8") as f:
            f.write(json.dumps(index, indent=2, sort_keys=True) + "\n")

    for shard_file, tensors in state_dict_split.filename_to_tensors.items():
        shard = {tensor: state_dict[tensor].contiguous() for tensor in tensors}
        save_file(shard, os.path.join(save_directory, shard_file), metadata=metadata)


def load_raw_state_dict(model_variant: str, load_path: str, upi_path: Optional[str] = None):
    """
    Load raw state dict from an FMS checkpoint without saving any intermediate files.
    For single .pth files, loads directly without model instantiation.
    For FSDP sharded checkpoints, instantiates a model only to get the state dict
    skeleton needed by load_state_dict, then discards the model immediately.
    Returns (config_data, state_dict).
    """
    config_data = get_model_config(model_variant)

    if os.path.isfile(load_path):
        # Single .pth file — load directly, no model needed
        print(f"Loading state dict from file: {load_path}")
        checkpoint_data = torch.load(load_path, map_location="cpu", weights_only=True)
        raw_sd = checkpoint_data.get("model_state", checkpoint_data)
    else:
        # FSDP sharded checkpoint — needs a skeleton state dict for load_state_dict
        print(f"Loading FSDP sharded state dict from: {load_path}")
        from mamba_ssm.models.config_mamba import MambaConfig
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

        mamba_config = MambaConfig(**config_data)
        model = MambaLMHeadModel(mamba_config)
        state_dict = {"model_state": model.state_dict()}
        del model  # free memory before loading weights
        load_state_dict(
            state_dict=state_dict, storage_reader=FileSystemReader(load_path), no_dist=True
        )
        raw_sd = state_dict["model_state"]

    # Apply UPI masks directly to the state dict if provided
    if upi_path:
        model_variant_upi = model_variant if "upi" in model_variant else model_variant + "_upi"
        config_data = get_model_config(model_variant_upi)

        print("Applying UPI masks...")
        upi_mask_dict = torch.load(upi_path, map_location="cpu")
        for i in range(config_data["n_layer"]):
            if i not in config_data["attn_layer_idx"]:
                key = f"backbone.layers.{i}.mixer.upi_mask"
                if "upi" not in model_variant:
                    raw_sd[key] = upi_mask_dict[i].to(torch.bfloat16)
                else:
                    raw_sd[key] *= upi_mask_dict[i].to(torch.bfloat16)

    return config_data, raw_sd


def convert_and_save(
    model_variant: str,
    load_path: str,
    save_path: str,
    tokenizer_name_or_path: str,
    upi_path: Optional[str] = None,
    precision: str = "fp32",
    sharded: bool = True,
):
    """
    End-to-end conversion: loads raw state dict, converts keys + config,
    and saves directly as HuggingFace safetensors. No intermediate files.
    """
    os.makedirs(save_path, exist_ok=True)

    # 1. Load raw mamba_ssm state dict (no model.save_pretrained, no pytorch_model.bin)
    config_data, raw_sd = load_raw_state_dict(model_variant, load_path, upi_path)

    # 2. Build and save HF config
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    token_ids = {}
    for key in ["bos_token_id", "eos_token_id", "pad_token_id"]:
        val = getattr(tokenizer, key, None)
        if val is not None:
            token_ids[key] = val

    unsettables = {
        "mamba_d_head": 64,
        "mamba_d_state": 128,
        "mamba_n_groups": 1,
        "rms_norm_eps": 1e-5,
    }

    hf_config = convert_ssm_config_to_hf_config(config_data, **token_ids, **unsettables)
    hf_config.save_pretrained(save_path)

    # 3. Convert state dict keys directly (mamba_ssm -> HF)
    print("Converting state dict keys to HuggingFace format...")
    hf_state_dict = convert_state_dict_from_mamba_ssm(raw_sd)
    del raw_sd  # free memory

    # 4. Cast to target dtype and save as safetensors
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[precision]
    hf_state_dict = {k: v.to(dtype) for k, v in hf_state_dict.items()}

    print(f"Saving safetensors to {save_path} ...")
    save_safetensors(hf_state_dict, save_path, metadata={"format": "pt"}, sharded=sharded)

    # 5. Save tokenizer
    print("Saving tokenizer...")
    tokenizer.save_pretrained(save_path)

    print(f"Done — model saved at {save_path}")


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
