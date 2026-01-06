import logging
import json
import os

import torch
from peft import get_peft_model_state_dict
from safetensors.torch import save_file, load_file
from tqdm import tqdm
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp import FullOptimStateDictConfig, FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from .torch_utils import set_logging


def get_kohya_state_dict(lora_layers, prefix="lora", dtype=torch.float32):
    kohya_ss_state_dict = {}
    for peft_key, weight in lora_layers.items():
        kohya_key = peft_key.replace("base_model.model", prefix)
        kohya_key = kohya_key.replace("lora_A", "lora_down")
        kohya_key = kohya_key.replace("lora_B", "lora_up")
        kohya_key = kohya_key.replace(".", "_", kohya_key.count(".") - 2)
        kohya_ss_state_dict[kohya_key] = weight.to(dtype)

    return kohya_ss_state_dict


def get_diffusers_state_dict(lora_layers, dtype=torch.float32):
    diffusers_ss_state_dict = {}
    for peft_key, weight in lora_layers.items():
        diffusers_key = peft_key.replace("base_model.model", "diffusion_model")
        diffusers_ss_state_dict[diffusers_key] = weight.to(dtype)

    return diffusers_ss_state_dict


def save_lora_checkpoint(transformer, rank, output_dir, step, ema=False):
    with FSDP.state_dict_type(
        transformer,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        full_state_dict = transformer.state_dict()

    if rank <= 0:
        if ema:
            save_dir = os.path.join(output_dir, f"checkpoint-{step}-ema")
        else:
            save_dir = os.path.join(output_dir, f"checkpoint-{step}")
        os.makedirs(save_dir, exist_ok=True)

        # save lora weight
        transformer_lora_layers = get_peft_model_state_dict(
            model=transformer, state_dict=full_state_dict
        )
        kohya_ss_state_dict = get_kohya_state_dict(lora_layers=transformer_lora_layers)
        diffusers_ss_state_dict = get_diffusers_state_dict(
            lora_layers=transformer_lora_layers
        )
        save_transformer_name = "pytorch_lora_transformers_weights.safetensors"
        save_kohya_name = "pytorch_lora_kohya_weights.safetensors"
        save_diffusers_name = "pytorch_lora_diffusers_weights.safetensors"

        save_file(transformer_lora_layers, os.path.join(save_dir, save_transformer_name))
        save_file(kohya_ss_state_dict, os.path.join(save_dir, save_kohya_name))
        save_file(diffusers_ss_state_dict, os.path.join(save_dir, save_diffusers_name))


def save_checkpoint(transformer, rank, output_dir, step, ema=False):
    with FSDP.state_dict_type(
        transformer,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        cpu_state = transformer.state_dict()

    if rank <= 0:
        if ema:
            save_dir = os.path.join(output_dir, f"checkpoint-{step}-ema")
        else:
            save_dir = os.path.join(output_dir, f"checkpoint-{step}")
        os.makedirs(save_dir, exist_ok=True)

        max_bytes = 5 * 1024 ** 3  # 5GB
        total_bytes = sum(v.numel() * v.element_size() for v in cpu_state.values())

        if total_bytes <= max_bytes:
            save_name = "diffusion_pytorch_model.safetensors"
            save_file(cpu_state, os.path.join(save_dir, save_name))
        else:
            shard, shards, current_size = {}, [], 0
            for k, v in sorted(cpu_state.items()):
                tensor_size = v.numel() * v.element_size()
                if current_size + tensor_size > max_bytes and shard:
                    shards.append(shard)
                    shard, current_size = {}, 0
            
                shard[k], current_size = v, current_size + tensor_size
            if shard:
                shards.append(shard)
            
            index_data = {
                "metadata": {
                    "total_size": total_bytes,
                },
                "weight_map": {}
            }

            for i, shard in enumerate(shards, start=1):
                save_name = f"diffusion_pytorch_model-{i:05}-of-{len(shards):05}.safetensors"
                save_file(shard, os.path.join(save_dir, save_name))
                for key in shard.keys():
                    index_data["weight_map"][key] = save_name
        
            with open(os.path.join(save_dir, "diffusion_pytorch_model.safetensors.index.json"), "w") as f:
                json.dump(index_data, f, indent=2)
            
        config_dict = dict(transformer.config)
        if "dtype" in config_dict:
            del config_dict["dtype"]  # TODO
        config_path = os.path.join(save_dir, "config.json")
        # save dict as json
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)

def load_state_dict(model_dir, postfix=".safetensors"):
    chunk_path_list = [os.path.join(model_dir, name) for name in os.listdir(model_dir) if name.endswith(postfix)]
    chunk_length = len(chunk_path_list)

    state_dict = {}
    for chunk_path in tqdm(chunk_path_list, total=chunk_length):
        if postfix == ".safetensors":
            chunk_state_dict = load_file(chunk_path, device="cpu")
        else:
            chunk_state_dict = torch.load(chunk_path, map_location="cpu")
        if "module" in chunk_state_dict.keys():
            chunk_state_dict = chunk_state_dict["module"]
        state_dict.update(chunk_state_dict)

    return state_dict


def print_parameters_information(model, name="Model name", rank=0):

    def format_params(params):
        if params < 1e6:
            return f"{params} (less than 1M)"
        elif params < 1e9:
            return f"{params / 1e6:.2f}M"
        else:
            return f"{params / 1e9:.2f}B"

    if model is None:
        logging.info(f"name {name} is none objects.")
        return

    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    param = next(model.parameters())
    logging.info(
        f"name [{name}] trainable params: {format_params(trainable_params)} || all params: {format_params(all_param)} || trainable%: {100 * trainable_params / all_param:.2f} || device: {param.device}, dtype: {param.dtype}."
    )
    logging.info(f"name [{name}] device: {param.device} || dtype: {param.dtype}.")

@torch.no_grad
def update_ema_model(transformer, ema_transformer, ema_decay):
    for p_averaged, p_model in zip(ema_transformer.parameters(), transformer.parameters()):
        if p_model.requires_grad:
            p_averaged.data.mul_(ema_decay).add_(p_model.data, alpha=1 - ema_decay)
