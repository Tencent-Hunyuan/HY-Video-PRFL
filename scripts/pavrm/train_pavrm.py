import argparse
import json
import logging
import os
import time
import itertools
from copy import deepcopy
from collections import deque
from easydict import EasyDict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

import torch.amp as amp
from diffusers.optimization import get_scheduler
from einops import rearrange
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from diffusers_lite.constants import PRECISION_TO_TYPE
from diffusers_lite.datasets.image2video_dataset import Image2VideoTrainDataset
from diffusers_lite.schedulers.scheduling_flow_match_discrete import (
    FlowMatchDiscreteScheduler,
)
from diffusers_lite.wan.modules.model import WanModel
from diffusers_lite.wan.modules.t5 import T5EncoderModel
from diffusers_lite.wan.modules.vae import WanVAE
from diffusers_lite.wan.modules.clip import CLIPModel
from diffusers_lite.utils.communication import (
    broadcast,
    sp_parallel_dataloader_wrapper_wanx,
)
from diffusers_lite.utils.data_utils import (
    LengthGroupedSampler,
    save_videos_grid,
    crop_tensor,
    BlockDistributedSampler,
    VideoImageBatchIterator
)
from diffusers_lite.utils.fsdp_utils import (
    apply_fsdp_checkpointing,
    get_dit_fsdp_kwargs,
)
from diffusers_lite.utils.parallel_states import initialize_sequence_parallel_state,nccl_info,get_sequence_parallel_state
from diffusers_lite.utils.torch_utils import set_manual_seed, free_memory, set_logging, set_worker_seed_builder
from diffusers_lite.utils.diffusion_utils import (
    batch2list,
    list2batch,
    vae_encode,
    vae_decode,
    image_encode,
    prompt2states,
    load_lora_state_dict,
    transformer_zero_init,
    prepare_video_condition_wanx,
    stable_mse_loss,
)
from diffusers_lite.utils.model_utils import (
    save_lora_checkpoint,
    save_checkpoint,
    load_state_dict,
    print_parameters_information,
    update_ema_model,
)

from diffusers_lite.utils.network import MLP, QueryAttention, forward_siamese, forward_mlp, train_model, save_model

NAME_MAPPING = {
    "t2v-1.3b": "Wan2.1-T2V-1.3B",
    "t2v-14b": "Wan2.1-T2V-14B",
    "i2v-1.3b": "Wan2.1-T2V-1.3B",
    "i2v-14b-480p": "Wan2.1-I2V-14B-480P",
    "i2v-14b-720p": "Wan2.1-I2V-14B-720P",
    "flf2v-14b-720p": "Wan2.1-FLF2V-14B-720P",
}

def validate_model_parameters(model, model_name="model"):
    has_invalid = False
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        total_params += 1
        if param.requires_grad:
            trainable_params += 1
            
        if torch.isnan(param).any():
            logging.error(f"ERROR: {model_name} parameter {name} contains NaN values!")
            has_invalid = True
        if torch.isinf(param).any():
            logging.error(f"ERROR: {model_name} parameter {name} contains Inf values!")
            has_invalid = True
    
    logging.info(f"{model_name}: {trainable_params}/{total_params} parameters are trainable")
    
    if has_invalid:
        logging.warning(f"WARNING: {model_name} has invalid parameters!")
        return False
    return True

def basic_init(config):
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dtype = PRECISION_TO_TYPE[config.train.precision]
    initialize_sequence_parallel_state(config.dataset.sp_size)
    set_logging(local_rank)

    set_manual_seed(config.train.seed + nccl_info.group_id)
    logging.info(f"lanuch with seed {config.train.seed + rank}")

    config.save.ckpt_dir = os.path.join(
        config.save.output_dir, f"{config.train_id}/checkpoints"
    )
    config.save.log_dir = os.path.join(
        config.save.output_dir, f"{config.train_id}/logs"
    )
    config.save.sanity_check_dir = f"outputs/sanity_check/wanx/{config.train_id}"
    config.save.tensorboard_dir = os.path.join(config.save.output_dir, f"{config.train_id}/tensorboard")
    config.save.mlp_dir = os.path.join(config.save.output_dir, f"{config.train_id}/mlp")
    log_path = os.path.join(config.save.log_dir, "log.txt")

    if rank == 0:
        os.makedirs(config.save.output_dir, exist_ok=True)
        os.makedirs(config.save.ckpt_dir, exist_ok=True)
        os.makedirs(config.save.log_dir, exist_ok=True)
        os.makedirs(config.save.tensorboard_dir, exist_ok=True)
        OmegaConf.save(config, os.path.join(config.save.log_dir, "train_config.yaml"))
        if not os.path.exists(log_path):
            with open(log_path, "w") as f:
                f.write(f"Start logging {config.train_id}:\n")
        if config.train.sanity_check_interval > 0:
            os.makedirs(config.save.sanity_check_dir, exist_ok=True)
        logging.info(f"save ckpt directory {config.save.ckpt_dir}")

    if config.train.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        logging.info(f"enable TF32")

    basic_kwargs = EasyDict(
        {
            "local_rank": local_rank,
            "rank": rank,
            "world_size": world_size,
            "device": device,
            "dtype": dtype,
            "log_path": log_path,
        }
    )

    return config, basic_kwargs

def model_init(config, basic_kwargs):
    assert config.task in NAME_MAPPING.keys()
    base_dir = config.model.base_path

    if config.model.init_transformer_path:
        logging.info(f"loading model tranformer from {config.model.init_transformer_path}") 
        transformer = WanModel.from_pretrained(config.model.init_transformer_path)
        resume_step = 0
    else:
        if config.task in [
            "t2v-1.3b",
            "t2v-14b",
            "i2v-14b-480p",
            "i2v-14b-720p",
            "flf2v-14b-720p",
        ]:
            logging.info(f"loading model tranformer from {base_dir}")
            transformer = WanModel.from_pretrained(base_dir)
        elif config.task in ["i2v-1.3b"]:
            transformer_config = json.load(
                open(os.path.join(base_dir, "config.json"), "r")
            )
            transformer_config["in_dim"] = 36
            transformer_config["model_type"] = "i2v"
            transformer = WanModel.from_config(transformer_config)
            transformer = transformer_zero_init(transformer)
            state_dict = load_state_dict(model_dir=base_dir)

            del state_dict["patch_embedding.bias"]
            del state_dict["patch_embedding.weight"]

            m, u = transformer.load_state_dict(state_dict, strict=False)
            logging.info(f"load lora from {base_dir}.")
            logging.info(f"miss {len(m)}; unexpect {len(u)}.")
        resume_step = 0

    frozen_modules = [
        'patch_embedding',
        'text_embedding',
        'time_embedding',
        'time_projection',
        'img_emb',
        # 'freqs',
    ]

    for module_name in frozen_modules:
        if hasattr(transformer, module_name):
            module = getattr(transformer, module_name)
            for param in module.parameters():
                param.requires_grad = False

    trainable_blocks = config.lrm.trainable_blocks

    logging.info(f"freezing all blocks except for {trainable_blocks}")

    new_blocks = []
    for i, block in enumerate(transformer.blocks):
        if i in trainable_blocks:
            logging.info(f"block {i} is set to be trainable.")
            for param in block.parameters():
                param.requires_grad = True
            new_blocks.append(block)
        else:
            logging.info(f"block {i} is frozen and removed.")
            for param in block.parameters():
                param.requires_grad = False

    transformer.blocks = nn.ModuleList(new_blocks)

    if hasattr(transformer, 'head'):
        del transformer.head
        transformer.head = None

    transformer.__class__.enable_teacache = False

    if config.model.lora.use_lora:
        lora_config = LoraConfig(
            r=config.model.lora.lora_rank,
            lora_alpha=config.model.lora.lora_rank,
            init_lora_weights=True,
            target_modules=config.model.lora.target_modules,
        )
        transformer = get_peft_model(transformer, lora_config)
        if config.model.lora.resume_lora_path:
            lora_state_dict = load_lora_state_dict(config.model.lora.resume_lora_path)
            m, u = transformer.load_state_dict(lora_state_dict, strict=False)
            logging.info(f"load lora from {config.model.lora.resume_lora_path}.")
            logging.info(f"miss {len(m)}; unexpect {len(u)}.")
            resume_step = int(config.model.lora.resume_lora_path.split("-")[-1])
    
    if config.model.resume_transformer_path:
        logging.info(f"loading model tranformer from {config.model.resume_transformer_path}")
        state_dict = load_state_dict(model_dir=config.model.resume_transformer_path)
        m, u = transformer.load_state_dict(state_dict, strict=False)
        logging.info(f"miss {len(m)}; unexpect {len(u)}.")
        resume_step = int(config.model.resume_transformer_path.split("-")[-1].split('.')[0])
    
    transformer = transformer.to(dtype=torch.float32)

    if config.model.ema.use_ema:
        logging.info("loading ema model")
        ema_transformer = deepcopy(transformer)

    else:
        ema_transformer = None

    fsdp_kwargs, no_split_modules = get_dit_fsdp_kwargs(
        transformer,
        config.model.fsdp.fsdp_sharding_startegy,
        config.model.lora.use_lora,
        config.model.fsdp.use_cpu_offload,
        master_weight_type="fp32",
    )

    if config.model.lora.use_lora:
        transformer.config.lora_rank = config.model.lora.lora_rank
        transformer.config.lora_alpha = config.model.lora.lora_rank
        transformer.config.lora_target_modules = config.model.lora.target_modules
        transformer._no_split_modules = [cls.__name__ for cls in no_split_modules]
        fsdp_kwargs["auto_wrap_policy"] = fsdp_kwargs["auto_wrap_policy"](transformer)

    transformer = FSDP(transformer, **fsdp_kwargs)

    if config.model.ema.use_ema:
        ema_transformer = FSDP(ema_transformer, **fsdp_kwargs)

    if config.model.gradient_checkpointing:
        apply_fsdp_checkpointing(
            transformer, no_split_modules, config.model.selective_checkpointing
        )
        if config.model.ema.use_ema:
            apply_fsdp_checkpointing(
                ema_transformer, no_split_modules, config.model.selective_checkpointing
            )
        logging.info("enable gradient checkpointing")

    transformer.train()
    print_parameters_information(transformer, "WAN", basic_kwargs.rank)

    if not validate_model_parameters(transformer, "Transformer"):
        logging.warning("transformer has invalid parameters!")

    if config.model.ema.use_ema:
        ema_transformer.requires_grad_(False)
        print_parameters_information(ema_transformer, "WAN EMA", basic_kwargs.rank)

        if not validate_model_parameters(ema_transformer, "EMA Transformer"):
            logging.warning("EMA Transformer has invalid parameters!")
    
    feature_layer = config.lrm.feature_layer
    mlp_input_dim = config.lrm.mlp_dim
    
    mlp = MLP(mlp_input_dim)
    if config.model.resume_mlp_path:
        logging.info(f"loading model mlp from {config.model.resume_mlp_path}")
        checkpoint = torch.load(config.model.resume_mlp_path)
        mlp.load_state_dict(checkpoint)
        resume_step = int(config.model.resume_mlp_path.split("_")[-1].split('.')[0])
    mlp.train()
    mlp = mlp.to(device=basic_kwargs.device, dtype=torch.float32)
    
    if not validate_model_parameters(mlp, "MLP"):
        logging.error("MLP has invalid parameters!")
        raise ValueError("MLP initialization failed")

    query_attention_config = getattr(config.lrm, 'query_attention', {})
    num_queries = query_attention_config.get('num_queries', 1)
    num_heads = query_attention_config.get('num_heads', 8)
    dropout = query_attention_config.get('dropout', 0.)
    layer_norm = query_attention_config.get('layer_norm', False)
    return_type = query_attention_config.get('return_type', None)
    product_text = query_attention_config.get('product_text', False)
    text_dim = query_attention_config.get('text_dim', 4096)

    query_attention = QueryAttention(
        feature_dim=mlp_input_dim, 
        num_queries=num_queries, 
        num_heads=num_heads, 
        dropout=dropout,
        return_type=return_type,
        product_text=product_text,
        text_dim=text_dim
    )
    if hasattr(config.model, 'resume_query_attention_path') and config.model.resume_query_attention_path:
        logging.info(f"loading model query_attention from {config.model.resume_query_attention_path}")
        checkpoint = torch.load(config.model.resume_query_attention_path)
        query_attention.load_state_dict(checkpoint)
    query_attention.train()
    query_attention = query_attention.to(device=basic_kwargs.device, dtype=torch.float32)
    
    if not validate_model_parameters(query_attention, "QueryAttention"):
        logging.error("QueryAttention has invalid parameters!")
        raise ValueError("QueryAttention initialization failed")

    criterion = nn.BCELoss()
    criterion = criterion.to(device=basic_kwargs.device)

    model_kwargs = EasyDict(
        {
            "transformer": transformer,
            "ema_transformer": ema_transformer,
            "resume_step": resume_step,
            "feature_layer": feature_layer,
            "mlp": mlp,
            "query_attention": query_attention,
            "criterion": criterion,
        }
    )
    return model_kwargs

def extra_model_init(config, basic_kwargs):
    base_dir = config.model.base_path

    noise_scheduler = FlowMatchDiscreteScheduler(
        shift=config.extra_model.scheduler.flow_shift
    )
    noise_scheduler.set_timesteps(
        config.extra_model.scheduler.num_train_timesteps, dtype=torch.int64
    )

    if config.train.sanity_check_interval > 0:
        vae = WanVAE(
            vae_pth=os.path.join(base_dir, config.extra_model.vae.name),
            device=basic_kwargs.device,
        )
        print_parameters_information(vae.model, "VAE", basic_kwargs.rank)
    else:
        vae = None

    tokenizer = None
    text_encoder = None
    image_encoder = None

    extra_model_kwargs = EasyDict(
        {
            "noise_scheduler": noise_scheduler,
            "vae": vae,
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
            "image_encoder": image_encoder,
        }
    )

    logging.info(f"extra model initialized")

    return extra_model_kwargs


def dataloader_init(config, basic_kwargs, resume_step=0):
    if config.lrm.loss == 'ce':
        dataset = Image2VideoTrainDataset(
            dataset_type="lrm_ce",
            task=config.task,
            meta_file_list=config.dataset.meta_file_list,
            uncond_prob=config.dataset.uncond_prob,
            sp_size=config.dataset.sp_size,
            patch_size=config.model.patch_size
        )
    elif config.lrm.loss == 'bt':
        dataset = Image2VideoTrainDataset(
        dataset_type="lrm_bt_online",
        task=config.task,
        meta_file_list=config.dataset.meta_file_list,
        meta_file_lose_list=config.dataset.meta_file_lose_list,
        uncond_prob=config.dataset.uncond_prob,
        sp_size=config.dataset.sp_size,
        patch_size=config.model.patch_size
    )

    logging.info(f"dataset length {len(dataset)}")

    sampler = BlockDistributedSampler(
        dataset=dataset,
        num_replicas=basic_kwargs.world_size // nccl_info.sp_size,
        rank=nccl_info.group_id,
        shuffle=True,
        seed=config.train.seed,
        drop_last=True,
        batch_size=config.dataset.batch_size,
        start_index=resume_step
    )

    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        pin_memory=True,
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_workers,
        drop_last=True,
        worker_init_fn=set_worker_seed_builder(basic_kwargs.rank), 
        persistent_workers=False if config.dataset.num_workers == 0 else True
    )
    
    return VideoImageBatchIterator(video_dataloader=dataloader, sp_size=nccl_info.sp_size)

def optimizer_init(config, basic_kwargs, model_kwargs):
    transformer = model_kwargs.transformer
    mlp = model_kwargs.mlp
    query_attention = model_kwargs.query_attention
    
    transformer_params = []
    for name, param in transformer.named_parameters():
        if param.requires_grad:
            transformer_params.append(param)
            logging.info(f"adding trainable parameter: {name}")
    
    mlp_params = []
    for name, param in mlp.named_parameters():
        if param.requires_grad:
            mlp_params.append(param)
            logging.info(f"adding MLP parameter: {name}")
    
    query_attention_params = []
    for name, param in query_attention.named_parameters():
        if param.requires_grad:
            query_attention_params.append(param)
            logging.info(f"adding QueryAttention parameter: {name}")
    
    logging.info(f"transformer parameters: {len(transformer_params)}")
    logging.info(f"MLP parameters: {len(mlp_params)}")
    logging.info(f"QueryAttention parameters: {len(query_attention_params)}")
    
    if len(transformer_params) == 0:
        logging.warning("no trainable transformer parameters found!")
        param_groups = [
            {"params": mlp_params, "lr": config.optimizer.learning_rate}
        ]
    else:
        if hasattr(config.optimizer, 'learning_rate_mlp'):
            param_groups = [
                {"params": transformer_params, "lr": config.optimizer.learning_rate},
                {"params": mlp_params, "lr": config.optimizer.learning_rate_mlp}
            ]
        else:
            param_groups = [
                {"params": transformer_params, "lr": config.optimizer.learning_rate},
                {"params": mlp_params, "lr": config.optimizer.learning_rate}
            ]
    if 'q_attn' in config.lrm.pool:
        if hasattr(config.optimizer, 'learning_rate_mlp'):
            param_groups += [{"params": query_attention_params, "lr": config.optimizer.learning_rate_mlp}]
        else:
            param_groups += [{"params": query_attention_params, "lr": config.optimizer.learning_rate}]
    
    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(config.optimizer.adam_beta1, config.optimizer.adam_beta2),
        weight_decay=config.optimizer.weight_decay,
        eps=1e-8,
    )

    lr_scheduler = get_scheduler(
        config.optimizer.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.optimizer.lr_warmup_steps,
        num_training_steps=config.optimizer.max_train_steps,
        num_cycles=config.optimizer.lr_num_cycles,
        power=config.optimizer.lr_power,
    )

    optimizer_kwargs = EasyDict({"optimizer": optimizer, "lr_scheduler": lr_scheduler})
    logging.info("optimizer initialized")

    return optimizer_kwargs


def before_train_step(config, sp_dataloader, basic_kwargs, extra_model_kwargs):
    vae = extra_model_kwargs.vae
    text_encoder = extra_model_kwargs.text_encoder
    image_encoder = extra_model_kwargs.image_encoder

    if config.lrm.loss == 'ce':
        (
            latents,
            text_states,
            uncond_text_states,
            image_embeds,
            latents_condition,
            data_from_model,
            text_alignment,
            blur_quality,
            physics_quality,
            human_quality
        ) = next(sp_dataloader)
    elif config.lrm.loss == 'bt':
        (
            latents,
            text_states,
            uncond_text_states,
            image_embeds,
            latents_condition,
            latents_lose,
            text_states_lose,
            uncond_text_states_lose,
            image_embeds_lose,
            latents_condition_lose,
        ) = next(sp_dataloader)

    latents = latents.to(basic_kwargs.device, dtype=basic_kwargs.dtype)
    text_states = text_states.to(basic_kwargs.device, dtype=basic_kwargs.dtype)
    latents_condition = (
        latents_condition.to(basic_kwargs.device, dtype=basic_kwargs.dtype)
        if "i2v" in config.task or "flf2v" in config.task
        else None
    )
    if config.lrm.loss == 'ce':
        if config.lrm.task == "text_alignment":
            label = text_alignment
        elif config.lrm.task == "blur_quality":
            label = blur_quality
        elif config.lrm.task == "physics_quality":
            label = physics_quality
        elif config.lrm.task == "human_quality":
            label = human_quality
        elif config.lrm.task == "motion_quality":
            label = physics_quality and human_quality
        else:
            label = None
    
    if latents_condition is not None:
        b,c,f,h,w = latents_condition.shape
        mask_lat_size = torch.ones((b,4,f,h,w), dtype=basic_kwargs.dtype, device=basic_kwargs.device) 
        mask_lat_size[:,:,1:,...]=0.0
        if int(c)==16:
            latents_condition = torch.concat([mask_lat_size, latents_condition], dim=1)
    image_embeds = (
        image_embeds.to(basic_kwargs.device, dtype=basic_kwargs.dtype)
        if "i2v" in config.task or "flf2v" in config.task
        else None
    )
    if image_embeds is not None:
        N = image_embeds.shape[1] // 257
        image_embeds = rearrange(image_embeds, "b (n s) d -> (b n) s d", n=N)

    if config.lrm.loss == 'bt':
        latents_lose = latents_lose.to(basic_kwargs.device, dtype=basic_kwargs.dtype)
        text_states_lose = text_states_lose.to(basic_kwargs.device, dtype=basic_kwargs.dtype)
        latents_condition_lose = (
            latents_condition_lose.to(basic_kwargs.device, dtype=basic_kwargs.dtype)
            if "i2v" in config.task or "flf2v" in config.task
            else None
        )
        if latents_condition is not None:
            latents_condition_lose = torch.concat([mask_lat_size, latents_condition_lose], dim=1)
        image_embeds_lose = (
            image_embeds_lose.to(basic_kwargs.device, dtype=basic_kwargs.dtype)
            if "i2v" in config.task or "flf2v" in config.task
            else None
        )
        if image_embeds is not None:
            image_embeds_lose = rearrange(image_embeds_lose, "b (n s) d -> (b n) s d", n=N)
    
    if config.dataset.sp_size <= 1:
        latents, latents_condition = crop_tensor(
            latents,
            latents_condition,
            config.dataset.crop_ratio[0],
            config.dataset.crop_ratio[1],
            config.dataset.crop_type,
            crop_time_ratio=config.dataset.crop_ratio[2],
        )
        if config.lrm.loss == 'bt':
            latents_lose, latents_condition_lose = crop_tensor(
                latents_lose,
                latents_condition_lose,
                config.dataset.crop_ratio[0],
                config.dataset.crop_ratio[1],
                config.dataset.crop_type,
                crop_time_ratio=config.dataset.crop_ratio[2],
            )

    _, _, latents_t, latents_h, latents_w = latents.shape
    max_sequence_length = (
        latents_t
        * latents_h
        * latents_w
        // (config.model.patch_size[1] * config.model.patch_size[2])
    )

    if config.lrm.loss == 'ce':
        data_kwargs = EasyDict(
            {
                "latents": latents,
                "text_states": text_states,
                "image_embeds": image_embeds,
                "latents_condition": latents_condition,
                "max_sequence_length": max_sequence_length,
                "label": label,
            }
        )
    elif config.lrm.loss == 'bt':
        data_kwargs = EasyDict(
        {
            "latents": latents,
            "text_states": text_states,
            "image_embeds": image_embeds,
            "latents_condition": latents_condition,
            "latents_lose": latents_lose,
            "text_states_lose": text_states_lose,
            "image_embeds_lose": image_embeds_lose,
            "latents_condition_lose": latents_condition_lose,
            "max_sequence_length": max_sequence_length,
        }
    )

    return data_kwargs

def train_step(
    config,
    step,
    basic_kwargs,
    model_kwargs,
    extra_model_kwargs,
    optimizer_kwargs,
    data_kwargs,
):
    if step % 100 == 0:
        if not validate_model_parameters(model_kwargs.transformer, "Transformer"):
            logging.error("transformer has invalid parameters during training!")
            return {"loss": torch.tensor(0.0), "grad_norm": 0}
        
        if not validate_model_parameters(model_kwargs.mlp, "MLP"):
            logging.error("MLP has invalid parameters during training!")
            return {"loss": torch.tensor(0.0), "grad_norm": 0}
    
    # Model
    transformer = model_kwargs.transformer
    vae = extra_model_kwargs.vae
    noise_scheduler = extra_model_kwargs.noise_scheduler
    MLP = model_kwargs.mlp
    query_attention = model_kwargs.query_attention
    criterion = model_kwargs.criterion
    
    latents = data_kwargs.latents 
    text_states = data_kwargs.text_states
    latents_condition = data_kwargs.latents_condition
    image_embeds = data_kwargs.image_embeds 
    max_sequence_length = data_kwargs.max_sequence_length
    if config.lrm.loss == 'ce':
        label = data_kwargs.label
    elif config.lrm.loss == 'bt':
        latents_lose = data_kwargs.latents_lose
        text_states_lose = data_kwargs.text_states_lose
        image_embeds_lose = data_kwargs.image_embeds_lose
        latents_condition_lose = data_kwargs.latents_condition_lose
    
    # Optimizer
    optimizer = optimizer_kwargs.optimizer
    lr_scheduler = optimizer_kwargs.lr_scheduler

    # Forward
    bsz = latents.shape[0]
    noise = torch.randn_like(latents)

    transformer.train()
    MLP.train()
    
    if hasattr(config.lrm, 'timestep'):
        desired_timesteps = config.lrm.timestep
        selected_timestep_value = desired_timesteps[step % len(desired_timesteps)]
        timestep = torch.full((1,), selected_timestep_value, device=latents.device, dtype=torch.int64)
        sigma = noise_scheduler.get_train_sigma(
            timestep, 
            n_dim=latents.ndim, 
            device=latents.device, 
            dtype=latents.dtype
        )
    else:
        timestep, sigma = noise_scheduler.get_train_timestep_and_sigma(
            weighting_scheme=config.extra_model.scheduler.weighting_scheme,
            batch_size=bsz,
            logit_mean=config.extra_model.scheduler.logit_mean,
            logit_std=config.extra_model.scheduler.logit_std,
            device=latents.device,
            n_dim=latents.ndim,
        )

    # Sequence parallel broadcast
    if config.dataset.sp_size > 1:
        if "i2v" in config.task or "flf2v" in config.task: 
            broadcast(latents_condition)
            broadcast(image_embeds)
        broadcast(sigma)
        broadcast(noise)
        broadcast(timestep)
        broadcast(latents)
        broadcast(text_states)

        if config.lrm.loss == 'bt':
            if "i2v" in config.task or "flf2v" in config.task: 
                broadcast(image_embeds_lose)
                broadcast(latents_condition_lose)
            broadcast(latents_lose)
            broadcast(text_states_lose)
    
    noisy_latents = noise_scheduler.add_noise(latents, noise, sigma)
    cond_kwargs = {
        "x": batch2list(noisy_latents),
        "t": timestep,
        "context": batch2list(text_states),
        "seq_len": max_sequence_length,
        "clip_fea": image_embeds,
        "y": (
            batch2list(latents_condition)
            if "i2v" in config.task or "flf2v" in config.task
            else None
        ),
        "output_features": True,
        "selected_layers": model_kwargs.feature_layer,
    }

    if config.lrm.loss == 'bt':
        noisy_latents_lose = noise_scheduler.add_noise(latents_lose, noise, sigma)
        cond_kwargs_lose = {
            "x": batch2list(noisy_latents_lose),
            "t": timestep,
            "context": batch2list(text_states_lose),
            "seq_len": max_sequence_length,
            "clip_fea": image_embeds_lose,
            "y": (
                batch2list(latents_condition_lose)
                if "i2v" in config.task or "flf2v" in config.task
                else None
            ),
            "output_features": True,
            "selected_layers": model_kwargs.feature_layer,
        }
        
    with torch.autocast("cuda", dtype=basic_kwargs.dtype):
        model_pred = transformer(**cond_kwargs)
        model_pred = list2batch(model_pred)
        
        if config.dataset.sp_size > 1:
            if len(model_pred.shape) == 4:  # [sp_size, batch, seq_len_per_device, feature_dim]
                if config.lrm.pool == 'q_attn':
                    model_pred_final = query_attention(model_pred)
                elif config.lrm.pool == 'max':
                    model_pred_pooled, _ = model_pred.max(dim=2)
                    model_pred_final, _ = model_pred_pooled.max(dim=0)
                else:
                    model_pred_pooled = model_pred.mean(dim=2)
                    model_pred_final = model_pred_pooled.mean(dim=0)
        else:
            if len(model_pred.shape) == 3:  # [batch, seq_len, feature_dim]
                if config.lrm.pool == 'q_attn':
                    model_pred_final = query_attention(model_pred)
                elif config.lrm.pool == 'max':
                    model_pred_final, _ = model_pred.max(dim=1)  # [batch, feature_dim]
                else:
                    model_pred_final = model_pred.mean(dim=1)

        if config.lrm.loss == 'bt':
            model_pred_lose = transformer(**cond_kwargs_lose)
            model_pred_lose = list2batch(model_pred_lose)

            if config.dataset.sp_size > 1:
                if len(model_pred.shape) == 4:  # [sp_size, batch, seq_len_per_device, feature_dim]
                    if config.lrm.pool == 'q_attn':
                        model_pred_final_lose = query_attention(model_pred_lose)
                    elif config.lrm.pool == 'max':
                        model_pred_pooled_lose, _ = model_pred_lose.max(dim=2)
                        model_pred_final_lose, _ = model_pred_pooled_lose.max(dim=0)
                    else:
                        model_pred_pooled_lose = model_pred_lose.mean(dim=2)
                        model_pred_final_lose = model_pred_pooled_lose.mean(dim=0)
                else:
                    batch_size = model_pred.shape[0]
                    model_pred_final = model_pred.view(batch_size, -1).mean(dim=1, keepdim=True)
            else:
                if len(model_pred.shape) == 3:  # [batch, seq_len, feature_dim]
                    if config.lrm.pool == 'q_attn':
                        model_pred_final_lose = query_attention(model_pred_lose)
                    elif config.lrm.pool == 'max':
                        model_pred_final_lose, _ = model_pred_lose.max(dim=1) 
                    else:
                        model_pred_final_lose = model_pred_lose.mean(dim=1)
                else:
                    batch_size = model_pred.shape[0]
                    model_pred_final = model_pred.view(batch_size, -1).mean(dim=1, keepdim=True)
        
        if config.lrm.loss == 'ce':
            outputs = forward_mlp(MLP, model_pred_final)
            label = label.to(device=basic_kwargs.device, dtype=torch.float32)
        elif config.lrm.loss == 'bt':
            random_seed_wl_tensor = torch.rand(1, device=basic_kwargs.device)
            broadcast(random_seed_wl_tensor)
            random_seed_wl = random_seed_wl_tensor.item()
            if random_seed_wl < 0.5:
                outputs = forward_siamese(MLP, model_pred_final, model_pred_final_lose)
                label = torch.ones(bsz, dtype=torch.float32, device=basic_kwargs.device)
                sample_order = "win vs lose"
            else:
                outputs = forward_siamese(MLP, model_pred_final_lose, model_pred_final)
                label = torch.zeros(bsz, dtype=torch.float32, device=basic_kwargs.device)
                sample_order = "lose vs win"
            if step % 100 == 0:
                print(f"Step {step}: Sample order: {sample_order}, Label: {label[0].item()}")
    
    if label is not None:
        while label.dim() < outputs.dim():
            label = label.unsqueeze(-1)
        outputs_for_loss = outputs.squeeze().float()
        label_for_loss = label.squeeze().float()
        if config.dataset.sp_size > 1:
            broadcast(label_for_loss)
        loss = criterion(outputs_for_loss, label_for_loss)
    else:
        logging.warning("Warning: label is None, using dummy loss")
        loss = torch.tensor(0.0, requires_grad=True, device=basic_kwargs.device)
    
    if torch.isnan(loss) or torch.isinf(loss):
        logging.error("ERROR: Loss is NaN or Inf!")
        return {"loss": torch.tensor(0.0), "grad_norm": 0}
    
    if abs(loss.item()) > 1e6:
        logging.warning(f"WARNING: Loss value {loss.item()} is very large, clipping to 1e6")
        loss = torch.clamp(loss, -1e6, 1e6)
    
    try:
        transformer_params = [p for p in transformer.parameters() if p.requires_grad and p.grad is not None]
        if transformer_params:
            torch.nn.utils.clip_grad_norm_(transformer_params, max_norm=1.0)
        
        mlp_params = [p for p in MLP.parameters() if p.requires_grad and p.grad is not None]
        if mlp_params:
            torch.nn.utils.clip_grad_norm_(mlp_params, max_norm=1.0)
    except Exception as e:
        logging.error(f"ERROR during gradient clipping: {e}")
        return {"loss": torch.tensor(0.0), "grad_norm": 0}
    try:
        loss.backward()
    except Exception as e:
        logging.error(f"ERROR during backward: {e}")
        return {"loss": torch.tensor(0.0), "grad_norm": 0}
    
    grad_norm = transformer.clip_grad_norm_(max_norm=1.0)
    
    optimizer.step()
    optimizer.zero_grad()
    lr_scheduler.step()

    avg_loss = loss.detach().clone()
    dist.all_reduce(avg_loss, dist.ReduceOp.AVG)

    log_kwargs = EasyDict(
        {
            "loss": avg_loss,
            "grad_norm": grad_norm,
        }
    )
    if dist.get_rank() == 0:
        print(log_kwargs)
    
    dist.barrier()
    free_memory()

    return log_kwargs

def after_train_step(config, step, basic_kwargs, model_kwargs, log_kwargs, writer):
    transformer = model_kwargs.transformer
    ema_transformer = model_kwargs.ema_transformer
    mlp = model_kwargs.mlp
    query_attention = model_kwargs.query_attention

    log_loss = log_kwargs.loss
    log_grad_norm = log_kwargs.grad_norm
    log_step_time = log_kwargs.step_time
    log_avg_step_time = log_kwargs.avg_step_time
    log_lr = log_kwargs.lr

    if basic_kwargs.local_rank == 0:
        log_info = (
            f"│ Rank {basic_kwargs.rank:02d} │ Workers: {basic_kwargs.world_size} │"
            f"Step {step:05d} │ LR: {log_lr:.2e} │"
            f"Loss: {log_loss:.4f} │ Grad: {log_grad_norm:.4f} │"
            f"Time: {log_step_time:>6.2f}s │ Avg Time: {log_avg_step_time:>6.2f}s │ "
        )
    
    if basic_kwargs.rank == 0:
        if writer is not None:
            writer.add_scalar('train/loss', log_loss, step)
            writer.add_scalar('train/grad_norm', log_grad_norm, step)
            writer.add_scalar('train/lr', log_lr, step)
            writer.add_scalar('train/step_time', log_step_time, step)
            writer.add_scalar('train/avg_step_time', log_avg_step_time, step)
    
    if basic_kwargs.rank == 0:
        with open(basic_kwargs.log_path, "a", encoding="utf-8") as f:
            f.write(log_info + "\n")
        if not os.path.exists(config.save.mlp_dir):
            os.makedirs(config.save.mlp_dir)

    if config.model.ema.use_ema:
        dist.barrier()
        update_ema_model(transformer, ema_transformer, config.model.ema.ema_decay)

    if config.train.save_interval > 0 and step % config.train.save_interval == 0:
        dist.barrier()
        if config.model.lora.use_lora:
            save_lora_checkpoint(
                transformer,
                basic_kwargs.rank,
                config.save.ckpt_dir,
                step,
            )
            if config.model.ema.use_ema:
                save_lora_checkpoint(
                    ema_transformer,
                    basic_kwargs.rank,
                    config.save.ckpt_dir,
                    step,
                    ema=True,
                )
        else:
            save_checkpoint(
                transformer,
                basic_kwargs.rank,
                config.save.ckpt_dir,
                step,
            )
            if config.model.ema.use_ema:
                save_checkpoint(
                    ema_transformer,
                    basic_kwargs.rank,
                    config.save.ckpt_dir,
                    step,
                    ema=True,
                )

        if basic_kwargs.rank == 0:
            if not os.path.exists(config.save.mlp_dir):
                os.makedirs(config.save.mlp_dir, exist_ok=True)
            save_model(mlp, os.path.join(config.save.mlp_dir, f"mlp_step_{step}.ckpt"))
            if 'q_attn' in config.lrm.pool:
                save_model(query_attention, os.path.join(config.save.mlp_dir, f"query_attention_step_{step}.ckpt"))
        
        logging.info(f"save checkpoint saved at {step}")
        free_memory()

def evaluate_model(config, model_kwargs, extra_model_kwargs, basic_kwargs,log_kwargs, writer, step, t):
    val_dataset = Image2VideoTrainDataset(
        dataset_type="lrm_ce",
        task=config.task,
        meta_file_list=config.dataset.val_meta_file_list,
        uncond_prob=config.dataset.uncond_prob,
        sp_size=config.dataset.sp_size,
        patch_size=config.model.patch_size
    )
    logging.info(f"val dataset length {len(val_dataset)}")
    val_sampler = BlockDistributedSampler(
        val_dataset,
        num_replicas=basic_kwargs.world_size // nccl_info.sp_size,
        rank=nccl_info.group_id,
        shuffle=False,
        seed=config.train.seed,
        drop_last=False,
        batch_size=config.dataset.batch_size
    )
    val_dataloader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        batch_size=config.dataset.batch_size,
        num_workers=0,
        drop_last=False,
        worker_init_fn=set_worker_seed_builder(basic_kwargs.rank), 
        persistent_workers=False
    )

    transformer = model_kwargs.transformer
    MLP = model_kwargs.mlp
    query_attention = model_kwargs.query_attention
    noise_scheduler = extra_model_kwargs.noise_scheduler
    criterion = model_kwargs.criterion
    
    transformer.eval()
    MLP.eval()
    query_attention.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        dataloader_iter = tqdm(val_dataloader, desc="Evaluating", disable=(basic_kwargs.rank != 0))
        for batch in dataloader_iter:
            (
                latents,
                text_states,
                uncond_text_states,
                image_embeds,
                latents_condition,
                data_from_model,
                text_alignment,
                blur_quality,
                physics_quality,
                human_quality
            ) = batch

            latents = latents.to(basic_kwargs.device, dtype=basic_kwargs.dtype)
            text_states = text_states.to(basic_kwargs.device, dtype=basic_kwargs.dtype)
            
            latents_condition = (
                latents_condition.to(basic_kwargs.device, dtype=basic_kwargs.dtype)
                if "i2v" in config.task or "flf2v" in config.task
                else None
            )
            image_embeds = (
                image_embeds.to(basic_kwargs.device, dtype=basic_kwargs.dtype)
                if "i2v" in config.task or "flf2v" in config.task
                else None
            )
            
            if config.lrm.task == "text_alignment":
                label = text_alignment
            elif config.lrm.task == "blur_quality":
                label = blur_quality
            elif config.lrm.task == "physics_quality":
                label = physics_quality
            elif config.lrm.task == "human_quality":
                label = human_quality
            elif config.lrm.task == "motion_quality":
                label = physics_quality and human_quality
            else:
                label = None
                
            if label is not None:
                label = label.to(basic_kwargs.device, dtype=torch.float32)

            if latents_condition is not None and latents_condition.shape[1] == 16:
                b, _, f, h, w = latents_condition.shape
                mask_lat_size = torch.ones((b, 4, f, h, w), dtype=basic_kwargs.dtype, device=basic_kwargs.device) 
                mask_lat_size[:, :, 1:, ...] = 0.0
                latents_condition = torch.cat([mask_lat_size, latents_condition], dim=1)

            if image_embeds is not None:
                N = image_embeds.shape[1] // 257
                image_embeds = rearrange(image_embeds, "b (n s) d -> (b n) s d", n=N)
            
            if config.dataset.sp_size <= 1:
                latents, latents_condition = crop_tensor(
                    latents,
                    latents_condition,
                    config.dataset.crop_ratio[0],
                    config.dataset.crop_ratio[1],
                    config.dataset.crop_type,
                    crop_time_ratio=config.dataset.crop_ratio[2],
                )

            _, _, latents_t, latents_h, latents_w = latents.shape
            max_sequence_length = (
                latents_t
                * latents_h
                * latents_w
                // (config.model.patch_size[1] * config.model.patch_size[2])
            )

            seed_g = torch.Generator(device=latents.device)
            seed_g.manual_seed(config.eval.seed)
            bsz = latents.shape[0]
            noise = torch.randn(latents.shape, device=latents.device, dtype=latents.dtype, generator=seed_g)

            timestep = torch.full((1,), t, device=latents.device, dtype=torch.int64)
            sigma = noise_scheduler.get_train_sigma(
                timestep, 
                n_dim=latents.ndim, 
                device=latents.device, 
                dtype=latents.dtype
            )

            if config.dataset.sp_size > 1:
                if "i2v" in config.task or "flf2v" in config.task: 
                    broadcast(latents_condition)
                    broadcast(image_embeds)
                broadcast(sigma)
                broadcast(noise)
                broadcast(timestep)
                broadcast(latents)
                broadcast(text_states)
            
            noisy_latents = noise_scheduler.add_noise(latents, noise, sigma)

            cond_kwargs = {
                "x": batch2list(noisy_latents),
                "t": timestep,
                "context": batch2list(text_states),
                "seq_len": max_sequence_length,
                "clip_fea": image_embeds,
                "y": (
                    batch2list(latents_condition)
                    if "i2v" in config.task or "flf2v" in config.task
                    else None
                ),
                "output_features": True,
                "selected_layers": model_kwargs.feature_layer,
            }

            with torch.autocast("cuda", dtype=basic_kwargs.dtype):
                model_pred = transformer(**cond_kwargs)
                model_pred = list2batch(model_pred)
                
                if config.dataset.sp_size > 1:
                    if len(model_pred.shape) == 4:
                        if config.lrm.pool == 'q_attn':
                            model_pred_final = query_attention(model_pred)
                        elif config.lrm.pool == 'max':
                            model_pred_pooled, _ = model_pred.max(dim=2)
                            model_pred_final, _ = model_pred_pooled.max(dim=0)
                        else:
                            model_pred_pooled = model_pred.mean(dim=2)
                            model_pred_final = model_pred_pooled.mean(dim=0)
                    else:
                        batch_size = model_pred.shape[0]
                        model_pred_final = model_pred.view(batch_size, -1).mean(dim=1, keepdim=True)
                else:
                    if len(model_pred.shape) == 3:
                        if config.lrm.pool == 'q_attn':
                            model_pred_final = query_attention(model_pred)
                        elif config.lrm.pool == 'max':
                            model_pred_final, _ = model_pred.max(dim=1)
                        else:
                            model_pred_final = model_pred.mean(dim=1)
                    else:
                        batch_size = model_pred.shape[0]
                        model_pred_final = model_pred.view(batch_size, -1).mean(dim=1, keepdim=True)
                
                outputs = forward_mlp(MLP, model_pred_final)
            
            if label is not None:
                outputs_squeezed = outputs.squeeze()
                label_squeezed = label.squeeze()
                
                outputs_for_loss = outputs_squeezed.float()
                label_for_loss = label_squeezed.float()
                
                if config.dataset.sp_size > 1:
                    broadcast(label_for_loss)
                
                loss = criterion(outputs_for_loss, label_for_loss)
                total_loss += loss.item()
                
                predictions = (outputs_squeezed > 0.5).long()
                
                if predictions.ndim == 0:
                    predictions = predictions.unsqueeze(0)
                if label_for_loss.ndim == 0:
                    label_for_loss = label_for_loss.unsqueeze(0)
                
                pred_np = predictions.cpu().numpy()
                label_np = label_for_loss.long().cpu().numpy()
                
                if pred_np.ndim > 1:
                    pred_np = pred_np.flatten()
                if label_np.ndim > 1:
                    label_np = label_np.flatten()
                
                all_predictions.append(pred_np)
                all_labels.append(label_np)
            
            num_batches += 1

    total_loss_tensor = torch.tensor(total_loss, device=basic_kwargs.device, dtype=torch.float32)
    num_batches_tensor = torch.tensor(num_batches, device=basic_kwargs.device, dtype=torch.float32)
    all_predictions = torch.tensor(np.stack(all_predictions), device=basic_kwargs.device, dtype=torch.float32)
    all_labels = torch.tensor(np.stack(all_labels), device=basic_kwargs.device, dtype=torch.float32)

    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.SUM)
    all_predictions_list = [torch.zeros_like(all_predictions) for _ in range(basic_kwargs.world_size)]
    dist.all_gather(all_predictions_list, all_predictions)
    all_labels_list = [torch.zeros_like(all_labels) for _ in range(basic_kwargs.world_size)]
    dist.all_gather(all_labels_list, all_labels)

    avg_loss = total_loss_tensor.item() / num_batches_tensor.item() if num_batches_tensor.item() > 0 else 0
    all_preds = torch.concat(all_predictions_list).cpu().numpy()
    all_labs = torch.concat(all_labels_list).cpu().numpy()

    if len(all_predictions) > 0 and len(all_labels) > 0:
        try:
            accuracy = accuracy_score(all_labs, all_preds)
            precision = precision_score(all_labs, all_preds, zero_division=0)
            recall = recall_score(all_labs, all_preds, zero_division=0)
            f1 = f1_score(all_labs, all_preds, zero_division=0)
        except Exception as e:
            if basic_kwargs.rank == 0:
                logging.error(f"Error: {e}")
            accuracy = precision = recall = f1 = 0.0
    else:
        accuracy = precision = recall = f1 = 0.0
    
    if basic_kwargs.rank == 0:
        logging.info(f"✨ Evaluation - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Avg Loss: {avg_loss:.4f}")

        log_info = (
            f"│ Rank {basic_kwargs.rank:02d} │ Workers: {basic_kwargs.world_size} │"
            f"Timstep: {t} │"
            f"VAL Loss: {avg_loss:.4f} │"
            f"VAL Acc:{accuracy:.4f} │"
            f"VAL Prec:{precision:.4f} │"
            f"VAL Recall:{recall:.4f} │"
            f"VAL F1:{f1:.4f} │"
        )
        with open(basic_kwargs.log_path, "a", encoding="utf-8") as f:
            f.write(log_info + "\n")
    
        if writer is not None:
            writer.add_scalar(f'val/loss_{t}', avg_loss, step)
            writer.add_scalar(f'val/acc_{t}', accuracy, step)
            writer.add_scalar(f'val/precision_{t}', precision, step)
            writer.add_scalar(f'val/recall_{t}', recall, step)
            writer.add_scalar(f'val/f1_{t}', f1, step)

    transformer.train()
    MLP.train()
     
    return accuracy, avg_loss, precision, recall, f1
    
def main(config):
    config, basic_kwargs = basic_init(config)

    model_kwargs = model_init(config, basic_kwargs)
    extra_model_kwargs = extra_model_init(config, basic_kwargs)
    optimizer_kwargs = optimizer_init(config, basic_kwargs, model_kwargs)

    sp_dataloader = dataloader_init(config, basic_kwargs, model_kwargs.resume_step)

    dist.barrier()
    free_memory()

    writer = SummaryWriter(config.save.tensorboard_dir) if basic_kwargs.rank == 0 else None
    total_batch_size = (
        config.dataset.batch_size
        * (basic_kwargs.world_size // nccl_info.sp_size)
        * config.train.gradient_accumulation_steps
    )
    logging.info("***** Running training *****")
    logging.info(
        f"  Total train batch size (w. data & sequence parallel, accumulation) = {total_batch_size}"
    )
    logging.info(
        f"  Total training parameters per FSDP shard = {sum(p.numel() for p in model_kwargs['transformer'].parameters() if p.requires_grad) / 1e9} B"
    )

    step_times = deque(maxlen=100)

    for step in range(
        model_kwargs.resume_step + 1, config.optimizer.max_train_steps + 1
    ):
        start_time = time.time()

        data_kwargs = before_train_step(
            config, sp_dataloader, basic_kwargs, extra_model_kwargs
        )

        with torch.autograd.set_detect_anomaly(True):
            log_kwargs = train_step(
                config,
                step,
                basic_kwargs,
                model_kwargs,
                extra_model_kwargs,
                optimizer_kwargs,
                data_kwargs,
            )
        step_time = time.time() - start_time
        step_times.append(step_time)
        avg_step_time = sum(step_times) / len(step_times)

        log_kwargs.update(
            {
                "step_time": step_time,
                "avg_step_time": avg_step_time,
                "lr": optimizer_kwargs.optimizer.param_groups[0]["lr"],
            }
        )
        after_train_step(config, step, basic_kwargs, model_kwargs, log_kwargs, writer)

        if step % config.train.save_interval == 0:
            if hasattr(config.lrm, 'timestep'):
                for t in config.lrm.timestep:
                    evaluate_model(config, model_kwargs, extra_model_kwargs, basic_kwargs,log_kwargs, writer, step, t)
            elif hasattr(config.eval, 'timestep'):
                for t in config.eval.timestep:
                    try:
                        evaluate_model(config, model_kwargs, extra_model_kwargs, basic_kwargs,log_kwargs, writer, step, t)
                    except:
                        continue
            else:
                for t in [201, 400, 600, 800, 1000]:
                    evaluate_model(config, model_kwargs, extra_model_kwargs, basic_kwargs,log_kwargs, writer, step, t)

    if basic_kwargs.rank == 0 and writer is not None:
        writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        default="scripts/train/train_wanx.yaml",
    )
    args = parser.parse_args()

    main(OmegaConf.load(args.config_path))