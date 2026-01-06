import argparse
import json
import logging
import os
import time
import itertools
from copy import deepcopy
from collections import deque
from easydict import EasyDict


import torch
import torch.distributed as dist
import torch.nn.functional as F
from torchvision import transforms
import torch.nn as nn
import numpy as np

import torch.amp as amp
from diffusers.optimization import get_scheduler
from einops import rearrange
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from diffusers_lite.constants import PRECISION_TO_TYPE
from diffusers_lite.datasets.image2video_dataset import Image2VideoTrainDataset
from diffusers_lite.schedulers.scheduling_flow_match_discrete import (
    FlowMatchDiscreteScheduler,
)
from diffusers_lite.wan.utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from diffusers_lite.wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from diffusers_lite.wan.modules.model import WanModel
from diffusers_lite.wan.modules.t5 import T5EncoderModel
from diffusers_lite.wan.modules.vae import WanVAE
from diffusers_lite.wan.modules.clip import CLIPModel
from diffusers_lite.utils.communication import (
    broadcast,
    sp_parallel_dataloader_wrapper_wanx,
    all_gather,
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
    get_vae_fsdp_kwargs,
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
import random
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

NAME_MAPPING = {
    "t2v-1.3b": "Wan2.1-T2V-1.3B",
    "t2v-14b": "Wan2.1-T2V-14B",
    "i2v-1.3b": "Wan2.1-T2V-1.3B",
    "i2v-14b-480p": "Wan2.1-I2V-14B-480P",
    "i2v-14b-720p": "Wan2.1-I2V-14B-720P",
    "flf2v-14b-720p": "Wan2.1-FLF2V-14B-720P",
}

from transformers import AutoProcessor, AutoModel
from PIL import Image
from diffusers_lite.utils.network import MLP, QueryAttention, forward_siamese, forward_mlp, train_model, save_model
import gc
import torch

def log_memory_usage(step_name, rank=None):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
        rank_str = f"[Rank {rank}] " if rank is not None else ""
        print(f"{rank_str}{step_name}: Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Max: {max_allocated:.2f}GB")

def basic_init(config):
    # Init process groups
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dtype = PRECISION_TO_TYPE[config.train.precision]
    initialize_sequence_parallel_state(config.dataset.sp_size)
    set_logging(local_rank)

    # Init seed
    set_manual_seed(config.train.seed + nccl_info.group_id)
    logging.info(f"lanuch with seed {config.train.seed + rank}")

    # Init repository creation
    config.save.ckpt_dir = os.path.join(
        config.save.output_dir, f"{config.train_id}/checkpoints"
    )
    config.save.log_dir = os.path.join(
        config.save.output_dir, f"{config.train_id}/logs"
    )
    config.save.sanity_check_dir = f"outputs/sanity_check/wanx/{config.train_id}"
    config.save.tensorboard_dir = os.path.join(config.save.output_dir, f"{config.train_id}/tensorboard")

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

    torch.cuda.set_per_process_memory_fraction(0.95, device=basic_kwargs.device)
    torch.cuda.memory_pressure_threshold = 0.8
    
    os.environ["FSDP_FLATTEN_PARAMS"] = "1"
    os.environ["FSDP_SHARD_GRAD_PARAMS"] = "1"
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    return config, basic_kwargs


def model_init(config, basic_kwargs):
    assert config.task in NAME_MAPPING.keys()
    base_dir = config.model.base_path

    if config.model.resume_transformer_path:
        logging.info(f"loading model tranformer from {config.model.resume_transformer_path}")
        transformer = WanModel.from_pretrained(config.model.resume_transformer_path)
        resume_step = int(config.model.resume_transformer_path.split("-")[-1])
    elif config.model.init_transformer_path:
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

    # lrm transformer init
    lrm_transformer = WanModel.from_pretrained(config.model.base_path)

    frozen_modules = [
        'patch_embedding',
        'text_embedding',
        'time_embedding',
        'time_projection',
        'img_emb',
        # 'freqs',
    ]
    for module_name in frozen_modules:
        if hasattr(lrm_transformer, module_name):
            module = getattr(lrm_transformer, module_name)
            for param in module.parameters():
                param.requires_grad = False

    trainable_blocks = config.lrm.trainable_blocks

    if not hasattr(config.lrm, 'feature_layer'):
        config.lrm.feature_layer = [6, 7]
        logging.info(f"Setting default feature_layer to {config.lrm.feature_layer}")

    logging.info(f"Freezing all blocks except for {trainable_blocks}")

    new_blocks = []
    for i, block in enumerate(lrm_transformer.blocks):
        if i in trainable_blocks:
            logging.info(f"Block {i} is set to be trainable.")
            for param in block.parameters():
                param.requires_grad = True
            new_blocks.append(block)
        else:
            logging.info(f"Block {i} is frozen and removed.")

            for param in block.parameters():
                param.requires_grad = False

    lrm_transformer.blocks = nn.ModuleList(new_blocks)

    if hasattr(lrm_transformer, 'head'):
        del lrm_transformer.head
        lrm_transformer.head = None

    if hasattr(config.model, 'lrm_transformer_path') and config.model.lrm_transformer_path:
        logging.info(f"loading LRM transformer from {config.model.lrm_transformer_path}")
        state_dict = load_state_dict(config.model.lrm_transformer_path)
        lrm_transformer.load_state_dict(state_dict, strict=False)
    else:
        logging.info("No LRM transformer path specified, using base transformer")
    lrm_transformer.to(dtype=torch.float32)

    mlp_input_dim = config.lrm.mlp_dim
    mlp = MLP(mlp_input_dim)
    
    if hasattr(config.model, 'lrm_mlp_path') and config.model.lrm_mlp_path:
        logging.info(f"loading MLP from {config.model.lrm_mlp_path}")
        try:
            mlp.load_state_dict(torch.load(config.model.lrm_mlp_path))
            logging.info("Successfully loaded MLP from checkpoint")
        except:
            try:
                mlp.load_state_dict(torch.load(config.model.lrm_mlp_path)["state_dict"])
                logging.info("Successfully loaded MLP from checkpoint with state_dict key")
            except Exception as e:
                logging.error(f"Failed to load MLP from {config.model.lrm_mlp_path}: {e}")
                logging.info("Using newly created MLP due to loading failure")
    else:
        logging.info("No MLP path specified, using newly created MLP")
    
    mlp.to(basic_kwargs.device)
    mlp.eval()
    for param in mlp.parameters():
        param.requires_grad = False

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
    if hasattr(config.model, 'lrm_query_attention_path') and config.model.lrm_query_attention_path:
        logging.info(f"loading model query_attention from {config.model.lrm_query_attention_path}")
        checkpoint = torch.load(config.model.lrm_query_attention_path)
        query_attention.load_state_dict(checkpoint)
    query_attention = query_attention.to(device=basic_kwargs.device, dtype=torch.float32)
    query_attention.eval()

    transformer.__class__.enable_teacache = False
    lrm_transformer.__class__.enable_teacache = False

    # Init LoRA for transformer
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

    transformer = transformer.to(dtype=torch.float32)

    # Init EMA
    if config.model.ema.use_ema:
        logging.info("loading ema model")
        ema_transformer = deepcopy(transformer)

    else:
        ema_transformer = None

    # Init FSDP
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
    lrm_transformer = FSDP(lrm_transformer, **fsdp_kwargs)

    if config.model.ema.use_ema:
        ema_transformer = FSDP(ema_transformer, **fsdp_kwargs)

    # Init gradient checkpointing
    if config.model.gradient_checkpointing:
        apply_fsdp_checkpointing(
            transformer, no_split_modules, config.model.selective_checkpointing
        )
        apply_fsdp_checkpointing(
            lrm_transformer, no_split_modules, config.model.selective_checkpointing
        )
        if config.model.ema.use_ema:
            apply_fsdp_checkpointing(
                ema_transformer, no_split_modules, config.model.selective_checkpointing
            )
        logging.info("enable gradient checkpointing")

    # Set model as trainable
    transformer.train()
    print_parameters_information(transformer, "WAN", basic_kwargs.rank)

    if config.model.ema.use_ema:
        ema_transformer.requires_grad_(False)
        print_parameters_information(ema_transformer, "WAN EMA", basic_kwargs.rank)

    model_kwargs = EasyDict(
        {
            "transformer": transformer,
            "ema_transformer": ema_transformer,
            "resume_step": resume_step,
            "lrm_transformer": lrm_transformer,
            "query_attention": query_attention,
            "mlp": mlp,
        }
    )

    return model_kwargs


def extra_model_init(config, basic_kwargs):
    # base_dir = os.path.join(config.model.base_path, NAME_MAPPING["i2v-14b-480p"])
    base_dir = config.model.base_path
    # Init noise scheduler
    noise_scheduler = FlowMatchDiscreteScheduler(
        shift=config.extra_model.scheduler.flow_shift
    )
    noise_scheduler.set_timesteps(
        config.extra_model.scheduler.num_train_timesteps, dtype=torch.int64
    )
    noise_scheduler_refl =FlowUniPCMultistepScheduler(num_train_timesteps= config.extra_model.scheduler.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)

    vae = WanVAE(
            vae_pth=os.path.join(base_dir, config.extra_model.vae.name),
            dtype = basic_kwargs.dtype,
            device=basic_kwargs.device,
        )
    tokenizer = None
    text_encoder =None 
    image_encoder = None

    extra_model_kwargs = EasyDict(
        {
            "noise_scheduler": noise_scheduler,
            "noise_scheduler_refl":noise_scheduler_refl,
            "vae": vae,
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
            "image_encoder": image_encoder,
            "reward_model": None,
        }
    )

    logging.info(f"extra model initialized")

    return extra_model_kwargs


def dataloader_init(config, basic_kwargs, resume_step=0):
    dataset = Image2VideoTrainDataset(
        dataset_type="refl",
        task=config.task,
        meta_file_list=config.dataset.meta_file_list,
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

    params_to_optimize = transformer.parameters()
    params_to_optimize = list(filter(lambda p: p.requires_grad, params_to_optimize))

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=config.optimizer.learning_rate,
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
    # Model
    vae = extra_model_kwargs.vae
    text_encoder = extra_model_kwargs.text_encoder
    image_encoder = extra_model_kwargs.image_encoder

    if vae is not None:
        vae.model.requires_grad_(False)
        vae.model.eval()

    # Data
    (
        latents,
        text_states,
        uncond_text_states,
        image_embeds,
        latents_condition,
        long_caption
    ) = next(sp_dataloader)
    latents = latents.to(basic_kwargs.device, dtype=basic_kwargs.dtype)
    text_states = text_states.to(basic_kwargs.device, dtype=basic_kwargs.dtype)
    uncond_text_states =uncond_text_states.to(basic_kwargs.device, dtype=basic_kwargs.dtype)

    latents_condition = (
        latents_condition.to(basic_kwargs.device, dtype=basic_kwargs.dtype)
        if "i2v" in config.task or "flf2v" in config.task
        else None
    )
    
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
    
    data_kwargs = EasyDict(
        {
            "latents": latents,
            "text_states": text_states,
            "image_embeds": image_embeds,
            "latents_condition": latents_condition,
            "max_sequence_length": max_sequence_length,
            "uncond_text_states":uncond_text_states,
            "text_prompt": long_caption,
        }
    )

    return data_kwargs

def train_step_refl(
    config,
    step,
    basic_kwargs,
    model_kwargs,
    extra_model_kwargs,
    optimizer_kwargs,
    data_kwargs,
):
    log_memory_usage("Training step start", dist.get_rank() if hasattr(dist, 'get_rank') else None)
    
    transformer = model_kwargs.transformer
    transformer.gradient_checkpointing_enable() if hasattr(transformer, 'gradient_checkpointing_enable') else None
    lrm_transformer = model_kwargs.lrm_transformer
    query_attention = model_kwargs.query_attention
    mlp = model_kwargs.mlp
    vae = extra_model_kwargs.vae
    
    if vae is not None:
        if hasattr(vae.model, 'gradient_checkpointing_enable'):
            vae.model.gradient_checkpointing_enable()
            logging.info("Enabled gradient checkpointing for VAE model")
        else:
            if hasattr(vae.model, 'enable_gradient_checkpointing'):
                vae.model.enable_gradient_checkpointing()
                logging.info("Enabled gradient checkpointing for VAE model via alternative method")
            else:
                logging.warning("Gradient checkpointing not supported for VAE model")
    else:
        logging.info("VAE is None, skipping VAE-related operations")
    
    noise_scheduler = extra_model_kwargs.noise_scheduler_refl
    
    latents = data_kwargs.latents 
    text_states = data_kwargs.text_states
    latents_condition = data_kwargs.latents_condition
    image_embeds = data_kwargs.image_embeds 
    max_sequence_length = data_kwargs.max_sequence_length
    prompt = data_kwargs.text_prompt

    # Optimizer
    optimizer = optimizer_kwargs.optimizer
    lr_scheduler = optimizer_kwargs.lr_scheduler
    
    # Forward
    bsz = latents.shape[0]
    
    inference_steps = 40
    noise_scheduler.set_timesteps(num_inference_steps=inference_steps, device=basic_kwargs.device, shift=config.extra_model.scheduler.flow_shift)
    timesteps = noise_scheduler.timesteps
    transformer.eval()

    latent = torch.randn_like(latents)

    if basic_kwargs.rank == 0:
        mid_timestep = random.randint(0, inference_steps - 2)
    else:
        mid_timestep = 0
    
    del latents
    torch.cuda.empty_cache()
    gc.collect()
    
    log_memory_usage("After creating noise latents", dist.get_rank() if hasattr(dist, 'get_rank') else None)

    mid_timestep_tensor = torch.tensor(mid_timestep, device=latent.device, dtype=torch.long)
    dist.broadcast(mid_timestep_tensor, src=0)
    mid_timestep = mid_timestep_tensor.item()
    
    # 序列并行广播
    if config.dataset.sp_size > 1:
        if "i2v" in config.task or "flf2v" in config.task: 
            broadcast(latents_condition)
            broadcast(image_embeds)
        broadcast(latent)
        broadcast(text_states)
    
    log_memory_usage("After sequence parallel broadcast", dist.get_rank() if hasattr(dist, 'get_rank') else None)
    
    # ========== 1. infer with no grad to mid timestep ==========
    with torch.no_grad():
        for i in range(mid_timestep):
            t = timesteps[i]
            
            with torch.autocast("cuda", dtype=basic_kwargs.dtype):
                latent_model_input = latent
                timestep_tensor = torch.tensor([t], device=basic_kwargs.device)
                
                arg_c = {
                    "x": batch2list(latent_model_input),
                    "t": timestep_tensor,
                    "context": batch2list(text_states),
                    "seq_len": max_sequence_length,
                    "clip_fea": image_embeds,
                    "y": (
                        batch2list(latents_condition)
                        if "i2v" in config.task or "flf2v" in config.task
                        else None
                    ),
                    'cond_flag': True,
                }
            
                noise_pred = transformer(**arg_c)
                noise_pred = list2batch(noise_pred)

                scheduler_output = noise_scheduler.step(noise_pred, t, latent, return_dict=False)
                latent = scheduler_output[0] if isinstance(scheduler_output, tuple) else scheduler_output
                
                del latent_model_input, timestep_tensor, noise_pred, scheduler_output, arg_c
                torch.cuda.empty_cache()
                
                if i % 10 == 0:
                    gc.collect()
                    dist.barrier()
                    log_memory_usage(f"After inference step {i}", dist.get_rank() if hasattr(dist, 'get_rank') else None)
    
    log_memory_usage("After inference loop", dist.get_rank() if hasattr(dist, 'get_rank') else None)
    
    # ========== 2. cal gradient ==========
    transformer.train()
    
    t_mid = timesteps[mid_timestep]
    timestep_mid = torch.tensor([t_mid], device=basic_kwargs.device)
    
    arg_c = {
                "x": batch2list(latent),
                "t": timestep_mid,
                "context": batch2list(text_states),
                "seq_len": max_sequence_length,
                "clip_fea": image_embeds,
                "y": (
                    batch2list(latents_condition)
                    if "i2v" in config.task or "flf2v" in config.task
                    else None
                ),
                'cond_flag': True,
            }
    
    with torch.autocast("cuda", dtype=basic_kwargs.dtype, enabled=True):
        noise_pred = transformer(**arg_c)
        noise_pred = list2batch(noise_pred)
    
    del timestep_mid, arg_c
    torch.cuda.empty_cache()
    gc.collect()
    
    log_memory_usage("After gradient computation", dist.get_rank() if hasattr(dist, 'get_rank') else None)

    # ========== 3. cal pred_original_sample ==========   
    scheduler_output = noise_scheduler.step(noise_pred, t_mid, latent,return_dict=False)
    latent = scheduler_output[0] if isinstance(scheduler_output, tuple) else scheduler_output
    
    del scheduler_output
    torch.cuda.empty_cache()
    gc.collect()
    dist.barrier()
    
    log_memory_usage("After pred_original_sample computation", dist.get_rank() if hasattr(dist, 'get_rank') else None)
    
    # ========== 4. cal reward ==========
    t_mid_1 = timesteps[mid_timestep+1]
    timestep_mid_1 = torch.tensor([t_mid_1], device=basic_kwargs.device)
    
    with torch.autocast("cuda", dtype=basic_kwargs.dtype, enabled=True):
        lrm_cond_kwargs = {
            "x": batch2list(latent),
            "t": timestep_mid_1,
            "context": batch2list(text_states),
            "seq_len": max_sequence_length,
            "clip_fea": image_embeds,
            "y": (
                batch2list(latents_condition)
                if "i2v" in config.task or "flf2v" in config.task
                else None
            ),
            "output_features": True,
            "selected_layers": config.lrm.feature_layer,
        }
        
        lrm_features = lrm_transformer(**lrm_cond_kwargs)
        lrm_features = list2batch(lrm_features)
        
        if config.dataset.sp_size > 1:
            if len(lrm_features.shape) == 4:  # [sp_size, batch, seq_len_per_device, feature_dim]
                if config.lrm.pool == 'q_attn':
                    lrm_features_final = query_attention(lrm_features)
                else:
                    lrm_features_pooled = lrm_features.mean(dim=2)  # [sp_size, batch, feature_dim]
                    lrm_features_final = lrm_features_pooled.mean(dim=0)  # [batch, feature_dim]
            else:
                original_batch_size = bsz
                lrm_features_flat = lrm_features.view(original_batch_size, -1)
                lrm_features_final = lrm_features_flat.mean(dim=1, keepdim=True)  # [batch, 1]
        else:
            if len(lrm_features.shape) == 3:  # [batch, seq_len, feature_dim]
                if config.lrm.pool == 'q_attn':
                    lrm_features_final = query_attention(lrm_features)
                else:
                    lrm_features_final = lrm_features.mean(dim=1)  # [batch, feature_dim]
            elif len(lrm_features.shape) == 4:  # [batch, channels, seq_len, feature_dim] or similar
                if config.lrm.pool == 'q_attn':
                    lrm_features_final = query_attention(lrm_features)
                else:
                    lrm_features_pooled = lrm_features.mean(dim=2)  # [batch, feature_dim]
                    lrm_features_final = lrm_features_pooled.mean(dim=1)  # [batch, feature_dim]
            elif len(lrm_features.shape) == 2:  # [batch, feature_dim] - already good
                lrm_features_final = lrm_features
            else:
                batch_size = lrm_features.shape[0]
                lrm_features_final = lrm_features.view(batch_size, -1).mean(dim=1, keepdim=True)  # [batch, 1]
        
        reward_scores = forward_mlp(mlp, lrm_features_final)
        target_reward = 2
        loss = 0.1 * F.relu(-reward_scores.squeeze() + target_reward).mean()
        
        # 检查损失值是否有效
        if torch.isnan(loss) or torch.isinf(loss):
            print("ERROR: Loss is NaN or Inf!")
            del lrm_features, lrm_features_final, reward_scores, lrm_cond_kwargs, timestep_mid_1, t_mid_1
            del image_embeds, text_states, latents_condition, noise_pred, latent
            torch.cuda.empty_cache()
            gc.collect()
            return {"loss": torch.tensor(0.0), "grad_norm": 0}
        
        if abs(loss.item()) > 1e6:
            print(f"WARNING: Loss value {loss.item()} is very large, clipping to 1e6")
            loss = torch.clamp(loss, -1e6, 1e6)
        
        del lrm_features, lrm_features_final, reward_scores, lrm_cond_kwargs, timestep_mid_1, t_mid_1, image_embeds, text_states, latents_condition
        torch.cuda.empty_cache()
        gc.collect()
        dist.barrier()
    
    log_memory_usage("After LRM computation", dist.get_rank() if hasattr(dist, 'get_rank') else None)

    # ========== 5. backwards ==========
    try:
        loss /= config.train.gradient_accumulation_steps
        loss.backward()

        grad_norm = transformer.clip_grad_norm_(max_norm=1.0)
        
        if (step + 1) % config.train.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
        
    except Exception as e:
        print(f"ERROR during backward/optimization: {e}")
        del latent
        torch.cuda.empty_cache()
        gc.collect()
        return {"loss": torch.tensor(0.0), "grad_norm": 0}

    torch.cuda.empty_cache()
    gc.collect()
    dist.barrier()
    
    log_memory_usage("After optimization", dist.get_rank() if hasattr(dist, 'get_rank') else None)

    avg_loss = loss.detach().clone()
    dist.all_reduce(avg_loss, dist.ReduceOp.AVG)

    # Logs results
    if (
        config.train.sanity_check_interval >= 0 and step <= 50 # and step % config.train.sanity_check_interval == 0
    ):
        if basic_kwargs.rank == 0:
            with torch.no_grad():
                sigma_t = noise_scheduler.sigmas[mid_timestep+1]

                pred_original_sample = latent - sigma_t * noise_pred
                pred_x0_s = vae_decode(
                    vae, pred_original_sample.clone().detach(), dtype=basic_kwargs.dtype, vae_type="wanx"
                )
                latents_s = vae_decode(
                    vae, latent.clone(), dtype=basic_kwargs.dtype, vae_type="wanx"
                )
                print("save_videos_grid:",os.path.join(
                        config.save.sanity_check_dir,
                        f"step{step}_pred_x0_rank{basic_kwargs.rank}_{sigma_t.item()}.mp4",
                    ))
                save_videos_grid(
                    pred_x0_s.to(torch.float32).cpu(),
                    os.path.join(
                        config.save.sanity_check_dir,
                        f"step{step}_pred_x0_rank{basic_kwargs.rank}_{sigma_t.item()}.mp4",
                    ),
                    fps=15,
                    rescale=True,
                )
                save_videos_grid(
                    latents_s.to(torch.float32).cpu(),
                    os.path.join(
                        config.save.sanity_check_dir,
                        f"step{step}_real_x0_rank{basic_kwargs.rank}.mp4",
                    ),
                    fps=15,
                    rescale=True,
                )
            del pred_original_sample, sigma_t, latents_s, pred_x0_s
            torch.cuda.empty_cache()
            gc.collect()

    log_kwargs = EasyDict({
        "loss": avg_loss,
        "grad_norm": grad_norm,
    })
    del latent,noise_pred
    dist.barrier()
    free_memory()
    
    log_memory_usage("Training step end", dist.get_rank() if hasattr(dist, 'get_rank') else None)
    return log_kwargs

def train_step(
    config,
    step,
    basic_kwargs,
    model_kwargs,
    extra_model_kwargs,
    optimizer_kwargs,
    data_kwargs,
):
    # Model
    transformer = model_kwargs.transformer
    vae = extra_model_kwargs.vae
    noise_scheduler = extra_model_kwargs.noise_scheduler
    
    latents = data_kwargs.latents 
    text_states = data_kwargs.text_states
    latents_condition = data_kwargs.latents_condition
    image_embeds = data_kwargs.image_embeds 
    max_sequence_length = data_kwargs.max_sequence_length

    # Optimizer
    optimizer = optimizer_kwargs.optimizer
    lr_scheduler = optimizer_kwargs.lr_scheduler

    # Forward
    bsz = latents.shape[0]
    noise = torch.randn_like(latents)

    timestep, sigma = noise_scheduler.get_train_timestep_and_sigma(
        weighting_scheme=config.extra_model.scheduler.weighting_scheme,
        batch_size=bsz,
        logit_mean=config.extra_model.scheduler.logit_mean,
        logit_std=config.extra_model.scheduler.logit_std, 
        device=latents.device,
        n_dim=latents.ndim,
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
    }
    with torch.autocast("cuda", dtype=basic_kwargs.dtype):
        model_pred = transformer(**cond_kwargs)
        model_pred = list2batch(model_pred)

    training_target = noise_scheduler.get_train_target(latents, noise)
    weighting = noise_scheduler.get_train_loss_weighting(sigma)

    loss = torch.mean(
        weighting.float() * (model_pred.float() - training_target.float()) ** 2
    )
    loss /= config.train.gradient_accumulation_steps
    loss.backward()
    grad_norm = transformer.clip_grad_norm_(max_norm=1.0)

    if (step + 1) % config.train.gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

    avg_loss = loss.detach().clone()
    dist.all_reduce(avg_loss, dist.ReduceOp.AVG)

    # Compute loss
    log_kwargs = EasyDict(
        {
            "loss": avg_loss,
            "grad_norm": grad_norm,
        }
    )
    del sigma, noise,timestep,latents,text_states, latents_condition, image_embeds,loss,training_target,weighting,model_pred
    torch.cuda.empty_cache()
    if dist.get_rank() == 0:
        print(log_kwargs)

    if (
        config.train.sanity_check_interval > 0
        and step % config.train.sanity_check_interval == 0
        and step <= 50
    ):
        if basic_kwargs.rank == 0:
            pred_x0 = noise_scheduler.get_x0(model_pred, noisy_latents, sigma)
            pred_x0 = pred_x0.to(dtype=basic_kwargs.dtype)
            pred_x0_s = vae_decode(
                vae, pred_x0.clone().detach(), dtype=basic_kwargs.dtype, vae_type="wanx"
            )
            latents_s = vae_decode(
                vae, latents.clone(), dtype=basic_kwargs.dtype, vae_type="wanx"
            )
            print("save_videos_grid_path:",os.path.join(
                    config.save.sanity_check_dir,
                    f"step{step}_pred_x0_rank{basic_kwargs.rank}_{sigma.item()}.mp4",
                ))

            save_videos_grid(
                pred_x0_s.to(torch.float32).cpu(),
                os.path.join(
                    config.save.sanity_check_dir,
                    f"step{step}_pred_x0_rank{basic_kwargs.rank}_{sigma.item()}.mp4",
                ),
                fps=15,
                rescale=True,
            )
            save_videos_grid(
                latents_s.to(torch.float32).cpu(),
                os.path.join(
                    config.save.sanity_check_dir,
                    f"step{step}_real_x0_rank{basic_kwargs.rank}.mp4",
                ),
                fps=15,
                rescale=True,
            )
    dist.barrier()
    free_memory()

    return log_kwargs

def after_train_step(config, step, basic_kwargs, model_kwargs, 
                    log_kwargs_normal, log_kwargs_reward, writer):
    transformer = model_kwargs.transformer
    ema_transformer = model_kwargs.ema_transformer

    log_loss_normal = log_kwargs_normal.loss
    log_grad_norm_normal = log_kwargs_normal.grad_norm
    log_step_time_normal = log_kwargs_normal.step_time
    log_avg_step_time_normal = log_kwargs_normal.avg_step_time
    log_lr = log_kwargs_normal.lr

    log_loss_reward = log_kwargs_reward.loss
    log_grad_norm_reward = log_kwargs_reward.grad_norm
    log_step_time_reward = log_kwargs_reward.step_time
    log_avg_step_time_reward = log_kwargs_reward.avg_step_time

    if basic_kwargs.local_rank == 0:
        log_info = (
            f"│ Rank {basic_kwargs.rank:02d} │ Workers: {basic_kwargs.world_size} │ "
            f"Step {step:05d} │ LR: {log_lr:.2e} │\n"
            f"│ Normal - Loss: {log_loss_normal:.4f} │ Grad: {log_grad_norm_normal:.4f} │ "
            f"Time: {log_step_time_normal:>6.2f}s │ Avg: {log_avg_step_time_normal:>6.2f}s │\n"
            f"│ Reward - Loss: {log_loss_reward:.4f} │ Grad: {log_grad_norm_reward:.4f} │ "
            f"Time: {log_step_time_reward:>6.2f}s │ Avg: {log_avg_step_time_reward:>6.2f}s │"
        )
        print(log_info)
    
    if basic_kwargs.rank == 0 and writer is not None:
        writer.add_scalar('train/normal_loss', log_loss_normal, step)
        writer.add_scalar('train/normal_grad_norm', log_grad_norm_normal, step)
        writer.add_scalar('train/normal_step_time', log_step_time_normal, step)
        writer.add_scalar('train/normal_avg_step_time', log_avg_step_time_normal, step)
        writer.add_scalar('train/reward_loss', log_loss_reward, step)
        writer.add_scalar('train/reward_grad_norm', log_grad_norm_reward, step)
        writer.add_scalar('train/reward_step_time', log_step_time_reward, step)
        writer.add_scalar('train/reward_avg_step_time', log_avg_step_time_reward, step)
        writer.add_scalar('train/lr', log_lr, step)
        
        total_loss = log_loss_normal + log_loss_reward
        total_time = log_step_time_normal + log_step_time_reward
        writer.add_scalar('train/total_loss', total_loss, step)
        writer.add_scalar('train/total_step_time', total_time, step)

    if basic_kwargs.rank == 0:
        with open(basic_kwargs.log_path, "a", encoding="utf-8") as f:
            f.write(log_info + "\n")

    if config.model.ema.use_ema:
        dist.barrier()
        update_ema_model(transformer, ema_transformer, config.model.ema.ema_decay)

    if config.train.save_interval > 0 and step % config.train.save_interval == 0:
        dist.barrier()
        if config.model.lora.use_lora:
            save_lora_checkpoint(transformer, basic_kwargs.rank, config.save.ckpt_dir, step)
            if config.model.ema.use_ema:
                save_lora_checkpoint(ema_transformer, basic_kwargs.rank, 
                                   config.save.ckpt_dir, step, ema=True)
        else:
            save_checkpoint(transformer, basic_kwargs.rank, config.save.ckpt_dir, step)
            if config.model.ema.use_ema:
                save_checkpoint(ema_transformer, basic_kwargs.rank, 
                              config.save.ckpt_dir, step, ema=True)
        logging.info(f"Checkpoint saved at step {step}")
        free_memory()

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
    step_times_2 = deque(maxlen=100)

    for step in range(
        model_kwargs.resume_step + 1, config.optimizer.max_train_steps + 1
    ):
        start_time = time.time()

        data_kwargs = before_train_step(
            config, sp_dataloader, basic_kwargs, extra_model_kwargs
        )

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

        start_time = time.time()

        log_kwargs2 = train_step_refl(
            config,
            step,
            basic_kwargs,
            model_kwargs,
            extra_model_kwargs,
            optimizer_kwargs,
            data_kwargs,
        )

        step_time_2 = time.time() - start_time
        step_times_2.append(step_time_2)
        avg_step_time_2 = sum(step_times_2) / len(step_times_2)

        log_kwargs2.update(
            {
                "step_time": step_time_2,
                "avg_step_time": avg_step_time_2,
                "lr": optimizer_kwargs.optimizer.param_groups[0]["lr"],
            }
        )

        after_train_step(config, step, basic_kwargs, model_kwargs, log_kwargs, log_kwargs2, writer)

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