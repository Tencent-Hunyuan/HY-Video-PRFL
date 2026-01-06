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
import random

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
    loss_type = config.lrm.loss
    
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
            "loss_type": loss_type,
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

def evaluate_model(config, model_kwargs, extra_model_kwargs, basic_kwargs, log_kwargs, writer, step, bucket_timesteps):
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
            
            t = bucket_timesteps[0]
            t_sample = random.choice(bucket_timesteps)
            timestep = torch.full((1,), t_sample, device=latents.device, dtype=torch.int64)
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
                        original_batch_size = bsz
                        model_pred_flat = model_pred.view(original_batch_size, -1)
                        model_pred_final = model_pred_flat.mean(dim=1, keepdim=True)
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
            logging.error(f"Error: {e}")
            accuracy = precision = recall = f1 = 0.0
    else:
        accuracy = precision = recall = f1 = 0.0
    
    if basic_kwargs.rank == 0:
        logging.info(f"✨ Evaluation - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Avg Loss: {avg_loss:.4f}")

        log_info = (
            f"│ Rank {basic_kwargs.rank:02d} │ Workers: {basic_kwargs.world_size} │"
            f"CKPT Step: {step} │"
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

    transformer.eval()
    MLP.eval()
     
    return accuracy, avg_loss, precision, recall, f1
    
def main(config):
    config, basic_kwargs = basic_init(config)

    model_kwargs = model_init(config, basic_kwargs)
    extra_model_kwargs = extra_model_init(config, basic_kwargs)

    dist.barrier()
    free_memory()

    writer = SummaryWriter(config.save.tensorboard_dir) if basic_kwargs.rank == 0 else None
    total_batch_size = (
        config.dataset.batch_size
        * (basic_kwargs.world_size // nccl_info.sp_size)
        * config.train.gradient_accumulation_steps
    )
    logging.info("***** Running evaluation *****")

    bucket_intervals = [(0, 200), (201, 400), (401, 600), (601,800), (801, 1000)]
    for i, (start_bound, end_bound) in enumerate(bucket_intervals):
        bucket_timesteps = []
        for t in extra_model_kwargs.noise_scheduler.timesteps:
            if t >= start_bound and t <= end_bound:
                bucket_timesteps.append(t)
        try:
            random.seed(config.eval.seed)
            evaluate_model(config, model_kwargs, extra_model_kwargs, basic_kwargs, EasyDict(), writer, model_kwargs.resume_step, bucket_timesteps)
        except:
            continue

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