# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import logging
import os
import sys
import warnings

warnings.filterwarnings('ignore')

import torch
import torch.distributed as dist
from easydict import EasyDict
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from diffusers_lite import wan
from diffusers_lite.wan.configs import WAN_CONFIGS, MAX_AREA_CONFIGS, SIZE_CONFIGS
from diffusers_lite.wan.utils.utils import cache_video
from diffusers_lite.arguments import args_wan_init
from diffusers_lite.datasets.image2video_dataset import Image2VideoEvalDataset


def _init_logging(rank):
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def basic_init(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if rank == 0:
        os.makedirs(args.save_folder, exist_ok=True)
    logging.info(f"Creating save directory: {args.save_folder}")

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
        
    if args.ulysses_size == 1 and args.ring_size == 1:
        args.ddp_mode = True
        # args.t5_fsdp = False
        # args.dit_fsdp = False
        logging.info(f"DDP mode enabled.")

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1 or args.ring_size > 1
        ), f"context parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_size, f"The number of ulysses_size and ring_size should be equal to the world size."
        from xfuser.core.distributed import (initialize_model_parallel,
                                             init_distributed_environment)
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )
    

    cfg = WAN_CONFIGS[args.task]
    
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`num_heads` must be divisible by `ulysses_size`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    

    basic_kwargs = EasyDict({
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": device,
        "cfg": cfg,
    })
    return basic_kwargs


def dataset_init(args, basic_kwargs):
    dataset = Image2VideoEvalDataset(
        args.dataset_path,
        do_scale=True,
        resolution=SIZE_CONFIGS[args.size]
    )
    logging.info(f"Dataset length: {len(dataset)}")
    
    if args.ddp_mode:
        sampler = DistributedSampler(
            dataset,
            num_replicas=basic_kwargs.world_size,
            rank=basic_kwargs.rank,
            shuffle=False,
            drop_last=False,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sampler,
            drop_last=False
        )
        dataset = dataloader
    
    return dataset


def pipeline_t2v_init(args, basic_kwargs):
    logging.info("Creating WanT2V pipeline.")
    wan_t2v = wan.WanT2V(
        config=basic_kwargs.cfg,
        checkpoint_dir=args.ckpt_dir,
        transformer_path=args.transformer_path,
        lora_path=args.lora_path,
        lora_alpha=args.lora_alpha,
        distill_lora_path=args.distill_lora_path,
        distill_lora_alpha=args.distill_lora_alpha,
        device_id=basic_kwargs.device,
        rank=basic_kwargs.rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
        t5_cpu=args.t5_cpu,
        teacache_thresh=args.teacache_thresh,
        sample_steps=args.sample_steps,
        ckpt_dir=args.ckpt_dir,
    )

    return wan_t2v


def pipeline_i2v_init(args, basic_kwargs):
    logging.info("Creating WanI2V pipeline.")
    wan_i2v = wan.WanI2V(
        config=basic_kwargs.cfg,
        checkpoint_dir=args.ckpt_dir,
        transformer_path=args.transformer_path,
        lora_path=args.lora_path,
        lora_alpha=args.lora_alpha,
        distill_lora_path=args.distill_lora_path,
        distill_lora_alpha=args.distill_lora_alpha,
        device_id=basic_kwargs.device,
        rank=basic_kwargs.rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
        t5_cpu=args.t5_cpu,
        teacache_thresh=args.teacache_thresh,
        sample_steps=args.sample_steps,
        ckpt_dir=args.ckpt_dir,
    )

    return wan_i2v


def pipeline_flf2v_init(args, basic_kwargs):
    logging.info("Creating WanFLF2V pipeline.")
    wan_flf2v = wan.WanFLF2V(
        config=basic_kwargs.cfg,
        checkpoint_dir=args.ckpt_dir,
        transformer_path=args.transformer_path,
        lora_path=args.lora_path,
        lora_alpha=args.lora_alpha,
        distill_lora_path=args.distill_lora_path,
        distill_lora_alpha=args.distill_lora_alpha,
        device_id=basic_kwargs.device,
        rank=basic_kwargs.rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
        t5_cpu=args.t5_cpu,
        teacache_thresh=args.teacache_thresh,
        sample_steps=args.sample_steps,
        ckpt_dir=args.ckpt_dir,
    )

    return wan_flf2v


def inference_t2v_loop(args, pipeline, batch):
    if args.ddp_mode:
        prompt = batch["prompt"][0]
        image_id = batch["image_id"][0]
    else:
        prompt = batch["prompt"]
        image_id = batch["image_id"]
    # image_id = prompt[:200]

    info_str = f"""
            height: {args.resolution[1]}
             width: {args.resolution[0]}
      video_length: {args.frame_num}
            prompt: {prompt}
        neg_prompt: {args.negative_prompt}
              seed: {int(batch["seed"])}
       infer_steps: {args.sample_steps}
    guidance_scale: {args.sample_guide_scale}
        flow_shift: {args.sample_shift}"""
    logging.info(info_str)

    video = pipeline.generate(
        prompt,
        n_prompt=args.negative_prompt,
        size=args.resolution,
        frame_num=args.frame_num,
        shift=args.sample_shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.sample_steps,
        guide_scale=args.sample_guide_scale,
        seed=int(batch["seed"]),
        # seed=args.base_seed,
        offload_model=args.offload_model,
        ddp_mode=args.ddp_mode,
    )

    return video, image_id


def inference_i2v_loop(args, pipeline, batch):
    if args.ddp_mode:
        prompt = batch["prompt"][0]
        image_id = batch["image_id"][0]
        cond_image = transforms.ToPILImage()(batch["image"][0])
    else:
        prompt = batch["prompt"]
        image_id = batch["image_id"]
        cond_image = transforms.ToPILImage()(batch["image"])

    width, height = cond_image.size[0], cond_image.size[1]

    info_str = f"""
            height: {height}
             width: {width}
      current_araa: {height} * {width}
          max_area: {MAX_AREA_CONFIGS[args.size]}
      video_length: {args.frame_num}
            prompt: {prompt}
        neg_prompt: {args.negative_prompt}
              seed: {int(batch["seed"])}
       infer_steps: {args.sample_steps}
    guidance_scale: {args.sample_guide_scale}
        flow_shift: {args.sample_shift}"""
    logging.info(info_str)

    video = pipeline.generate(
        prompt,
        cond_image,
        n_prompt=args.negative_prompt,
        max_area=MAX_AREA_CONFIGS[args.size],
        frame_num=args.frame_num,
        shift=args.sample_shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.sample_steps,
        guide_scale=args.sample_guide_scale,
        # seed=args.base_seed,
        seed=int(batch["seed"]),
        offload_model=args.offload_model,
        ddp_mode=args.ddp_mode,
    )

    return video, image_id


def inference_flf2v_loop(args, pipeline, batch):
    if args.ddp_mode:
        prompt = batch["prompt"][0]
        image_id = batch["image_id"][0]
        cond_image = transforms.ToPILImage()(batch["image"][0])
        last_image = transforms.ToPILImage()(batch["last_image"][0])
    else:
        prompt = batch["prompt"]
        image_id = batch["image_id"]
        cond_image = transforms.ToPILImage()(batch["image"])
        last_image = transforms.ToPILImage()(batch["last_image"])
    width, height = cond_image.size[0], cond_image.size[1]

    info_str = f"""
            height: {height}
             width: {width}
          max_area: {MAX_AREA_CONFIGS[args.size]}
      video_length: {args.frame_num}
            prompt: {prompt}
        neg_prompt: {args.negative_prompt}
              seed: {args.base_seed}
       infer_steps: {args.sample_steps}
    guidance_scale: {args.sample_guide_scale}
        flow_shift: {args.sample_shift}"""
    logging.info(info_str)

    video = pipeline.generate(
        prompt,
        cond_image,
        last_image,
        n_prompt=args.negative_prompt,
        max_area=MAX_AREA_CONFIGS[args.size],
        frame_num=args.frame_num,
        shift=args.sample_shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.sample_steps,
        guide_scale=args.sample_guide_scale,
        seed=args.base_seed,
        offload_model=args.offload_model,
        ddp_mode=args.ddp_mode,
    )

    return video, image_id


def main(args):

    basic_kwargs = basic_init(args)
    dataset = dataset_init(args, basic_kwargs)

    if "t2v" in args.task:
        pipeline = pipeline_t2v_init(args, basic_kwargs)
    elif "i2v" in args.task:
        pipeline = pipeline_i2v_init(args, basic_kwargs)
    elif "flf2v" in args.task:
        pipeline = pipeline_flf2v_init(args, basic_kwargs)

    for i, batch in enumerate(dataset):
        image_id = batch["image_id"][0]
        save_path = os.path.join(args.save_folder, f"{image_id}.mp4")
        if os.path.exists(save_path):
            continue
        else:
            if "t2v" in args.task:
                video, image_id = inference_t2v_loop(
                    args, pipeline, batch
                )
            elif "i2v" in args.task:
                video, image_id = inference_i2v_loop(
                    args, pipeline, batch
                )
            elif "flf2v" in args.task:
                video, image_id = inference_flf2v_loop(
                    args, pipeline, batch
                )

            if basic_kwargs.rank == 0 or args.ddp_mode:
                save_path = os.path.join(args.save_folder, f"{image_id}.mp4")
                cache_video(
                    tensor=video[None],
                    save_file=save_path,
                    fps=basic_kwargs.cfg.sample_fps,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1)
                )

                logging.info(f"Saving generated video to {save_path}")

    logging.info("Finished.")


if __name__ == "__main__":
    args = args_wan_init()
    main(args)
