import os
import sys
import logging
import torch
import numpy as np
import argparse
import math
import random
import traceback
import json
import glob
import io
import urllib
import requests
import cv2
import time

from decord import VideoReader, cpu
from easydict import EasyDict
from einops import rearrange
from tqdm import tqdm
from torchvision import transforms
from transformers import AutoProcessor

from diffusers_lite.arguments import args_init
from diffusers_lite.constants import PRECISION_TO_TYPE
from diffusers_lite.wan.modules.vae import WanVAE
from diffusers_lite.wan.modules.t5 import T5EncoderModel
from diffusers_lite.wan.modules.clip import CLIPModel
from diffusers_lite.utils.data_utils import split_list, align_ceil_to, align_floor_to
from diffusers_lite.utils.diffusion_utils import (
    vae_encode,
    image_encode,
    prompt2states,
)
from omegaconf import OmegaConf

DEVICE = "cuda"
DTYPE = torch.float16

def read_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def write_json(json_data,json_file, encoding='utf-8'):
    with open(json_file, 'w') as file:
        json.dump(json_data,file,indent=4)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

     
logging.basicConfig(stream=sys.stdout,
                    filemode='a',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S',
                    format='%(asctime)s.%(msecs)03d %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')

logger = logging.getLogger('default')
logFormater = logging.Formatter("%(asctime)s.%(msecs)03d %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
                                datefmt='%Y-%m-%d %H:%M:%S')

def load_and_analyze_video(video_path, args):
    if video_path.startswith('http'):
        req = urllib.request.Request(video_path)
        with urllib.request.urlopen(req, timeout=20) as resp:
            video_reader = VideoReader(io.BytesIO(resp.read()), ctx=cpu(0))
    else:
        video_reader = VideoReader(video_path)
    
    video_fps = video_reader.get_avg_fps()
    total_frames = len(video_reader)
    frame_interval = video_fps / args.extract_fps
    extract_frames = min(
        int(math.ceil((total_frames * args.extract_fps) / video_fps)), 
        args.num_frames
    )
    
    return video_reader, video_fps, total_frames, frame_interval, extract_frames

def get_common_video_params(win_video_path, lose_video_path, args):
    win_reader, win_fps, win_total, win_interval, win_frames = load_and_analyze_video(win_video_path, args)
    lose_reader, lose_fps, lose_total, lose_interval, lose_frames = load_and_analyze_video(lose_video_path, args)
    
    common_frames = min(win_frames, lose_frames)
    common_frames = align_floor_to(common_frames-1, alignment=4) + 1
    
    print(f"Win video - fps:{win_fps}, total_frames:{win_total}, extract_frames:{win_frames}")
    print(f"Lose video - fps:{lose_fps}, total_frames:{lose_total}, extract_frames:{lose_frames}")
    print(f"Common extract_frames: {common_frames}")
    
    return win_reader, lose_reader, common_frames

def extract_video_frames(video_reader, common_frames, args, video_path):
    total_frames = len(video_reader)
    video_fps = video_reader.get_avg_fps()
    frame_interval = video_fps / args.extract_fps
    
    frame_indices = []
    current_position = args.start_idx
    
    while len(frame_indices) < common_frames and current_position < total_frames:
        frame_indices.append(int(current_position))
        current_position += frame_interval
    
    frame_indices = np.array(frame_indices[:common_frames])
    print(f"Frame indices: {frame_indices}, count: {len(frame_indices)}")
    
    frames = video_reader.get_batch(frame_indices).asnumpy()
    
    return frames

def height_width_scale(frames, args):
    height, width = frames.shape[1], frames.shape[2]
    scale = args.resolution[0] / min(height, width)

    resize_height_scale = align_ceil_to(int(height * scale), 32)
    resize_width_scale = align_ceil_to(int(width * scale), 32)
    
    max_resolution = args.resolution[0] * args.aspect_ratio
    max_resolution = align_ceil_to(max_resolution, 32)
    height_scale = resize_height_scale
    width_scale = resize_width_scale
    
    
    if resize_height_scale > max_resolution:
        height_scale = max_resolution
        
    if resize_width_scale > max_resolution:
        width_scale = max_resolution
    if int(width * scale) < width_scale:
        scale_new = width_scale / width
    else:
        scale_new = scale
    if int(height * scale_new) < height_scale:
        scale_new = height_scale/height
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((int(height * scale_new), int(width * scale_new))),
        transforms.CenterCrop((height_scale, width_scale)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    return  height_scale, width_scale, transform



def process_video_frames(frames, args, save_first_frame_path, height_scale, width_scale,transform):
    processed_frames = []
    for i, frame in enumerate(frames):
        processed_frame = transform(frame)
        processed_frames.append(processed_frame)
        
        if i == 0 and save_first_frame_path:
            denormalized_frame = processed_frame * 0.5 + 0.5
            denormalized_frame = denormalized_frame.clamp(0, 1)
            first_frame = transforms.ToPILImage()(denormalized_frame)
            first_frame.save(save_first_frame_path)

    print(f"Processed video scale height {height_scale} width {width_scale}")
    return torch.stack(processed_frames)

def encode_single_video(video_tensor, basic_kwargs, model_kwargs):
    vae = model_kwargs.vae
    image_encoder = model_kwargs.image_encoder
    
    video = video_tensor.unsqueeze(0).to(basic_kwargs.device)  # (b, t, c, h, w)
    video = rearrange(video, "b t c h w -> b c t h w")

    batch_size, _, num_frames, height, width = video.shape

    image = video[:, :, 0:1, :, :]
    video_condition = torch.cat([
        image, 
        image.new_zeros(image.shape[0], image.shape[1], num_frames - 1, height, width)
    ], dim=2).to(basic_kwargs.device)

    with torch.autocast(device_type="cuda", dtype=basic_kwargs.dtype):
        latents = vae_encode(vae, video, vae_type="wanx")
        latents_condition = vae_encode(vae, video_condition, vae_type="wanx")

    image_embeds = image_encode(image_encoder, image, image_encoder_type="wanx")

    return {
        "latents": latents,
        "image_embeds": image_embeds,
        "latents_condition": latents_condition
    }

def encode_video(args, video_path, basic_kwargs, model_kwargs, save_first_frame_path):
    video_reader, video_fps, total_frames, frame_interval, extract_frames = load_and_analyze_video(video_path, args)
    extract_frames = align_floor_to(extract_frames-1, alignment=4) + 1
    
    frames = extract_video_frames(video_reader, extract_frames, args, video_path)
    height_scale, width_scale,transform = height_width_scale(frames, args)
    video_tensor = process_video_frames(frames, args, save_first_frame_path, height_scale, width_scale,transform)
    
    encode_kwargs = encode_single_video(video_tensor, basic_kwargs, model_kwargs)

    print(f"Encoded shapes -latents: {encode_kwargs['latents'].shape}, "
          f"Lose latents: {encode_kwargs['latents'].shape}")
    
    return encode_kwargs

def basic_init(args):
    device = torch.device("cuda", 0)
    dtype = PRECISION_TO_TYPE[args.precision]

    basic_kwargs = EasyDict({
        "device": device,
        "dtype": dtype,
    })

    return basic_kwargs

def model_init(args, basic_kwargs):
    vae = WanVAE(
        vae_pth=args.vae_path,
        device=basic_kwargs.device,
    )

    image_encoder = CLIPModel(
        checkpoint_path=args.image_encoder_path,
        tokenizer_path=args.image_processor_path,
        dtype=basic_kwargs.dtype,
        device=basic_kwargs.device,
    )

    text_encoder = T5EncoderModel(
        checkpoint_path=args.text_encoder_path,
        tokenizer_path=args.tokenizer_path,
        text_len=args.max_sequence_length,
        dtype=basic_kwargs.dtype,
        device=basic_kwargs.device,
        shard_fn=None,
    )

    model_kwargs = EasyDict({
        "vae": vae,
        "image_encoder": image_encoder,
        "text_encoder": text_encoder,
    })

    return model_kwargs

def encode_caption(args, caption, basic_kwargs, model_kwargs):
    text_encoder = model_kwargs.text_encoder

    text_states = prompt2states(
        caption, text_encoder, device=basic_kwargs.device, text_encoder_type=args.model_type,
    )

    return text_states

@torch.no_grad()
def main_wan(config):
    seed_everything(config.seed)
    start = time.time()
    
    basic_kwargs = basic_init(config)
    model_kwargs = model_init(config, basic_kwargs)
    print(f"Load VAE: {time.time() - start:.2f}s")

    output_base_dir = config.save_dir
    save_latents_dir = os.path.join(output_base_dir, 'latents')
    save_first_frame_dir = os.path.join(output_base_dir, 'first_frame')
    save_clip_dir = os.path.join(output_base_dir, 'meta_v1')

    for dir_path in [save_latents_dir, save_clip_dir, save_first_frame_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    data = read_json(config.json_path)

    for clip_data in data:
        caption_short = clip_data['short_caption']
        caption_long = clip_data['long_caption']
        
        if "video_path" in clip_data  and clip_data['video_path']:
            video_path = clip_data["video_path"]
            base_name = clip_data["source_id"]
            refl_metafile_path = os.path.join(save_clip_dir, base_name + '_meta_v1.json')
            if not os.path.isfile(refl_metafile_path):
                vae_latent_path = os.path.join(save_latents_dir, base_name + '.npy')
                f1_black_path = os.path.join(save_latents_dir, base_name + '_f1_black.npy')
                imgclip_path = os.path.join(save_latents_dir, base_name + '_img_clip.npy')
                first_frame_path = os.path.join(save_first_frame_dir, base_name + '.jpg')

                textshort_path = os.path.join(save_latents_dir, base_name + '_textshort.npy')
                textlong_path = os.path.join(save_latents_dir, base_name + '_textlong.npy')
                
                try:
                    encode_kwargs = encode_video(
                        config,video_path, basic_kwargs, model_kwargs, first_frame_path
                    )
                    
                    text_states_short = encode_caption(config, caption_short, basic_kwargs, model_kwargs)
                    text_states_long = encode_caption(config, caption_long, basic_kwargs, model_kwargs)
                    
                    np.save(vae_latent_path, encode_kwargs["latents"].to(torch.float32).cpu().numpy())
                    np.save(f1_black_path, encode_kwargs["latents_condition"].to(torch.float32).cpu().numpy())
                    np.save(imgclip_path, encode_kwargs["image_embeds"].to(torch.float32).cpu().numpy())

                    np.save(textshort_path, text_states_short.to(torch.float32).cpu().numpy())
                    np.save(textlong_path, text_states_long.to(torch.float32).cpu().numpy())
                    
                    dpo_meta_data = clip_data.copy()
                    dpo_meta_data.update({
                        'vae_latent_path': vae_latent_path,
                        'f1_black_path': f1_black_path,
                        'imgclip_path': imgclip_path,
                        'latent_shape': encode_kwargs["latents"].shape,

                        'textshort_path': textshort_path,
                        'text_states_short_shape': text_states_short.shape,
                        'textlong_path': textlong_path,
                        'text_states_long_shape': text_states_long.shape,
                    })
                    
                    with open(refl_metafile_path, 'w') as file:
                        json.dump(dpo_meta_data, file, indent=4, ensure_ascii=False)
                    
                    print(f'Data processed successfully: {refl_metafile_path}')
                    
                except Exception as e:
                    print(f'Error processing DPO pair: {e}')
                    traceback.print_exc()
                    continue
                    
            else:
                print(f'Data already processed: {refl_metafile_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='', type=str)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    main_wan(config)