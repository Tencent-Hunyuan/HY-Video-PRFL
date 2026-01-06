import json
import os
import random
import traceback
from PIL import Image

import torch
import numpy as np
from decord import VideoReader
from easydict import EasyDict
from einops import rearrange
from torch.utils.data import Dataset
from torchvision import transforms

from ..utils.data_utils import align_floor_to, align_ceil_to
from ..constants import NULL_DIR


class Image2VideoTrainDataset(Dataset):
    def __init__(
        self,
        task="i2v-14b-480p",
        dataset_type="wanx",
        meta_file_list=[],
        meta_file_lose_list=[],
        uncond_prob=[0.0, 0.0],
        sp_size=1,
        patch_size=[1,2,2],
    ):
        self.task = task
        self.dataset_type = dataset_type
        self.uncond_prompt_prob = uncond_prob[0]
        self.uncond_image_prob = uncond_prob[-1]
        self.sp_size = sp_size
        self.patch_size = patch_size
        self.meta_paths = []

        for meta_file in meta_file_list:
            self.meta_paths.extend(
                [line.strip() for line in open(meta_file, "r").readlines()]
            )
        if len(meta_file_lose_list) > 0:
            self.meta_paths_lose = [] 
            for meta_file in meta_file_lose_list:
                self.meta_paths_lose.extend(
                    [line.strip() for line in open(meta_file, "r").readlines()]
                )

    def __len__(self):
        return len(self.meta_paths)

    def __getitem__(self, idx):
        try_times = 100
        for _ in range(try_times):
            try:
                if self.dataset_type in ["refl"]:
                    return self.get_batch_lrm_refl(idx)
                elif self.dataset_type in ["lrm_ce"]:
                    return self.get_batch_lrm_ce(idx)
                elif self.dataset_type in ["lrm_bt_online"]:
                    return self.get_batch_lrm_bt_online(idx)
            except Exception as e:
                print(
                    f"Error details: {str(e)}-{idx}-{self.meta_paths[idx]}-{traceback.format_exc()}\n"
                )
                idx = np.random.randint(len(self.meta_paths))

        raise RuntimeError("Too many bad data.")
    
    def get_batch_lrm_refl(self, idx):
        data_json_path = self.meta_paths[idx]

        with open(data_json_path, "r") as f:
            data_dict = json.load(f)

        # video
        if 'video_vae_latent_path' in data_dict.keys():
            latents_path = data_dict["video_vae_latent_path"]
        elif 'vae_latent_path' in data_dict.keys():
            latents_path = data_dict["vae_latent_path"]
        else:
            latents_path = data_dict["latents_path"]
        latents = np.load(latents_path)[0]
        latents = torch.from_numpy(latents)
        frames = latents.shape[1]

        # text states
        if 'textshort_path' in data_dict and 'textlong_path' in data_dict:
            text_states_path = data_dict["textshort_path"]
            text_states_path_long = data_dict["textlong_path"]
            prompt= data_dict["short_caption"]
            if random.random() <= 0.7:
                text_states_path = text_states_path_long
                prompt=data_dict["long_caption"]
        else:
            text_states_path = data_dict["text_en_path"]
            prompt= data_dict["prompt"]
        text_states = np.load(text_states_path)[0]
        text_states = torch.from_numpy(text_states)
        
        # image embeds
        image_embeds_path = data_dict["imgclip_path"]
        image_embeds = torch.from_numpy(np.load(image_embeds_path))
        image_embeds = rearrange(image_embeds, "b s d -> (b s) d")

        # latents condition
        if "f1_black_path" in data_dict.keys():
            latents_condition_path = data_dict["f1_black_path"]
        else:
            latents_condition_path = data_dict["latents_condition_path"]
        latents_condition = np.load(latents_condition_path)[0]
        latents_condition = torch.from_numpy(latents_condition)

        # distill prompts
        if "flf2v" in self.task:
            uncond_text_states_path = os.path.join(NULL_DIR, f"wanx/uncond_flf2v.npy")
        else:
            uncond_text_states_path = os.path.join(NULL_DIR, f"wanx/uncond.npy")
        uncond_text_states = np.load(uncond_text_states_path)[0]
        uncond_text_states = torch.from_numpy(uncond_text_states)

        # drop prompts
        random_number = random.random()
        if random_number < self.uncond_prompt_prob:
            null_text_states_path = os.path.join(NULL_DIR, f"wanx/null.npy")
            null_text_states = np.load(null_text_states_path)[0]
            text_states = torch.from_numpy(null_text_states)

        return latents, text_states, uncond_text_states, image_embeds, latents_condition,prompt#,inference_data_dict
    
    def get_batch_refl(self, idx):
        data_json_path = self.meta_paths[idx]

        with open(data_json_path, "r") as f:
            data_dict = json.load(f)

        # video
        if 'video_vae_latent_path' in data_dict.keys():
            latents_path = data_dict["video_vae_latent_path"]
        elif 'vae_latent_path' in data_dict.keys():
            latents_path = data_dict["vae_latent_path"]
        else:
            latents_path = data_dict["latents_path"]
        latents = np.load(latents_path)[0]
        latents = torch.from_numpy(latents)
        frames = latents.shape[1]

        # text states
        text_states_path = data_dict["textshort_path"]
        text_states_path_long = data_dict["textlong_path"]
        prompt= data_dict["short_caption"]
        if random.random() <= 0.7:
            text_states_path = text_states_path_long
            prompt=data_dict["long_caption"]
        text_states = np.load(text_states_path)[0]
        text_states = torch.from_numpy(text_states)
        
        image_embeds_path = data_dict["imgclip_path"]
        image_embeds = torch.from_numpy(np.load(image_embeds_path))
        image_embeds = rearrange(image_embeds, "b s d -> (b s) d")

        if "f1_black_path" in data_dict.keys():
            latents_condition_path = data_dict["f1_black_path"]
        else:
            latents_condition_path = data_dict["latents_condition_path"]
        latents_condition = np.load(latents_condition_path)[0]
        latents_condition = torch.from_numpy(latents_condition)

        if "flf2v" in self.task:
            uncond_text_states_path = os.path.join(NULL_DIR, f"wanx/uncond_flf2v.npy")
        else:
            uncond_text_states_path = os.path.join(NULL_DIR, f"wanx/uncond.npy")
        uncond_text_states = np.load(uncond_text_states_path)[0]
        uncond_text_states = torch.from_numpy(uncond_text_states)

        random_number = random.random()
        if random_number < self.uncond_prompt_prob:
            null_text_states_path = os.path.join(NULL_DIR, f"wanx/null.npy")
            null_text_states = np.load(null_text_states_path)[0]
            text_states = torch.from_numpy(null_text_states)

        return latents, text_states, uncond_text_states, image_embeds, latents_condition, prompt # inference_data_dict
    
    def get_batch_lrm_ce(self, idx):
        data_json_path = self.meta_paths[idx]

        with open(data_json_path, "r") as f:
            data_dict = json.load(f)

        source_id = data_dict["source_id"]

        if 'video_vae_latent_path' in data_dict:
            latents_path = data_dict["video_vae_latent_path"]
        else:
            latents_path = data_dict["vae_latent_path"]
        
        latents = np.load(latents_path)[0]
        latents = torch.from_numpy(latents)
        frames = latents.shape[1]
        
        if 'save_textshort_path' in data_dict:
            text_states_path = data_dict["save_textshort_path"]
        elif 'textshort_path' in data_dict:
            text_states_path = data_dict["textshort_path"]
        else:
            text_states_path = data_dict["text_en_path"]
        
        text_states = np.load(text_states_path)[0]
        text_states = torch.from_numpy(text_states)

        if "image_embeds" in data_dict:
            image_embeds_path = data_dict["image_embeds"]
        else:
            image_embeds_path = data_dict["imgclip_path"]
        
        image_embeds = torch.from_numpy(np.load(image_embeds_path))
        image_embeds = rearrange(image_embeds, "b s d -> (b s) d")

        if "f1_black_path" in data_dict:
            latents_condition_path = data_dict["f1_black_path"]
        else:
            latents_condition_path = data_dict["latents_condition_path"]
        
        latents_condition = np.load(latents_condition_path)[0]
        latents_condition = torch.from_numpy(latents_condition)
        
        if "flf2v" in self.task:
            uncond_text_states_path = os.path.join(NULL_DIR, "wanx/uncond_flf2v.npy")
        else:
            uncond_text_states_path = os.path.join(NULL_DIR, "wanx/uncond.npy")
        
        uncond_text_states = np.load(uncond_text_states_path)[0]
        uncond_text_states = torch.from_numpy(uncond_text_states)
        
        if "model" in data_dict:
            data_from_model = data_dict["model"]
        else:
            data_from_model = ""
        if "text_alignment" in data_dict:
            text_alignment = data_dict["text_alignment"]
        else:
            text_alignment = 0
        if "blur_quality" in data_dict:
            blur_quality = data_dict["blur_quality"]
        else:
            blur_quality = 0
        if "physics_quality" in data_dict:
            physics_quality = data_dict["physics_quality"]
        else:
            physics_quality = 0
        if "human_quality" in data_dict:
            human_quality = data_dict["human_quality"]
        else:
            human_quality = 0

        if text_alignment == "poor" or text_alignment is None: text_alignment = 0
        if blur_quality == "poor" or blur_quality is None: blur_quality = 0 
        if physics_quality == "poor" or physics_quality is None: physics_quality = 0 
        if human_quality == "poor" or human_quality is None: human_quality = 0 
        if text_alignment == "good": text_alignment = 1 
        if blur_quality == "good": blur_quality = 1 
        if physics_quality == "good": physics_quality = 1 
        if human_quality == "good": human_quality = 1 

        return (latents, text_states, uncond_text_states, image_embeds, latents_condition,
                data_from_model, text_alignment, blur_quality, physics_quality, human_quality)
    
    def get_batch_lrm_bt_online(self, idx):
        data_json_path = self.meta_paths[idx]
        
        if self.meta_paths_lose is None or len(self.meta_paths_lose) == 0:
            raise ValueError("meta_paths_lose is None or empty. Please ensure bt=True and meta_file_list_lose is provided.")
        
        data_json_path_lose = self.meta_paths_lose[random.randint(0, len(self.meta_paths_lose)-1)]

        with open(data_json_path, "r") as f:
            data_dict = json.load(f)
        with open(data_json_path_lose, "r") as f:
            data_dict_lose = json.load(f)

        if 'video_vae_latent_path' in data_dict:
            latents_path = data_dict["video_vae_latent_path"]
            latents_path_lose = data_dict_lose["video_vae_latent_path"]
        else:
            latents_path = data_dict["vae_latent_path"]
            latents_path_lose = data_dict_lose["vae_latent_path"]
        
        latents = np.load(latents_path)[0]
        latents = torch.from_numpy(latents)
        frames = latents.shape[1]
        latents_lose = np.load(latents_path_lose)[0]
        latents_lose = torch.from_numpy(latents_lose)
        frames_lose = latents_lose.shape[1]
        assert latents.shape == latents_lose.shape, f'latents.shape {latents.shape} != latents_lose.shape {latents_lose.shape}'

        if 'save_textshort_path' in data_dict:
            text_states_path = data_dict["save_textshort_path"]
            text_states_path_lose = data_dict_lose["save_textshort_path"]
        elif 'textshort_path' in data_dict:
            text_states_path_lose = data_dict_lose["textshort_path"]
            text_states_path = data_dict["textshort_path"]
        else:
            text_states_path = data_dict["text_en_path"]
            text_states_path_lose = data_dict_lose["text_en_path"]
        
        text_states = np.load(text_states_path)[0]
        text_states = torch.from_numpy(text_states)
        text_states_lose = np.load(text_states_path_lose)[0]
        text_states_lose = torch.from_numpy(text_states_lose)

        if "image_embeds" in data_dict:
            image_embeds_path = data_dict["image_embeds"]
            image_embeds_path_lose = data_dict_lose["image_embeds"]
        else:
            image_embeds_path = data_dict["imgclip_path"]
            image_embeds_path_lose = data_dict_lose["imgclip_path"]
        
        image_embeds = torch.from_numpy(np.load(image_embeds_path))
        image_embeds = rearrange(image_embeds, "b s d -> (b s) d")
        image_embeds_lose = torch.from_numpy(np.load(image_embeds_path_lose))
        image_embeds_lose = rearrange(image_embeds_lose, "b s d -> (b s) d")

        if "f1_black_path" in data_dict:
            latents_condition_path = data_dict["f1_black_path"]
            latents_condition_path_lose = data_dict_lose["f1_black_path"]
        else:
            latents_condition_path = data_dict["latents_condition_path"]
            latents_condition_path_lose = data_dict_lose["latents_condition_path"]
        
        latents_condition = np.load(latents_condition_path)[0]
        latents_condition = torch.from_numpy(latents_condition)
        latents_condition_lose = np.load(latents_condition_path_lose)[0]
        latents_condition_lose = torch.from_numpy(latents_condition_lose)

        if "flf2v" in self.task:
            uncond_text_states_path = os.path.join(NULL_DIR, "wanx/uncond_flf2v.npy")
            uncond_text_states_path_lose = os.path.join(NULL_DIR, "wanx/uncond_flf2v.npy")
        else:
            uncond_text_states_path = os.path.join(NULL_DIR, "wanx/uncond.npy")
            uncond_text_states_path_lose = os.path.join(NULL_DIR, "wanx/uncond.npy")
        
        uncond_text_states = np.load(uncond_text_states_path)[0]
        uncond_text_states = torch.from_numpy(uncond_text_states)
        uncond_text_states_lose = np.load(uncond_text_states_path_lose)[0]
        uncond_text_states_lose = torch.from_numpy(uncond_text_states_lose)
        
        return (latents, text_states, uncond_text_states, image_embeds, latents_condition,
                latents_lose, text_states_lose, uncond_text_states_lose, image_embeds_lose, latents_condition_lose)
    

class Image2VideoEvalDataset(Dataset):
    def __init__(self, file_path, resolution=(512,512), alignment=16, do_scale=True):
        self.prompts = []
        self.image_ids = []
        self.image_paths = []
        self.last_image_paths = []
        self.seeds = []

        if file_path.endswith(".txt"):
            with open(file_path, "r") as file:
                for line in file:
                    prompt = line.strip()
                    self.prompts.append(prompt)
        
        elif file_path.endswith(".json"):
            with open(file_path, "r") as f:
                datas = json.load(f)
            for data in datas:
                self.prompts.append(data["caption"].strip())
                if "image_id" in data.keys():
                    self.image_ids.append(data["image_id"])
                if "image_path" in data.keys():
                    self.image_paths.append(data["image_path"])
                if "last_image_path" in data.keys():
                    self.last_image_paths.append(data["last_image_path"])
                if "seed" in data.keys():
                    self.seeds.append(data["seed"])

        self.resolution = resolution
        self.alignment = alignment
        self.do_scale = do_scale

        print(f"[INFO] Load text and image dataset done, total len {len(self.prompts)}")

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        prompt = self.prompts[index]

        if len(self.image_paths) > 0:
            image_path = self.image_paths[index]
            image_id = image_path.split("/")[-1].split(".")[0]

            # Load image
            image = Image.open(image_path).convert("RGB")

            # Resize image
            width, height = image.size
            scale = min(min(self.resolution) / min(width, height), max(self.resolution) / max(width, height))

            width_scale = align_ceil_to(int(width * scale), self.alignment)
            height_scale = align_ceil_to(int(height * scale), self.alignment)

            if not self.do_scale:
                width_scale = width
                height_scale = height

            transform = transforms.Compose(
                [
                    transforms.Resize((height_scale, width_scale)),
                    transforms.ToTensor(),
                ]
            )

            image = transform(image)
        else:
            image_path = ""
            image = ""
            image_id = str(index)
        
        if len(self.image_ids) > 0:
            image_id = self.image_ids[index]

        # Load last image
        if len(self.last_image_paths) > 0:
            last_image_path = self.last_image_paths[index]
            last_image = Image.open(last_image_path).convert("RGB")
            last_image = transform(last_image)
        else:
            last_image = ""
        
        if len(self.seeds) > 0:
            seed = self.seeds[index]
            image_id += f'_seed_{seed}'
        else:
            seed = 42

        return {
            "prompt": prompt, 
            "image": image, 
            "last_image": last_image,
            "image_id": image_id, 
            "image_path": image_path,
            "seed": seed,
        }
    
    