import os
import torch
import torch.amp as amp
import torch.nn.functional as F
from einops import rearrange
from safetensors.torch import load_file


# tensor
def expand_tensor_dims(tensor, ndim):
    while len(tensor.shape) < ndim:
        tensor = tensor.unsqueeze(-1)
    return tensor


# vae
def vae_encode(vae, images, dtype=torch.bfloat16, vae_type="wanx"):
    if vae_type in ["wanx"]:
        images = batch2list(images)
        latents = vae.encode(images)
        latents = list2batch(latents)
    elif vae_type in ["ltx"]:
        with amp.autocast("cuda", dtype=dtype):
            latents = vae.encode(images).latent_dist.sample()

        latents_mean = vae.latents_mean
        latents_std = vae.latents_std
        scaling_factor = 1.0

        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents = (latents - latents_mean) * scaling_factor / latents_std

    return latents

def vae_decode(vae, latents, dtype=torch.bfloat16, vae_type="wanx"):
    if vae_type in ["wanx"]:
        latents = batch2list(latents)
        images = vae.decode(latents)
        images = list2batch(images)

    elif vae_type in ["ltx"]:
        latents_mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = vae.latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        scaling_factor = 1.0
        latents = latents * latents_std / scaling_factor + latents_mean

        with amp.autocast("cuda", dtype=dtype):
            images = vae.decode(latents, return_dict=False)[0]

    return images


def image_encode(
    image_encoder, 
    image, 
    last_image=None,
    image_encoder_type="wanx"
):
    if image_encoder_type in ["wanx"]:
        if image.ndim == 5:
            image = image[:,:,0]
        image = rearrange(image, "b c h w -> c b h w")

        if last_image is not None:
            if last_image.ndim == 5:
                last_image = last_image[:,:,0]

            last_image = rearrange(last_image, "b c h w -> c b h w")
            image_embeds = image_encoder.visual([image, last_image])
        else:
            image_embeds = image_encoder.visual([image])

    return image_embeds


def pack_latents(latents, patch_size=1, patch_size_t=1):
    # Unpacked latents of shape are [B, C, F, H, W] are patched into tokens of shape [B, C, F // p_t, p_t, H // p, p, W // p, p].
    # The patch dimensions are then permuted and collapsed into the channel dimension of shape:
    # [B, F // p_t * H // p * W // p, C * p_t * p * p] (an ndim=3 tensor).
    # dim=0 is the batch size, dim=1 is the effective video sequence length, dim=2 is the effective number of input features
    batch_size, num_channels, num_frames, height, width = latents.shape
    post_patch_num_frames = num_frames // patch_size_t
    post_patch_height = height // patch_size
    post_patch_width = width // patch_size
    latents = latents.reshape(
        batch_size,
        -1,
        post_patch_num_frames,
        patch_size_t,
        post_patch_height,
        patch_size,
        post_patch_width,
        patch_size,
    )
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
    latents = latents.contiguous()
    return latents


def unpack_latents(latents, num_frames, height, width, patch_size=1, patch_size_t=1):
    # Packed latents of shape [B, S, D] (S is the effective video sequence length, D is the effective feature dimensions)
    # are unpacked and reshaped into a video tensor of shape [B, C, F, H, W]. This is the inverse operation of
    # what happens in the `_pack_latents` method.
    batch_size = latents.size(0)
    latents = latents.reshape(
        batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size
    )
    latents = (
        latents.permute(0, 4, 1, 5, 2, 6, 3, 7)
        .flatten(6, 7)
        .flatten(4, 5)
        .flatten(2, 3)
    )
    latents = latents.contiguous()
    return latents


# text encoder
def prompt2states(
    prompt,
    text_encoder,
    device="cuda:0",
    tokenizer=None,
    max_length=128,
    text_encoder_type="wanx",
):
    if isinstance(prompt, str):
        prompt = [prompt]

    if text_encoder_type in ["wanx"]:
        text_states = text_encoder(prompt, device)[0]
        text_states = text_states.unsqueeze(0)
        return text_states
    elif text_encoder_type in ["ltx"]:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_ids = text_inputs.input_ids.to(device)
        text_mask = text_inputs.attention_mask
        text_mask = text_mask.bool().to(device)
        text_states = text_encoder(text_ids)[0]

        return text_states, text_mask


def load_lora_for_pipeline(
    pipeline,
    lora_path,
    LORA_PREFIX_TRANSFORMER="",
    LORA_PREFIX_TEXT_ENCODER="",
    alpha=1.0,
    rank=0,
):
    # load LoRA weight from .safetensors
    state_dict = load_file(lora_path, device=rank)

    visited = []

    # directly update weight in diffusers model
    for key in state_dict:
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        # as we have set the alpha beforehand, so just skip
        if "alpha" in key or key in visited:
            continue

        if "text" in key:
            layer_infos = (
                key.split(".")[0].split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            )
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = (
                key.split(".")[0].split(LORA_PREFIX_TRANSFORMER + "_")[-1].split("_")
            )
            curr_layer = pipeline.transformer

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        pair_keys = []
        if "lora_down" in key:
            pair_keys.append(key.replace("lora_down", "lora_up"))
            pair_keys.append(key)
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace("lora_up", "lora_down"))

        # update weight
        if len(state_dict[pair_keys[0]].shape) == 4:
            weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
            weight_down = (
                state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
            )
            curr_layer.weight.data += alpha * torch.mm(
                weight_up, weight_down
            ).unsqueeze(2).unsqueeze(3)
        else:
            weight_up = state_dict[pair_keys[0]].to(torch.float32)
            weight_down = state_dict[pair_keys[1]].to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down)

        # update visited list
        for item in pair_keys:
            visited.append(item)
    del state_dict

    return pipeline


def load_lora_for_model(
    model,
    lora_path,
    LORA_PREFIX_TRANSFORMER="",
    LORA_PREFIX_TEXT_ENCODER="",
    alpha=1.0,
    rank=0,
):
    # load LoRA weight from .safetensors
    state_dict = load_file(lora_path, device="cpu")

    visited = []

    # directly update weight in diffusers model
    for key in state_dict:
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        # as we have set the alpha beforehand, so just skip
        if "alpha" in key or key in visited:
            continue

        layer_infos = (
            key.split(".")[0].split(LORA_PREFIX_TRANSFORMER + "_")[-1].split("_")
        )
        curr_layer = model

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        pair_keys = []
        if "lora_down" in key:
            pair_keys.append(key.replace("lora_down", "lora_up"))
            pair_keys.append(key)
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace("lora_up", "lora_down"))

        # update weight
        if len(state_dict[pair_keys[0]].shape) == 4:
            weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
            weight_down = (
                state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
            )
            curr_layer.weight.data += alpha * torch.mm(
                weight_up, weight_down
            ).unsqueeze(2).unsqueeze(3)
        else:
            weight_up = state_dict[pair_keys[0]].to(torch.float32)
            weight_down = state_dict[pair_keys[1]].to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down)

        # update visited list
        for item in pair_keys:
            visited.append(item)
    del state_dict

    return model


def load_lora_state_dict(lora_dir):
    lora_path = os.path.join(lora_dir, 'pytorch_lora_transformers_weights.safetensors')
    lora_weights = load_file(lora_path)
    load_lora_weights = {}
    for key in lora_weights:
        load_lora_weights[key.replace('.weight','.default.weight')] = lora_weights[key]

    return load_lora_weights


def transformer_zero_init(transformer):
    for p in transformer.parameters():
        if p.dim() > 1:
            torch.nn.init.zeros_(p.data)
        else:
            torch.nn.init.normal_(p.data)

    return transformer


def prepare_video_condition_wanx(
    vae, 
    video, 
    mask_strategy=[0.4, 0.25, 0.3, 0.05],
):
    # Get mask strategy
    mask_id = torch.multinomial(torch.tensor(mask_strategy), num_samples=1).item()
    bsz, _, num_frames, height, width = video.shape
    latents_height, latents_width = height // 8, width // 8

    # Get video mask
    if mask_id == 0:
        mask = torch.cat([
            torch.ones(bsz, 1, 1, height, width),
            torch.zeros(bsz, 1, num_frames-1, height, width)
        ], dim=2)

    elif mask_id == 1:
        mid_frame = (num_frames - 1) // 2 + 1
        mask = torch.cat([
            torch.ones(bsz, 1, mid_frame, height, width),
            torch.zeros(bsz, 1, num_frames-mid_frame, height, width)
        ], dim=2)

    elif mask_id == 2:
        mask = torch.cat([
            torch.ones(bsz, 1, 1, height, width),
            torch.zeros(bsz, 1, num_frames-2, height, width),
            torch.ones(bsz, 1, 1, height, width)
        ], dim=2)

    elif mask_id == 3:
        num_masked = torch.randint(1, num_frames, (bsz,)).item()
        indices = torch.randperm(num_frames)[:num_masked].sort().values
        mask = torch.zeros(bsz, 1, num_frames, height, width)
        mask[:,:, indices] = 1

    # Encode video mask
    mask = mask.to(video.device, dtype=video.dtype)
    mask_lat_size = torch.cat([
        torch.repeat_interleave(mask[:,:,:1,:,:], dim=2, repeats=4),
        mask[:,:,1:,:,:],
    ], dim=2)
    mask_lat_size = mask_lat_size[:,:,:,::8,::8]
    mask_lat_size = mask_lat_size.view(bsz, -1, 4, latents_height, latents_width).transpose(1,2)

    # Encode video condition
    video_condition = video * mask
    latents_condition = torch.cat([
        mask_lat_size,
        vae_encode(vae, video_condition, "wanx")
    ], dim=1)

    return latents_condition


def batch2list(batch):
    return [item for item in batch]

def list2batch(list):
    return torch.stack(list)


def stable_mse_loss(model_pred, target, weighting=None, threshold=50):
    if weighting is None:
        weighting = torch.ones_like(target)

    diff = model_pred - target
    mask = (diff.abs() <= threshold).float()
    loss = F.mse_loss(model_pred, target, reduction="none")
    masked_loss = weighting * mask * loss
    masked_loss = masked_loss.mean()

    return masked_loss