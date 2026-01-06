# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Any, Tuple

import os
import torch
import functools
import torch.distributed as dist
from torch import Tensor

from ..utils.parallel_states import nccl_info, get_teacher_student_parallel_state


def broadcast(input_: torch.Tensor):
    src = nccl_info.group_id * nccl_info.sp_size
    dist.broadcast(input_, src=src, group=nccl_info.group)

def broadcast_within_ts_unit(input_):
    src = nccl_info.ts_unit_group_id * nccl_info.ts_unit_size
    dist.broadcast(input_, src=src, group=nccl_info.ts_unit_group)

def broadcast_global(input_: torch.Tensor):
    dist.broadcast(input_, src=0, group=None)    
    
def broadcast_dict(input_: dict):
    src = nccl_info.group_id * nccl_info.sp_size
    for k, v in input_.items():
        if isinstance(input_[k], torch.Tensor):
            dist.broadcast(input_[k], src=src, group=nccl_info.group)

def broadcast_dict_within_ts_unit(input_: dict):
    src = nccl_info.ts_unit_group_id * nccl_info.ts_unit_size
    for k, v in input_.items():
        if isinstance(input_[k], torch.Tensor):
            dist.broadcast(input_[k], src=src, group=nccl_info.ts_unit_group)

def _all_to_all_4D(
    input: torch.tensor, scatter_idx: int = 2, gather_idx: int = 1, group=None
) -> torch.tensor:
    """
    all-to-all for QKV

    Args:
        input (torch.tensor): a tensor sharded along dim scatter dim
        scatter_idx (int): default 1
        gather_idx (int): default 2
        group : torch process group

    Returns:
        torch.tensor: resharded tensor (bs, seqlen/P, hc, hs)
    """
    assert (
        input.dim() == 4
    ), f"input must be 4D tensor, got {input.dim()} and shape {input.shape}"

    seq_world_size = dist.get_world_size(group)

    if scatter_idx == 2 and gather_idx == 1:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen/P, hc, hs) output: (bs, seqlen, hc/P, hs)
        bs, shard_seqlen, hc, hs = input.shape
        seqlen = shard_seqlen * seq_world_size
        shard_hc = hc // seq_world_size

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen/P, hc, hs) -reshape-> (bs, seq_len/P, P, hc/P, hs) -transpose(0,2)-> (P, seq_len/P, bs, hc/P, hs)
        input_t = (
            input.reshape(bs, shard_seqlen, seq_world_size, shard_hc, hs)
            .transpose(0, 2)
            .contiguous()
        )

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, seq_len/P, bs, hc/P, hs) scatter seqlen -all2all-> (P, seq_len/P, bs, hc/P, hs) scatter head
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            torch.cuda.synchronize()
        else:
            output = input_t
        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(seqlen, bs, shard_hc, hs)

        # (seq_len, bs, hc/P, hs) -reshape-> (bs, seq_len, hc/P, hs)
        output = output.transpose(0, 1).contiguous().reshape(bs, seqlen, shard_hc, hs)

        return output

    elif scatter_idx == 1 and gather_idx == 2:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen, hc/P, hs) output: (bs, seqlen/P, hc, hs)
        bs, seqlen, shard_hc, hs = input.shape
        hc = shard_hc * seq_world_size
        shard_seqlen = seqlen // seq_world_size
        seq_world_size = dist.get_world_size(group)

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen, hc/P, hs) -reshape-> (bs, P, seq_len/P, hc/P, hs) -transpose(0, 3)-> (hc/P, P, seqlen/P, bs, hs) -transpose(0, 1) -> (P, hc/P, seqlen/P, bs, hs)
        input_t = (
            input.reshape(bs, seq_world_size, shard_seqlen, shard_hc, hs)
            .transpose(0, 3)
            .transpose(0, 1)
            .contiguous()
            .reshape(seq_world_size, shard_hc, shard_seqlen, bs, hs)
        )

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, bs x hc/P, seqlen/P, hs) scatter seqlen -all2all-> (P, bs x seq_len/P, hc/P, hs) scatter head
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            torch.cuda.synchronize()
        else:
            output = input_t

        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(hc, shard_seqlen, bs, hs)

        # (hc, seqlen/N, bs, hs) -tranpose(0,2)-> (bs, seqlen/N, hc, hs)
        output = output.transpose(0, 2).contiguous().reshape(bs, shard_seqlen, hc, hs)

        return output
    else:
        raise RuntimeError("scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")


class SeqAllToAll4D(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        input: Tensor,
        scatter_idx: int,
        gather_idx: int,
    ) -> Tensor:
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx

        return _all_to_all_4D(input, scatter_idx, gather_idx, group=group)

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, None, None]:
        return (
            None,
            SeqAllToAll4D.apply(
                ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx
            ),
            None,
            None,
        )


def all_to_all_4D(
    input_: torch.Tensor,
    scatter_dim: int = 2,
    gather_dim: int = 1,
):
    return SeqAllToAll4D.apply(nccl_info.group, input_, scatter_dim, gather_dim)


def _all_to_all(
    input_: torch.Tensor,
    world_size: int,
    group: dist.ProcessGroup,
    scatter_dim: int,
    gather_dim: int,
):
    input_list = [
        t.contiguous() for t in torch.tensor_split(input_, world_size, scatter_dim)
    ]
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()


class _AllToAll(torch.autograd.Function):
    """All-to-all communication.

    Args:
        input_: input matrix
        process_group: communication group
        scatter_dim: scatter dimension
        gather_dim: gather dimension
    """

    @staticmethod
    def forward(ctx, input_, process_group, scatter_dim, gather_dim):
        ctx.process_group = process_group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.world_size = dist.get_world_size(process_group)
        output = _all_to_all(
            input_, ctx.world_size, process_group, scatter_dim, gather_dim
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = _all_to_all(
            grad_output,
            ctx.world_size,
            ctx.process_group,
            ctx.gather_dim,
            ctx.scatter_dim,
        )
        return (
            grad_output,
            None,
            None,
            None,
        )


def all_to_all(
    input_: torch.Tensor,
    scatter_dim: int = 2,
    gather_dim: int = 1,
):
    return _AllToAll.apply(input_, nccl_info.group, scatter_dim, gather_dim)


class _AllGather(torch.autograd.Function):
    """All-gather communication with autograd support.

    Args:
        input_: input tensor
        dim: dimension along which to concatenate
    """

    @staticmethod
    def forward(ctx, input_, dim):
        ctx.dim = dim
        world_size = nccl_info.sp_size
        group = nccl_info.group
        input_size = list(input_.size())

        ctx.input_size = input_size[dim]

        tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
        input_ = input_.contiguous()
        dist.all_gather(tensor_list, input_, group=group)

        output = torch.cat(tensor_list, dim=dim)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        world_size = nccl_info.sp_size
        rank = nccl_info.rank_within_group
        dim = ctx.dim
        input_size = ctx.input_size

        sizes = [input_size] * world_size

        grad_input_list = torch.split(grad_output, sizes, dim=dim)
        grad_input = grad_input_list[rank]

        return grad_input, None


def all_gather(input_: torch.Tensor, dim: int = 1):
    """Performs an all-gather operation on the input tensor along the specified dimension.

    Args:
        input_ (torch.Tensor): Input tensor of shape [B, H, S, D].
        dim (int, optional): Dimension along which to concatenate. Defaults to 1.

    Returns:
        torch.Tensor: Output tensor after all-gather operation, concatenated along 'dim'.
    """
    return _AllGather.apply(input_, dim)

class _AllGather_TeacherStudent(torch.autograd.Function):
    """All-gather communication with autograd support.

    Args:
        input_: input tensor
        dim: dimension along which to concatenate
    """

    @staticmethod
    def forward(ctx, input_, dim):
        ctx.dim = dim
        world_size = nccl_info.ts_unit_size
        group = nccl_info.ts_unit_group
        input_size = list(input_.size())

        ctx.input_size = input_size[dim]

        tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
        input_ = input_.contiguous()
        dist.all_gather(tensor_list, input_, group=group)

        output = torch.cat(tensor_list, dim=dim)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        world_size = nccl_info.ts_unit_size
        rank = nccl_info.rank_within_ts_unit_group
        dim = ctx.dim
        input_size = ctx.input_size

        sizes = [input_size] * world_size
        grad_input_list = torch.split(grad_output, sizes, dim=dim)
        grad_input = grad_input_list[rank]
        return grad_input, None    
        
def all_gather_ts(input_: torch.Tensor, dim: int = 1):
    """Performs an all-gather operation on the input tensor along the specified dimension.

    Args:
        input_ (torch.Tensor): Input tensor of shape [B, H, S, D].
        dim (int, optional): Dimension along which to concatenate. Defaults to 1.

    Returns:
        torch.Tensor: Output tensor after all-gather operation, concatenated along 'dim'.
    """
    return _AllGather_TeacherStudent.apply(input_, dim)    

    
def prepare_sequence_parallel_data_wanx(
    hidden_states, encoder_hidden_states, uncond_text_states, image_embeds, latents_condition
):
    if nccl_info.sp_size == 1:
        return (
            hidden_states,
            encoder_hidden_states,
            uncond_text_states,
            image_embeds,
            latents_condition,
        )

    def prepare(hidden_states, encoder_hidden_states, uncond_text_states, image_embeds, latents_condition):
        hidden_states = all_to_all(hidden_states, scatter_dim=2, gather_dim=0)
        encoder_hidden_states = all_to_all(
            encoder_hidden_states, scatter_dim=1, gather_dim=0
        )
        uncond_text_states = all_to_all(
            uncond_text_states, scatter_dim=1, gather_dim=0
        )
        image_embeds = all_to_all(image_embeds, scatter_dim=1, gather_dim=0)
        latents_condition = all_to_all(latents_condition, scatter_dim=2, gather_dim=0)

        return (
            hidden_states,
            encoder_hidden_states,
            uncond_text_states,
            image_embeds,
            latents_condition,
        )

    sp_size = nccl_info.sp_size
    frame = hidden_states.shape[2]
    assert frame % sp_size == 0, "frame should be a multiple of sp_size"

    (
        hidden_states,
        encoder_hidden_states,
        uncond_text_states,
        image_embeds,
        latents_condition,
    ) = prepare(
        hidden_states,
        encoder_hidden_states.repeat(1, sp_size, 1),
        uncond_text_states.repeat(1, sp_size, 1),
        image_embeds.repeat(1, sp_size, 1),
        latents_condition,
    )

    return hidden_states, encoder_hidden_states, uncond_text_states, image_embeds, latents_condition


def sp_parallel_dataloader_wrapper_wanx(
    dataloader, device, train_batch_size, sp_size, train_sp_batch_size
):
    while True:
        for data_item in dataloader:
            latents, text_states, uncond_text_states, image_embeds, latents_condition = data_item
            latents = latents.to(device)
            text_states = text_states.to(device)
            uncond_text_states = uncond_text_states.to(device)
            image_embeds = image_embeds.to(device)
            latents_condition = latents_condition.to(device)
            frame = latents.shape[2]
            if frame == 1:
                yield latents, text_states, uncond_text_states, image_embeds, latents_condition
            else:
                latents, text_states, uncond_text_states, image_embeds, latents_condition = (
                    prepare_sequence_parallel_data_wanx(
                        latents, text_states, uncond_text_states, image_embeds, latents_condition
                    )
                )
                assert (
                    train_batch_size * sp_size >= train_sp_batch_size
                ), "train_batch_size * sp_size should be greater than train_sp_batch_size"
                for iter in range(train_batch_size * sp_size // train_sp_batch_size):
                    st_idx = iter * train_sp_batch_size
                    ed_idx = (iter + 1) * train_sp_batch_size
                    yield (
                        latents[st_idx:ed_idx],
                        text_states[st_idx:ed_idx],
                        uncond_text_states[st_idx:ed_idx],
                        image_embeds[st_idx:ed_idx],
                        latents_condition[st_idx:ed_idx],
                    )

def prepare_sequence_parallel_data_wanx_dpo(
    hidden_states, encoder_hidden_states, uncond_text_states, image_embeds, latents_condition,latents_lose
):
    if nccl_info.sp_size == 1:
        return (
            hidden_states,
            encoder_hidden_states,
            uncond_text_states,
            image_embeds,
            latents_condition,
            latents_lose,
        )

    def prepare(hidden_states, encoder_hidden_states, uncond_text_states, image_embeds, latents_condition, latents_lose):
        hidden_states = all_to_all(hidden_states, scatter_dim=2, gather_dim=0)
        latents_lose = all_to_all(latents_lose, scatter_dim=2, gather_dim=0)
        encoder_hidden_states = all_to_all(
            encoder_hidden_states, scatter_dim=1, gather_dim=0
        )
        uncond_text_states = all_to_all(
            uncond_text_states, scatter_dim=1, gather_dim=0
        )
        image_embeds = all_to_all(image_embeds, scatter_dim=1, gather_dim=0)
        latents_condition = all_to_all(latents_condition, scatter_dim=2, gather_dim=0)

        return (
            hidden_states,
            encoder_hidden_states,
            uncond_text_states,
            image_embeds,
            latents_condition,
            latents_lose,
        )

    sp_size = nccl_info.sp_size
    frame = hidden_states.shape[2]
    assert frame % sp_size == 0, "frame should be a multiple of sp_size"

    (
        hidden_states,
        encoder_hidden_states,
        uncond_text_states,
        image_embeds,
        latents_condition,latents_lose,
    ) = prepare(
        hidden_states,
        encoder_hidden_states.repeat(1, sp_size, 1),
        uncond_text_states.repeat(1, sp_size, 1),
        image_embeds.repeat(1, sp_size, 1),
        latents_condition,
        latents_lose
    )

    return hidden_states, encoder_hidden_states, uncond_text_states, image_embeds, latents_condition,latents_lose

def sp_parallel_dataloader_wrapper_wanx_dpo(
    dataloader, device, train_batch_size, sp_size, train_sp_batch_size
):
    while True:
        for data_item in dataloader:
            latents, text_states, uncond_text_states, image_embeds, latents_condition,latent_lose = data_item
            latents = latents.to(device)
            latents_lose = latents.to(device)
            text_states = text_states.to(device)
            uncond_text_states = uncond_text_states.to(device)
            image_embeds = image_embeds.to(device)
            latents_condition = latents_condition.to(device)
            frame = latents.shape[2]
            if frame == 1:
                yield latents, text_states, uncond_text_states, image_embeds, latents_condition,latent_lose
            else:
                latents, text_states, uncond_text_states, image_embeds, latents_condition, latents_lose = (
                    prepare_sequence_parallel_data_wanx_dpo(
                        latents, text_states, uncond_text_states, image_embeds, latents_condition,latent_lose
                    )
                )
                assert (
                    train_batch_size * sp_size >= train_sp_batch_size
                ), "train_batch_size * sp_size should be greater than train_sp_batch_size"
                for iter in range(train_batch_size * sp_size // train_sp_batch_size):
                    st_idx = iter * train_sp_batch_size
                    ed_idx = (iter + 1) * train_sp_batch_size
                    yield (
                        latents[st_idx:ed_idx],
                        text_states[st_idx:ed_idx],
                        uncond_text_states[st_idx:ed_idx],
                        image_embeds[st_idx:ed_idx],
                        latents_condition[st_idx:ed_idx],
                        latents_lose[st_idx:ed_idx],
                    )

def prepare_sequence_parallel_data_ltx(
    hidden_states, encoder_hidden_states, text_mask, uncond_text_states, uncond_text_mask
):
    if nccl_info.sp_size == 1:
        return (
            hidden_states,
            encoder_hidden_states,
            text_mask,
            uncond_text_states,
            uncond_text_mask,
        )

    def prepare(hidden_states, encoder_hidden_states, text_mask, uncond_text_states, uncond_text_mask):
        hidden_states = all_to_all(hidden_states, scatter_dim=2, gather_dim=0)
        encoder_hidden_states = all_to_all(
            encoder_hidden_states, scatter_dim=1, gather_dim=0
        )
        text_mask = all_to_all(text_mask, scatter_dim=1, gather_dim=0)
        uncond_text_states = all_to_all(
            uncond_text_states, scatter_dim=1, gather_dim=0
        )
        uncond_text_mask = all_to_all(
            uncond_text_mask, scatter_dim=1, gather_dim=0
        )

        return (
            hidden_states,
            encoder_hidden_states,
            text_mask,
            uncond_text_states,
            uncond_text_mask,
        )

    sp_size = nccl_info.sp_size
    frame = hidden_states.shape[2]
    assert frame % sp_size == 0, "frame should be a multiple of sp_size"

    (
        hidden_states,
        encoder_hidden_states,
        text_mask,
        uncond_text_states,
        uncond_text_mask,
    ) = prepare(
        hidden_states,
        encoder_hidden_states.repeat(1, sp_size, 1),
        text_mask.repeat(1, sp_size),
        uncond_text_states.repeat(1, sp_size, 1),
        uncond_text_mask.repeat(1, sp_size)
    )

    return hidden_states, encoder_hidden_states, text_mask, uncond_text_states, uncond_text_mask


def sp_parallel_dataloader_wrapper_ltx(
    dataloader, device, train_batch_size, sp_size, train_sp_batch_size
):
    while True:
        for data_item in dataloader:
            latents, text_states, text_mask, uncond_text_states, uncond_text_mask = data_item
            latents = latents.to(device)
            text_states = text_states.to(device)
            text_mask = text_mask.to(device)
            uncond_text_states = uncond_text_states.to(device)
            uncond_text_mask = uncond_text_mask.to(device)
            frame = latents.shape[2]
            if frame == 1:
                yield latents, text_states, text_mask, uncond_text_states, uncond_text_mask
            else:
                latents, text_states, text_mask, uncond_text_states, uncond_text_mask = (
                    prepare_sequence_parallel_data_ltx(
                        latents, text_states, text_mask, uncond_text_states, uncond_text_mask
                    )
                )
                assert (
                    train_batch_size * sp_size >= train_sp_batch_size
                ), "train_batch_size * sp_size should be greater than train_sp_batch_size"
                for iter in range(train_batch_size * sp_size // train_sp_batch_size):
                    st_idx = iter * train_sp_batch_size
                    ed_idx = (iter + 1) * train_sp_batch_size
                    yield (
                        latents[st_idx:ed_idx],
                        text_states[st_idx:ed_idx],
                        text_mask[st_idx:ed_idx],
                        uncond_text_states[st_idx:ed_idx],
                        uncond_text_mask[st_idx:ed_idx],
                    )
                    
                    
def parallelize_model(model):
    original_forward = model.forward

    @functools.wraps(model.__class__.forward)
    def new_forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        text_states: torch.Tensor,
        text_states_2: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        output_features=False,
        output_features_stride=8,
        attention_kwargs=None,
        freqs_cos=None,
        freqs_sin=None,
        return_dict=False,
        guidance=None,
    ):
        x = hidden_states
        sp_size = nccl_info.sp_size
        sp_rank = nccl_info.rank_within_group

        if x.shape[-2] // 2 % sp_size == 0:
            # try to split x by height
            split_dim = -2
        elif x.shape[-1] // 2 % sp_size == 0:
            # try to split x by width
            split_dim = -1
        else:
            raise ValueError(f"Cannot split video sequence into ulysses_degree  ({sp_size}) parts evenly")

        _, _, ot, oh, ow = x.shape
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )
        freqs_cos, freqs_sin = self.get_rotary_pos_embed((tt, th, tw))
        # patch sizes for the temporal, height, and width dimensions are 1, 2, and 2.
        temporal_size, h, w = x.shape[2], x.shape[3] // 2, x.shape[4] // 2

        x = torch.chunk(x, sp_size,dim=split_dim)[sp_rank]

        dim_thw = freqs_cos.shape[-1]
        freqs_cos = freqs_cos.reshape(temporal_size, h, w, dim_thw)
        freqs_cos = torch.chunk(freqs_cos, sp_size,dim=split_dim - 1)[sp_rank]
        freqs_cos = freqs_cos.reshape(-1, dim_thw)
        dim_thw = freqs_sin.shape[-1]
        freqs_sin = freqs_sin.reshape(temporal_size, h, w, dim_thw)
        freqs_sin = torch.chunk(freqs_sin, sp_size,dim=split_dim - 1)[sp_rank]
        freqs_sin = freqs_sin.reshape(-1, dim_thw)

        output = original_forward(
            x,
            timestep,
            text_states,
            text_states_2,
            encoder_attention_mask,
            output_features,
            output_features_stride,
            attention_kwargs,
            freqs_cos,
            freqs_sin,
            return_dict,
            guidance,
        )

        return_dict = not isinstance(output, tuple)
        shape = (tt, th, tw)
        if return_dict:
            assert not output_features, "output_feature is not compatible with return_dict"
            sample = output["x"]
            sample = all_gather(sample, dim=split_dim)
            output["x"] = sample
        else:
            sample = output[0]
            sample = all_gather(sample, dim=split_dim)
            if output_features:
                features_list = output[1]
                features_list = all_gather(features_list, dim=split_dim)
            else:
                features_list = None

            output = (sample, features_list, shape)
        return output

    new_forward = new_forward.__get__(model)
    model.forward = new_forward


def all_reduce_tensor_item(item):
    world_size = int(os.environ["WORLD_SIZE"])
    item = item.detach().clone()
    dist.all_reduce(item, op=dist.ReduceOp.SUM)
    item = item / nccl_info.ts_group_size if get_teacher_student_parallel_state() else item / world_size
    return item

def broadcast_item(item, idx):
    item_list = [item]
    dist.broadcast_object_list(item_list, src=idx)
    return item_list[0]
