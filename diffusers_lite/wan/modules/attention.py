# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

import warnings

__all__ = [
    'flash_attention',
    'attention',
]


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out


# # @torch.compiler.disable
# def sequence_parallel_attention(q, k, v, img_q_len, img_kv_len, text_mask):
#     # 1GPU torch.Size([1, 11264, 24, 128]) tensor([    0, 11275, 11520], device='cuda:0', dtype=torch.int32)
#     # 2GPU torch.Size([1, 5632, 24, 128]) tensor([   0, 5643, 5888], device='cuda:0', dtype=torch.int32)
#     query, encoder_query = q
#     key, encoder_key = k
#     value, encoder_value = v
#     if get_sequence_parallel_state():
#         # batch_size, seq_len, attn_heads, head_dim
#         query = all_to_all_4D(query, scatter_dim=2, gather_dim=1)
#         key = all_to_all_4D(key, scatter_dim=2, gather_dim=1)
#         value = all_to_all_4D(value, scatter_dim=2, gather_dim=1)

#         def shrink_head(encoder_state, dim):
#             local_heads = encoder_state.shape[dim] // nccl_info.sp_size
#             return encoder_state.narrow(
#                 dim, nccl_info.rank_within_group * local_heads, local_heads
#             )

#         encoder_query = shrink_head(encoder_query, dim=2)
#         encoder_key = shrink_head(encoder_key, dim=2)
#         encoder_value = shrink_head(encoder_value, dim=2)
#         # [b, s, h, d]

#     sequence_length = query.size(1)
#     encoder_sequence_length = encoder_query.size(1)

#     # Hint: please check encoder_query.shape
#     query = torch.cat([query, encoder_query], dim=1)
#     key = torch.cat([key, encoder_key], dim=1)
#     value = torch.cat([value, encoder_value], dim=1)
#     # B, S, 3, H, D
#     qkv = torch.stack([query, key, value], dim=2)

#     attn_mask = F.pad(text_mask, (sequence_length, 0), value=True)
#     hidden_states = flash_attn_no_pad(
#         qkv, attn_mask, causal=False, dropout_p=0.0, softmax_scale=None
#     )

#     hidden_states, encoder_hidden_states = hidden_states.split_with_sizes(
#         (sequence_length, encoder_sequence_length), dim=1
#     )
#     if get_sequence_parallel_state():
#         hidden_states = all_to_all_4D(hidden_states, scatter_dim=1, gather_dim=2)
#         encoder_hidden_states = all_gather(encoder_hidden_states, dim=2).contiguous()
#     hidden_states = hidden_states.to(query.dtype)
#     encoder_hidden_states = encoder_hidden_states.to(query.dtype)

#     attn = torch.cat([hidden_states, encoder_hidden_states], dim=1)

#     b, s, a, d = attn.shape
#     attn = attn.reshape(b, s, -1)

#     return attn
