import typing as t

import torch
from xformers.ops import LowerTriangularMask, memory_efficient_attention


def gpt2_wrapped_scaled_dot_product(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: t.Optional[torch.Tensor] = None,
    head_mask: t.Optional[torch.Tensor] = None,
):
    assert head_mask is None
    batch_size = query.shape[0]

    mask_value = torch.finfo(value.dtype).min
    mask_value = torch.full([], mask_value, dtype=value.dtype)

    # in gpt-neo-x and gpt-j the query and keys are always in fp32
    # thus we need to cast them to the value dtype
    self.downcast_qk = True
    self.dropout_prob_attn = 0.0

    if self.downcast_qk:
        query = query.to(value.dtype)
        key = key.to(value.dtype)

    if batch_size == 1 and attention_mask is not None and attention_mask[0, 0, 0, -1] < -1:
        raise ValueError("BetterTransformer does not support padding='max_length' with a batch size of 1.")

    query = query.permute(0, 2, 1, 3).contiguous()
    key = key.permute(0, 2, 1, 3).contiguous()
    value = value.permute(0, 2, 1, 3).contiguous()

    dropout_p = self.dropout_prob_attn if self.training else 0.0
    if batch_size == 1 or self.training:
        if query.shape[2] > 1:
            sdpa_result = memory_efficient_attention(query, key, value, attn_bias=LowerTriangularMask())
            # sdpa_result = torch.nn.functional.scaled_dot_product_attention(
            #     query, key, value, attn_mask=None, dropout_p=dropout_p, is_causal=True
            # )
        else:
            sdpa_result = memory_efficient_attention(query, key, value, attn_bias=None)
            # sdpa_result = torch.nn.functional.scaled_dot_product_attention(
            #     query, key, value, attn_mask=None, dropout_p=dropout_p, is_causal=False
            # )
    else:
        assert False, "xformers inference with bsz > 1 not implemented"
        query_length, key_length = query.size(-2), key.size(-2)

        # causal_mask is always [True, ..., True] otherwise, so executing this
        # is unnecessary
        if query_length > 1:
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].to(torch.bool)

            causal_mask = torch.where(causal_mask, 0, mask_value)

            # torch.Tensor.expand does no memory copy
            causal_mask = causal_mask.expand(batch_size, -1, -1, -1)
            attention_mask = causal_mask + attention_mask

        sdpa_result = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=dropout_p, is_causal=False
        )

    # in gpt-neo-x and gpt-j the query and keys are always in fp32
    # thus we need to cast them to the value dtype
    if self.downcast_qk:
        sdpa_result = sdpa_result.to(value.dtype)

    return sdpa_result, None

def gpt_merge_heads(_self, tensor, num_attention_heads, attn_head_size):
    # -> [bs, seq_len, num_attention_heads, attn_head_size]
    tensor = tensor.view(tensor.size(0), tensor.size(1), num_attention_heads * attn_head_size)
    # -> [bs, seq_len, hidden_size]
    return tensor
