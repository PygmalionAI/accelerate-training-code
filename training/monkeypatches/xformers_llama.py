import typing as t

import torch
import transformers
from xformers.ops import memory_efficient_attention


def llama_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: t.Optional[torch.Tensor] = None,
    position_ids: t.Optional[torch.LongTensor] = None,
    past_key_value: t.Optional[t.Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> t.Tuple[torch.Tensor, t.Optional[torch.Tensor], t.Optional[t.Tuple[torch.Tensor]]]:
    assert not output_attentions, "xformers cannot be used when output_attentions = True"
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = transformers.models.llama.modeling_llama.apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    # [bsz, nh, t, hd]

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    dtype = query_states.dtype

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    # This is a nasty hack. We know attention_mask in transformers is either
    # LowerTriangular or all Zeros. We therefore check if one element in the
    # upper triangular portion is zero. If it is, then the mask is all zeros.
    if attention_mask is None or attention_mask[0, 0, 0, 1] == 0:
        # input and output should be of form (bsz, q_len, num_heads, head_dim)
        attn_output = memory_efficient_attention(query_states, key_states, value_states, attn_bias=None)
    else:
        # input and output should be of form (bsz, q_len, num_heads, head_dim)
        from xformers.ops import LowerTriangularMask
        attn_output = memory_efficient_attention(query_states, key_states, value_states, attn_bias=LowerTriangularMask())
    attn_weights = None

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    return attn_output, attn_weights, past_key_value
