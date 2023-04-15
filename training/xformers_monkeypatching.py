import typing as t

import torch
import transformers

from torch import nn
from transformers.models.gptj.modeling_gptj import apply_rotary_pos_emb as gptj_apply_rotary_pos_emb
from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb as gptneox_apply_rotary_pos_emb
from transformers.utils import is_torch_fx_proxy
from xformers.ops import memory_efficient_attention
from xformers.components.positional_embedding import RotaryEmbedding

def gptj_forward(
        self,
        hidden_states: torch.FloatTensor,
        layer_past: t.Optional[t.Tuple[torch.Tensor]] = None,
        attention_mask: t.Optional[torch.FloatTensor] = None,
        position_ids: t.Optional[torch.LongTensor] = None,
        head_mask: t.Optional[torch.FloatTensor] = None,
        use_cache: t.Optional[bool] = False,
        output_attentions: t.Optional[bool] = False,
    ) -> t.Union[
        t.Tuple[torch.Tensor, t.Tuple[torch.Tensor]],
        t.Optional[t.Tuple[torch.Tensor, t.Tuple[torch.Tensor], t.Tuple[torch.Tensor, ...]]],
    ]:
    query = self.q_proj(hidden_states)
    key = self.k_proj(hidden_states)
    value = self.v_proj(hidden_states)

    query = self._split_heads(query, self.num_attention_heads, self.head_dim, True)
    key = self._split_heads(key, self.num_attention_heads, self.head_dim, True)
    value = self._split_heads(value, self.num_attention_heads, self.head_dim, False)

    if is_torch_fx_proxy(position_ids):
        # The logic to conditionally copy to GPU could not be traced, so we do this
        # every time in the torch.fx case
        embed_positions = get_embed_positions(self.embed_positions, position_ids)
    else:
        embed_positions = self._get_embed_positions(position_ids)

    repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)

    if self.rotary_dim is not None:
        k_rot = key[:, :, :, : self.rotary_dim]
        k_pass = key[:, :, :, self.rotary_dim :]

        q_rot = query[:, :, :, : self.rotary_dim]
        q_pass = query[:, :, :, self.rotary_dim :]

        k_rot = gptj_apply_rotary_pos_emb(k_rot, sin, cos)
        q_rot = gptj_apply_rotary_pos_emb(q_rot, sin, cos)

        key = torch.cat([k_rot, k_pass], dim=-1)
        query = torch.cat([q_rot, q_pass], dim=-1)
    else:
        key = gptj_apply_rotary_pos_emb(key, sin, cos)
        query = gptj_apply_rotary_pos_emb(query, sin, cos)

    key = key.permute(0, 2, 1, 3)
    query = query.permute(0, 2, 1, 3)

    # Upcast value to fp32 so that xformers doesn't error out
    if value.dtype != torch.float32:
        value = value.to(torch.float32)

    if layer_past is not None:
        past_key = layer_past[0]
        past_value = layer_past[1]
        key = torch.cat((past_key, key), dim=-2)
        value = torch.cat((past_value, value), dim=-2)

    if use_cache is True:
        present = (key, value)
    else:
        present = None

    # compute self-attention: V x Softmax(QK^T)
    # Casting required for xformers to not error out
    # but also shows up in normal GPT-J _attn fn
    # query = query.to(dtype=torch.float32)
    # key = key.to(dtype=torch.float32)
    attn_output = memory_efficient_attention(query, key, value, attn_bias=attention_mask)

    attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_dim)
    attn_output = self.out_proj(attn_output)
    attn_output = self.resid_dropout(attn_output)

    outputs = (attn_output, present)
    # if output_attentions:
    #     outputs += (attn_weights,)

    return outputs  # a, present, (attentions)

def gptneox_forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        position_ids: torch.LongTensor,
        head_mask: t.Optional[torch.FloatTensor] = None,
        layer_past: t.Optional[t.Tuple[torch.Tensor]] = None,
        use_cache: t.Optional[bool] = False,
        output_attentions: t.Optional[bool] = False,
    ):
    has_layer_past = layer_past is not None

    # Compute QKV
    # Attention heads [batch, seq_len, hidden_size]
    #   --> [batch, seq_len, (np * 3 * head_size)]
    qkv = self.query_key_value(hidden_states)

    # [batch, seq_len, (num_heads * 3 * head_size)]
    #   --> [batch, seq_len, num_heads, 3 * head_size]
    new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_size)
    qkv = qkv.view(*new_qkv_shape)

    # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
    query = qkv[..., : self.head_size].permute(0, 2, 1, 3)
    key = qkv[..., self.head_size : 2 * self.head_size].permute(0, 2, 1, 3)
    value = qkv[..., 2 * self.head_size :].permute(0, 2, 1, 3)

    # Compute rotary embeddings on rotary_ndims
    query_rot = query[..., : self.rotary_ndims]
    query_pass = query[..., self.rotary_ndims :]
    key_rot = key[..., : self.rotary_ndims]
    key_pass = key[..., self.rotary_ndims :]

    # Compute token offset for rotary embeddings (when decoding)
    seq_len = key.shape[-2]
    if has_layer_past:
        seq_len += layer_past[0].shape[-2]
        
    cos, sin = self.rotary_emb(value, seq_len=seq_len)
    query, key = gptneox_apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)
    query = torch.cat((query, query_pass), dim=-1)
    key = torch.cat((key, key_pass), dim=-1)

    # Upcast value to fp32 so that xformers doesn't error out
    if value.dtype != torch.float32:
        value = value.to(torch.float32)

    # Cache QKV values
    if has_layer_past:
        past_key = layer_past[0]
        past_value = layer_past[1]
        key = torch.cat((past_key, key), dim=-2)
        value = torch.cat((past_value, value), dim=-2)
    present = (key, value) if use_cache else None
    
    # Permute tensors so that it fits into xformers MHA
    query = query.permute(0, 2, 1, 3).contiguous()
    key = key.permute(0, 2, 1, 3).contiguous()
    value = value.permute(0, 2, 1, 3).contiguous()
    
    # Compute attention
    attn_output = memory_efficient_attention(query, key, value, attn_bias=attention_mask)

    # Reshape outputs
    attn_output = attn_output.view(attn_output.size(0), attn_output.size(1), self.num_attention_heads * self.head_size)
    attn_output = self.dense(attn_output)

    outputs = (attn_output, present)
    if output_attentions:
        outputs += (attn_weights,)

    return outputs

def llama_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: t.Optional[torch.Tensor] = None,
        position_ids: t.Optional[torch.LongTensor] = None,
        past_key_value: t.Optional[t.Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> t.Tuple[torch.Tensor, t.Optional[torch.Tensor], t.Optional[t.Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = gptneox_apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    # [bsz, nh, t, hd]

    # Upcast value to fp32 so that xformers doesn't error out
    if value_states.dtype != torch.float32:
        value_states = value_states.to(torch.float32)

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    #if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
    #    raise ValueError(
    #        f"Attention weights should be of size {(bsz * self.num_heads, q_len, kv_seq_len)}, but is"
    #        f" {attn_weights.size()}"
    #    )

    #if attention_mask is not None:
    #    if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
    #        raise ValueError(
    #            f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
    #        )
    #    attn_weights = attn_weights + attention_mask
    #    attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

    # upcast attention to fp32
    #attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    #attn_output = torch.matmul(attn_weights, value_states)

    #if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
    #    raise ValueError(
    #        f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
    #        f" {attn_output.size()}"
    #    )
            
    attn_output = memory_efficient_attention(query_states, key_states, value_states, attn_bias=attention_mask)

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def apply_xformers_to_model(model: t.Any) -> None:
    # This feels illegal to do in Python, but it works
    MODEL_FORWARD_REPLACEMENTS = {
        transformers.GPTJForCausalLM: apply_xformers_to_gptj,
        transformers.GPTNeoXForCausalLM: apply_xformers_to_gptneox,
        transformers.LlamaForCausalLM: apply_xformers_to_llama,
    }
    MODEL_FORWARD_REPLACEMENTS[type(model)]()
    
def apply_xformers_to_gptj() -> None:
    transformers.models.gptj.modeling_gptj.GPTJAttention.forward = gptj_forward
    
def apply_xformers_to_gptneox() -> None:
    transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXAttention.forward = gptneox_forward
    
def apply_xformers_to_llama() -> None:
    transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_forward
