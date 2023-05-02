import transformers
from monkeypatches.xformers_gpt import (gpt2_wrapped_scaled_dot_product,
                                        gpt_merge_heads)
from monkeypatches.xformers_llama import llama_attention_forward


def apply_xformers_monkeypatches() -> None:
    # LLaMA
    transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attention_forward

    # GPT-J
    transformers.models.gptj.modeling_gptj.GPTJAttention._attn = gpt2_wrapped_scaled_dot_product
    transformers.models.gptj.modeling_gptj.GPTJAttention._merge_heads = gpt_merge_heads

    # NeoX
    transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXAttention._attn = gpt2_wrapped_scaled_dot_product
    transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXAttention._merge_heads = gpt_merge_heads
