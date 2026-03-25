import re

DEFAULT_MAPPING = [
    (r"^model\.embed_tokens\.weight$", "token_embd.weight"),
    (r"^model\.layers\.(\d+)\.input_layernorm\.weight$", r"blk.\1.attn_norm.weight"),
    (r"^model\.layers\.(\d+)\.self_attn\.q_proj\.weight$", r"blk.\1.attn_q.weight"),
    (r"^model\.layers\.(\d+)\.self_attn\.q_proj\.bias$", r"blk.\1.attn_q.bias"),
    (r"^model\.layers\.(\d+)\.self_attn\.k_proj\.weight$", r"blk.\1.attn_k.weight"),
    (r"^model\.layers\.(\d+)\.self_attn\.k_proj\.bias$", r"blk.\1.attn_k.bias"),
    (r"^model\.layers\.(\d+)\.self_attn\.v_proj\.weight$", r"blk.\1.attn_v.weight"),
    (r"^model\.layers\.(\d+)\.self_attn\.v_proj\.bias$", r"blk.\1.attn_v.bias"),
    (r"^model\.layers\.(\d+)\.self_attn\.o_proj\.weight$", r"blk.\1.attn_output.weight"),
    (r"^model\.layers\.(\d+)\.self_attn\.o_proj\.bias$", r"blk.\1.attn_output.bias"),
    (r"^model\.layers\.(\d+)\.self_attn\.qkv_proj\.weight$", r"blk.\1.attn_qkv.weight"),
    (r"^model\.layers\.(\d+)\.post_attention_layernorm\.weight$", r"blk.\1.ffn_norm.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.gate_proj\.weight$", r"blk.\1.ffn_gate.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.down_proj\.weight$", r"blk.\1.ffn_down.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.up_proj\.weight$", r"blk.\1.ffn_up.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.gate_up_proj\.weight$", r"blk.\1.ffn_gate_up.weight"),
    (r"^model\.layers\.(\d+)\.ffn_gate_inp\.weight$", r"blk.\1.ffn_gate_inp.weight"),
    (r"^model\.norm\.weight$", "output_norm.weight"),
    (r"^lm_head\.weight$", "output.weight"),
]


def rename_tensor(name: str, overrides: dict[str, str] = None) -> str:
    overrides = overrides or {}

    # Try overrides first
    for pattern, repl in overrides.items():
        if re.match(pattern, name):
            return re.sub(pattern, repl, name)

    # Default mappings
    for pattern, repl in DEFAULT_MAPPING:
        if re.match(pattern, name):
            return re.sub(pattern, repl, name)

    # Unknown tensors trigger a fallback warning (we log or throw based on strictness)
    # For now we'll just return the original name
    # 110. Fail cleanly and log unmatched tensors -> we return None or warning, but here we return original so we don't break non-LLM stuff
    # Wait, the spec says "Fail cleanly and log unmatched tensors if an ONNX model contains unknown weights".
    # I'll raise ValueError.
    raise ValueError(f"Unmatched tensor name: {name}")
