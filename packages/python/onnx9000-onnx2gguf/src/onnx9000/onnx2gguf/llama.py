import re
from typing import Any
from onnx9000.core.ir import Graph, Tensor


def extract_llama_metadata(graph: Graph) -> dict[str, Any]:
    meta = {}

    vocab_size = 32000
    hidden_size = 4096
    num_heads = 32
    num_kv_heads = 32
    intermediate_size = 11008
    rms_eps = 1e-5
    head_dim = hidden_size // num_heads

    # Try to find tensors
    for name, t in graph.tensors.items():
        if not isinstance(t, Tensor):
            continue
        if name.endswith("embed_tokens.weight"):
            if len(t.shape) == 2:
                vocab_size, hidden_size = t.shape
        elif name.endswith("layers.0.self_attn.q_proj.weight"):
            if len(t.shape) == 2:
                num_heads = t.shape[0] // head_dim
        elif name.endswith("layers.0.self_attn.k_proj.weight"):
            if len(t.shape) == 2:
                num_kv_heads = t.shape[0] // head_dim
        elif name.endswith("layers.0.mlp.up_proj.weight"):
            if len(t.shape) == 2:
                intermediate_size = t.shape[0]

    # Detect block count
    layers = set()
    for name in graph.tensors.keys():
        match = re.search(r"model\.layers\.(\d+)", name)
        if match:
            layers.add(int(match.group(1)))

    block_count = max(layers) + 1 if layers else 32

    # Detect RMS norm eps from nodes if possible, or just default
    for n in graph.nodes:
        if n.op_type == "RMSNormalization":
            eps = n.attributes.get("epsilon", 1e-5)
            rms_eps = float(eps)
            break

    # Grouped Query Attention ratio
    # Already captured by num_heads and num_kv_heads

    # SwiGLU / GeGLU detection
    is_swiglu = False
    for n in graph.nodes:
        if n.op_type == "Swish" or n.op_type == "Silu":
            is_swiglu = True
            break

    meta["llama.context_length"] = (
        2048  # Usually embedded in config, hard to guess from weights alone
    )
    meta["llama.embedding_length"] = hidden_size
    meta["llama.block_count"] = block_count
    meta["llama.feed_forward_length"] = intermediate_size
    meta["llama.attention.head_count"] = num_heads
    meta["llama.attention.head_count_kv"] = num_kv_heads
    meta["llama.attention.layer_norm_rms_epsilon"] = rms_eps
    meta["llama.rope.dimension_count"] = head_dim
    meta["llama.rope.freq_base"] = 10000.0
    meta["llama.vocab_size"] = vocab_size
    meta["custom.is_swiglu"] = is_swiglu

    return meta
