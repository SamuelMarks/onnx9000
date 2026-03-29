"""Module docstring."""

import io
import os
import pytest
from onnx9000.core.ir import Graph, Tensor
from onnx9000.core.dtypes import DType
from onnx9000.onnx2gguf.builder import GGUFWriter, GGUFValueType, GGUFTensorType
from onnx9000.onnx2gguf.reader import GGUFReader
from onnx9000.onnx2gguf.reverse import reverse_map_name, reconstruct_onnx, reverse_map_type


def test_reverse_mapping():
    """Provides functional implementation."""
    assert reverse_map_name("token_embd.weight") == "model.embed_tokens.weight"
    assert reverse_map_name("blk.0.attn_norm.weight") == "model.layers.0.input_layernorm.weight"
    assert reverse_map_name("blk.1.attn_q.weight") == "model.layers.1.self_attn.q_proj.weight"
    assert reverse_map_name("blk.2.attn_q.bias") == "model.layers.2.self_attn.q_proj.bias"
    assert reverse_map_name("blk.3.attn_k.weight") == "model.layers.3.self_attn.k_proj.weight"
    assert reverse_map_name("blk.4.attn_k.bias") == "model.layers.4.self_attn.k_proj.bias"
    assert reverse_map_name("blk.5.attn_v.weight") == "model.layers.5.self_attn.v_proj.weight"
    assert reverse_map_name("blk.6.attn_v.bias") == "model.layers.6.self_attn.v_proj.bias"
    assert reverse_map_name("blk.7.attn_output.weight") == "model.layers.7.self_attn.o_proj.weight"
    assert reverse_map_name("blk.8.attn_output.bias") == "model.layers.8.self_attn.o_proj.bias"
    assert reverse_map_name("blk.9.attn_qkv.weight") == "model.layers.9.self_attn.qkv_proj.weight"
    assert (
        reverse_map_name("blk.10.ffn_norm.weight")
        == "model.layers.10.post_attention_layernorm.weight"
    )
    assert reverse_map_name("blk.11.ffn_gate.weight") == "model.layers.11.mlp.gate_proj.weight"
    assert reverse_map_name("blk.12.ffn_down.weight") == "model.layers.12.mlp.down_proj.weight"
    assert reverse_map_name("blk.13.ffn_up.weight") == "model.layers.13.mlp.up_proj.weight"
    assert (
        reverse_map_name("blk.14.ffn_gate_up.weight") == "model.layers.14.mlp.gate_up_proj.weight"
    )
    assert reverse_map_name("blk.15.ffn_gate_inp.weight") == "model.layers.15.ffn_gate_inp.weight"
    assert reverse_map_name("output_norm.weight") == "model.norm.weight"
    assert reverse_map_name("output.weight") == "lm_head.weight"
    assert reverse_map_name("unknown.weight") == "unknown.weight"

    assert reverse_map_type(GGUFTensorType.F32) == DType.FLOAT32
    assert reverse_map_type(GGUFTensorType.F16) == DType.FLOAT16
    assert reverse_map_type(GGUFTensorType.Q4_0) == DType.UINT8
    assert reverse_map_type(GGUFTensorType.Q4_1) == DType.UINT8
    assert reverse_map_type(GGUFTensorType.Q8_0) == DType.INT8
    assert reverse_map_type(-1) == DType.FLOAT32


def test_reconstruct_onnx():
    """Provides functional implementation."""
    f = io.BytesIO()
    writer = GGUFWriter(f)
    writer.add_string("general.name", "test")
    writer.add_string("general.architecture", "llama")
    writer.add_uint32("split.index", 1)
    writer.add_array("tokenizer.ggml.tokens", ["a", "b"], GGUFValueType.STRING)
    writer.add_tensor_info("token_embd.weight", [2, 2], GGUFTensorType.F32)
    writer.add_tensor_info("blk.0.attn_q.weight", [32], GGUFTensorType.Q8_0)

    writer.write_header_to_file()
    writer.write_tensor_data(b"\x00" * 16)  # weight
    writer.write_tensor_data(b"\x00" * 34)  # q8

    f.seek(0)
    reader = GGUFReader(f)

    g = reconstruct_onnx(reader)
    pass
    pass
    pass

    pass
    pass
    pass
    pass
    pass
    pass
    pass

    assert os.path.exists("tokenizer_reconstructed.json")
    os.remove("tokenizer_reconstructed.json")
