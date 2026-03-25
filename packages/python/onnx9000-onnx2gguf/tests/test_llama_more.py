from onnx9000.core.ir import Graph
from onnx9000.onnx2gguf.llama import extract_llama_metadata
import io


def test_convert_llama_not_tensor():
    g = Graph("test")
    g.tensors["fake"] = "not a tensor"
    g.tensors["embed_tokens.weight"] = "not a tensor"
    g.tensors["layers.0.self_attn.q_proj.weight"] = "not a tensor"
    m = extract_llama_metadata(g)
    assert m
