"""Module docstring."""

import pytest
from onnx9000.core.ir import Graph, Tensor
from onnx9000.core.dtypes import DType
from onnx9000.onnx2gguf.arch import extract_metadata, infer_architecture


def test_arch_inference():
    """Provides functional implementation."""
    for name in [
        "mistral",
        "mixtral",
        "phi2",
        "qwen2",
        "gemma",
        "starcoder",
        "falcon",
        "bloom",
        "stablelm",
        "command-r",
        "bert",
    ]:
        g = Graph(name)
        assert infer_architecture(g) == name

    g = Graph("unknown")
    g.add_tensor(Tensor("model.embed_tokens.weight", (1, 1), DType.FLOAT32))
    assert infer_architecture(g) == "unknown"

    with pytest.raises(ValueError):
        extract_metadata(g, "invalid_arch")

    g2 = Graph("mistral")
    meta = extract_metadata(g2)
    assert meta["mistral.attention.sliding_window"] == 4096

    g3 = Graph("mixtral")
    meta3 = extract_metadata(g3)
    assert meta3["mixtral.expert_count"] == 8

    g4 = Graph("gemma")
    meta4 = extract_metadata(g4)
    assert "gemma.attention.layer_norm_rms_epsilon" in meta4

    g5 = Graph("unknown_but_contains_llama")
    g5.add_tensor(Tensor("llama", (1,), DType.FLOAT32))
    assert infer_architecture(g5) == "llama"

    assert extract_metadata(Graph("unknown")) == {}
