"""Module docstring."""

import io
from onnx9000.core.ir import Graph, Tensor
from onnx9000.core.dtypes import DType
from onnx9000.onnx2gguf.compiler import compile_gguf, infer_architecture, sanitize_doc_string


def test_compile_gguf():
    """Docstring."""
    g = Graph("test")
    g.producer_name = "test_producer"
    g.model_version = 1
    g.doc_string = "doc"

    t1 = Tensor("model.layers.0.weight", (2, 2), DType.FLOAT32, data=bytearray(b"1234567812345678"))
    t2 = Tensor("model.layers.0.bias", (2,), DType.FLOAT16, data=bytearray(b"1234"))

    g.add_tensor(t1)
    g.add_tensor(t2)
    g.initializers.append("model.layers.0.weight")
    g.initializers.append("model.layers.0.bias")

    out = io.BytesIO()
    compile_gguf(
        g,
        out,
        {
            "general.alignment": 64,
            "custom.bool": True,
            "custom.float": 1.0,
            "custom.int": 1,
            "custom.str": "str",
            "general.float": 1.0,
            "general.bool": False,
            "tensorNameOverrides": {
                "^model\\.layers\\.0\\.weight$": "mapped.weight",
                "^model\\.layers\\.0\\.bias$": "mapped.bias",
            },
        },
    )
    assert len(out.getvalue()) > 0


def test_compile_gguf_llama():
    """Docstring."""
    from onnx9000.core.dtypes import DType
    from onnx9000.onnx2gguf.compiler import get_gguf_type

    get_gguf_type(DType.FLOAT32.value)
    get_gguf_type(DType.FLOAT16.value)
    get_gguf_type(-1)
    g = Graph("llama")

    t3 = Tensor("x", (2,), DType.INT64, data=bytearray(b"12341234"))
    g.add_tensor(t3)
    g.initializers.append("x")

    out = io.BytesIO()
    compile_gguf(g, out)
    assert len(out.getvalue()) > 0


def test_compiler_helpers():
    """Docstring."""
    g = Graph("mock")
    assert infer_architecture(g) == "unknown"
    g.name = "llama_model"
    assert infer_architecture(g) == "llama"
    assert sanitize_doc_string("") == ""
    assert sanitize_doc_string(" a ") == "a"
