"""Comprehensive unit tests for onnx9000.keras2onnx and onnx9000_keras2onnx."""

import io
import json

import h5py
import onnx9000_keras2onnx
import pytest
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor, ValueInfo
from onnx9000.keras2onnx import Keras2OnnxConverter, convert


def test_convert_and_keras_bridge() -> None:
    """Verify convert(), Keras2OnnxConverter.to_onnx(), and to_keras_code()."""
    with pytest.raises(ValueError, match="Invalid model string"):
        convert("")
    assert convert("test") == "[ONNX-IR] from keras test"
    assert onnx9000_keras2onnx.convert("v3") == "[ONNX-IR] from keras v3"

    converter = Keras2OnnxConverter()
    buf = io.BytesIO()
    with h5py.File(buf, "w") as h5f:
        h5f.attrs["model_config"] = json.dumps(
            {"class_name": "Sequential", "config": {"layers": []}}
        )
    onnx_graph = converter.to_onnx(buf.getvalue(), is_v3=False)
    assert isinstance(onnx_graph, Graph)

    g = Graph("MiniKeras")
    x = Tensor("X", (1, 4), DType.FLOAT32)
    y = Tensor("Y", (1, 4), DType.FLOAT32)
    g.add_tensor(x)
    g.add_tensor(y)
    g.inputs.append(ValueInfo("X", (1, 4), DType.FLOAT32))
    g.outputs.append(ValueInfo("Y", (1, 4), DType.FLOAT32))
    g.add_node(Node("Relu", [x], [y]))

    code = converter.to_keras_code(g)
    assert "class Model_MiniKeras(keras.Model):" in code
    assert "ops.relu" in code
