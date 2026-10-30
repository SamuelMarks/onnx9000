"""Comprehensive unit tests for onnx9000.tflite_converter and onnx9000_tflite_converter."""

import onnx9000_tflite_converter
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor, ValueInfo
from onnx9000.tflite_converter import TFLiteConverter, run


def test_tflite_converter_and_run() -> None:
    """Verify run() and TFLiteConverter.convert() on an ONNX9000 Graph."""
    assert run() == "[tflite-converter] processed"
    assert onnx9000_tflite_converter.run() == "[tflite-converter] processed"

    g = Graph("tflite_conv")
    g.add_tensor(Tensor("X", (1, 4), DType.FLOAT32))
    g.add_tensor(Tensor("Y", (1, 4), DType.FLOAT32))
    g.inputs.append(ValueInfo("X", (1, 4), DType.FLOAT32))
    g.outputs.append(ValueInfo("Y", (1, 4), DType.FLOAT32))
    g.add_node(Node("Relu", ["X"], ["Y"]))

    converter = TFLiteConverter(quant_mode="none")
    buf = converter.convert(g)
    assert isinstance(buf, bytes)
    assert len(buf) > 0
