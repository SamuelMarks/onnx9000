"""Comprehensive unit tests for onnx9000.onnx2tf and onnx9000_onnx2tf."""

import onnx9000_onnx2tf
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor, ValueInfo
from onnx9000.onnx2tf import convert_onnx_to_tflite, run, transpose_nchw_to_nhwc_shape


def test_onnx2tf_transpose_and_export() -> None:
    """Verify run(), transpose_nchw_to_nhwc_shape(), and convert_onnx_to_tflite()."""
    assert run() == "[onnx2tf] processed"
    assert onnx9000_onnx2tf.run() == "[onnx2tf] processed"

    assert transpose_nchw_to_nhwc_shape((1, 3, 224, 224)) == (1, 224, 224, 3)
    assert transpose_nchw_to_nhwc_shape((1, 10)) == (1, 10)

    g = Graph("tf_export")
    g.add_tensor(Tensor("X", (1, 4), DType.FLOAT32))
    g.add_tensor(Tensor("Y", (1, 4), DType.FLOAT32))
    g.inputs.append(ValueInfo("X", (1, 4), DType.FLOAT32))
    g.outputs.append(ValueInfo("Y", (1, 4), DType.FLOAT32))
    g.add_node(Node("Relu", ["X"], ["Y"]))

    buf = convert_onnx_to_tflite(g)
    assert isinstance(buf, bytes)
    assert len(buf) > 0
