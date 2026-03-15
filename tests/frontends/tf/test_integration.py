import pytest
import time
from onnx9000.frontends.tf.api import (
    convert_tf_to_onnx,
    convert_keras_to_onnx,
    convert_tflite_to_onnx,
)


def _build_mock_tf_mnist() -> bytes:
    return b"\n\x07input\x12\x0bPlaceholder\n\x0bweight1\x12\x05Const\n\tbias1\x12\x05Const\n\rmatmul1\x12\x06MatMul\x1a\x05input\x1a\x07weight1\n\radd1\x12\x07BiasAdd\x1a\x07matmul1\x1a\x05bias1\n\nrelu1\x12\x04Relu\x1a\x04add1\n\x0bweight2\x12\x05Const\n\tbias2\x12\x05Const\n\rmatmul2\x12\x06MatMul\x1a\x05relu1\x1a\x07weight2\n\radd2\x12\x07BiasAdd\x1a\x07matmul2\x1a\x05bias2\n\rsoftmax\x12\x07Softmax\x1a\x04add2"


def test_integration_tf_mnist():
    from onnx9000.frontends.tf.parsers import TFGraph, TFNode
    from onnx9000.frontends.tf.api import _convert_tfgraph

    tf_graph = TFGraph(
        [
            TFNode("input", "Placeholder"),
            TFNode("weight1", "Const"),
            TFNode("bias1", "Const"),
            TFNode("matmul1", "MatMul", inputs=["input", "weight1"]),
            TFNode("add1", "BiasAdd", inputs=["matmul1", "bias1"]),
            TFNode("relu1", "Relu", inputs=["add1"]),
            TFNode("weight2", "Const"),
            TFNode("bias2", "Const"),
            TFNode("matmul2", "MatMul", inputs=["relu1", "weight2"]),
            TFNode("add2", "BiasAdd", inputs=["matmul2", "bias2"]),
            TFNode("softmax", "Softmax", inputs=["add2"]),
        ]
    )
    tf_graph.versions = {"producer": 15}
    graph = _convert_tfgraph(tf_graph)
    op_types = [n.op_type for n in graph.nodes]
    assert "MatMul" in op_types
    assert "Add" in op_types
    assert "Relu" in op_types
    assert "Softmax" in op_types


def test_integration_resnet_mobilenet_mock():
    from onnx9000.frontends.tf.parsers import TFGraph, TFNode
    from onnx9000.frontends.tf.api import _convert_tfgraph

    start_time = time.time()
    nodes = [TFNode("in", "Placeholder")]
    for i in range(50):
        nodes.append(
            TFNode(f"conv{i}", "Conv2D", inputs=[f"conv{i - 1}" if i > 0 else "in"])
        )
    tf_graph = TFGraph(nodes)
    graph = _convert_tfgraph(tf_graph)
    end_time = time.time()
    assert len(graph.nodes) >= 50
    assert end_time - start_time < 0.5


def test_integration_keras_sequential():
    graph = convert_keras_to_onnx(b"")
    assert graph.name == "keras_graph"


def test_integration_keras_functional():
    graph = convert_keras_to_onnx(b"", is_v3=True)
    assert graph.name == "keras_graph"


def test_integration_tflite_quantized():
    from onnx9000.frontends.tf.parsers import TFGraph, TFNode
    from onnx9000.frontends.tf.api import _convert_tfgraph

    tf_graph = TFGraph(
        [
            TFNode("in", "Placeholder"),
            TFNode("quant", "QUANTIZE", inputs=["in"]),
            TFNode("conv", "CONV_2D", inputs=["quant"]),
            TFNode("dequant", "DEQUANTIZE", inputs=["conv"]),
        ]
    )
    graph = _convert_tfgraph(tf_graph)
    op_types = [n.op_type for n in graph.nodes]
    assert "QuantizeLinear" in op_types
    assert "DequantizeLinear" in op_types
    assert "Custom_TFLiteConv2D" in op_types
