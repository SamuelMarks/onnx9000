import pytest
from onnx9000.frontends.tf.builder import TFToONNXGraphBuilder
from onnx9000.frontends.tf.parsers import TFNode
from onnx9000.frontends.tf.tflite_ops import TFLITE_OPS_MAPPING


def test_tflite_ops_simple():
    builder = TFToONNXGraphBuilder()
    outs = TFLITE_OPS_MAPPING["ADD"](builder, TFNode("n1", "ADD", inputs=["a", "b"]))
    assert builder.graph.nodes[-1].op_type == "Add"
    outs = TFLITE_OPS_MAPPING["MUL"](builder, TFNode("n2", "MUL", inputs=["a", "b"]))
    assert builder.graph.nodes[-1].op_type == "Mul"


def test_tflite_ops_pools_and_convs():
    builder = TFToONNXGraphBuilder()
    for op in ["AVERAGE_POOL_2D", "MAX_POOL_2D", "L2_POOL_2D"]:
        outs = TFLITE_OPS_MAPPING[op](builder, TFNode("n", op, inputs=["a"]))
        assert builder.graph.nodes[-1].op_type.startswith("Custom_TFLite")
    for op in ["CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"]:
        outs = TFLITE_OPS_MAPPING[op](builder, TFNode("n", op, inputs=["a"]))
        assert builder.graph.nodes[-1].op_type.startswith("Custom_TFLite")


def test_tflite_ops_tensor_misc():
    builder = TFToONNXGraphBuilder()
    outs = TFLITE_OPS_MAPPING["RESHAPE"](
        builder, TFNode("n", "RESHAPE", inputs=["a", "shape"])
    )
    assert builder.graph.nodes[-1].op_type == "Reshape"
    outs = TFLITE_OPS_MAPPING["RESIZE_BILINEAR"](
        builder, TFNode("n", "RESIZE_BILINEAR", inputs=["a", "size"])
    )
    assert builder.graph.nodes[-1].op_type == "Resize"
    assert builder.graph.nodes[-1].attributes["mode"] == "linear"
    outs = TFLITE_OPS_MAPPING["CONCATENATION"](
        builder, TFNode("n", "CONCATENATION", inputs=["a", "b"], attr={"axis": 1})
    )
    assert builder.graph.nodes[-1].op_type == "Concat"
    assert builder.graph.nodes[-1].attributes["axis"] == 1


def test_tflite_ops_activations():
    builder = TFToONNXGraphBuilder()
    outs = TFLITE_OPS_MAPPING["SOFTMAX"](builder, TFNode("n", "SOFTMAX", inputs=["a"]))
    assert builder.graph.nodes[-1].op_type == "Softmax"
    outs = TFLITE_OPS_MAPPING["LOGISTIC"](
        builder, TFNode("n", "LOGISTIC", inputs=["a"])
    )
    assert builder.graph.nodes[-1].op_type == "Sigmoid"
    outs = TFLITE_OPS_MAPPING["TANH"](builder, TFNode("n", "TANH", inputs=["a"]))
    assert builder.graph.nodes[-1].op_type == "Tanh"
    outs = TFLITE_OPS_MAPPING["RELU"](builder, TFNode("n", "RELU", inputs=["a"]))
    assert builder.graph.nodes[-1].op_type == "Relu"
    outs = TFLITE_OPS_MAPPING["RELU6"](builder, TFNode("n", "RELU6", inputs=["a"]))
    assert builder.graph.nodes[-1].op_type == "Clip"
    outs = TFLITE_OPS_MAPPING["RELU_N1_TO_1"](
        builder, TFNode("n", "RELU_N1_TO_1", inputs=["a"])
    )
    assert builder.graph.nodes[-1].op_type == "Clip"


def test_tflite_ops_quantization():
    builder = TFToONNXGraphBuilder()
    outs = TFLITE_OPS_MAPPING["DEQUANTIZE"](
        builder, TFNode("n", "DEQUANTIZE", inputs=["a"])
    )
    assert builder.graph.nodes[-1].op_type == "DequantizeLinear"
    outs = TFLITE_OPS_MAPPING["QUANTIZE"](
        builder, TFNode("n", "QUANTIZE", inputs=["a"])
    )
    assert builder.graph.nodes[-1].op_type == "QuantizeLinear"


def test_tflite_ops_others():
    builder = TFToONNXGraphBuilder()
    outs = TFLITE_OPS_MAPPING["EMBEDDING_LOOKUP"](
        builder, TFNode("n", "EMBEDDING_LOOKUP", inputs=["a", "b"])
    )
    assert builder.graph.nodes[-1].op_type == "Gather"
    outs = TFLITE_OPS_MAPPING["L2_NORMALIZATION"](
        builder, TFNode("n", "L2_NORMALIZATION", inputs=["a"])
    )
    assert builder.graph.nodes[-1].op_type == "LpNormalization"
    assert builder.graph.nodes[-1].attributes["p"] == 2
    outs = TFLITE_OPS_MAPPING["LOCAL_RESPONSE_NORMALIZATION"](
        builder, TFNode("n", "LOCAL_RESPONSE_NORMALIZATION", inputs=["a"])
    )
    assert builder.graph.nodes[-1].op_type == "LRN"
    outs = TFLITE_OPS_MAPPING["SPACE_TO_DEPTH"](
        builder, TFNode("n", "SPACE_TO_DEPTH", inputs=["a"])
    )
    assert builder.graph.nodes[-1].op_type == "SpaceToDepth"
    outs = TFLITE_OPS_MAPPING["DEPTH_TO_SPACE"](
        builder, TFNode("n", "DEPTH_TO_SPACE", inputs=["a"])
    )
    assert builder.graph.nodes[-1].op_type == "DepthToSpace"
    outs = TFLITE_OPS_MAPPING["FLOOR"](builder, TFNode("n", "FLOOR", inputs=["a"]))
    assert builder.graph.nodes[-1].op_type == "Floor"
    custom_mappings = {
        "HASHTABLE_LOOKUP": "HashtableLookup",
        "LSH_PROJECTION": "LshProjection",
        "LSTM": "LSTM",
        "RNN": "RNN",
        "SVDF": "SVDF",
        "CONCAT_EMBEDDINGS": "ConcatEmbeddings",
        "SKIP_GRAM": "SkipGram",
        "CALL": "Call",
        "CUSTOM": "Custom",
    }
    for op, target in custom_mappings.items():
        outs = TFLITE_OPS_MAPPING[op](builder, TFNode(f"n_{op}", op, inputs=["a"]))
        assert builder.graph.nodes[-1].op_type == f"Custom_TFLite{target}"
