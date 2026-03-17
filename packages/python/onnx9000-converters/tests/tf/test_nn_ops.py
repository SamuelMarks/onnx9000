"""Tests the nn ops module functionality."""

from onnx9000.converters.tf.builder import TFToONNXGraphBuilder
from onnx9000.converters.tf.nn_ops import NN_OPS_MAPPING
from onnx9000.converters.tf.parsers import TFNode


def test_nn_ops_mapping_simple() -> None:
    """Tests the nn ops mapping simple functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n1", "MatMul", inputs=["a", "b"])
    outs = NN_OPS_MAPPING["MatMul"](builder, node)
    assert len(outs) == 1
    assert builder.graph.nodes[-1].op_type == "MatMul"
    node = TFNode("n2", "Relu", inputs=["a"])
    outs = NN_OPS_MAPPING["Relu"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Relu"


def test_nn_ops_relu6() -> None:
    """Tests the nn ops relu6 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "Relu6", inputs=["x"])
    NN_OPS_MAPPING["Relu6"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Clip"
    assert len(builder.graph.nodes[-1].inputs) == 3


def test_nn_ops_leaky_relu() -> None:
    """Tests the nn ops leaky relu functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "LeakyRelu", inputs=["x"], attr={"alpha": 0.3})
    NN_OPS_MAPPING["LeakyRelu"](builder, node)
    assert builder.graph.nodes[-1].op_type == "LeakyRelu"
    assert builder.graph.nodes[-1].attributes["alpha"] == 0.3


def test_nn_ops_elu() -> None:
    """Tests the nn ops elu functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "Elu", inputs=["x"], attr={"alpha": 2.0})
    NN_OPS_MAPPING["Elu"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Elu"
    assert builder.graph.nodes[-1].attributes["alpha"] == 2.0


def test_nn_ops_selu() -> None:
    """Tests the nn ops selu functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "Selu", inputs=["x"])
    NN_OPS_MAPPING["Selu"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Selu"


def test_nn_ops_softplus() -> None:
    """Tests the nn ops softplus functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "Softplus", inputs=["x"])
    NN_OPS_MAPPING["Softplus"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Softplus"


def test_nn_ops_softsign() -> None:
    """Tests the nn ops softsign functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "Softsign", inputs=["x"])
    NN_OPS_MAPPING["Softsign"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Softsign"


def test_nn_ops_softmax() -> None:
    """Tests the nn ops softmax functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "Softmax", inputs=["x"])
    NN_OPS_MAPPING["Softmax"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Softmax"
    assert builder.graph.nodes[-1].attributes["axis"] == -1


def test_nn_ops_log_softmax() -> None:
    """Tests the nn ops log softmax functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "LogSoftmax", inputs=["x"], attr={"axis": 1})
    NN_OPS_MAPPING["LogSoftmax"](builder, node)
    assert builder.graph.nodes[-1].op_type == "LogSoftmax"
    assert builder.graph.nodes[-1].attributes["axis"] == 1


def test_nn_ops_conv2d_nhwc() -> None:
    """Tests the nn ops conv2d nhwc functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode(
        "n",
        "Conv2D",
        inputs=["input", "weight"],
        attr={"data_format": b"NHWC", "strides": [1, 2, 2, 1], "dilations": [1, 1, 1, 1]},
    )
    NN_OPS_MAPPING["Conv2D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Transpose"
    conv_node = builder.graph.nodes[-2]
    assert conv_node.op_type == "Conv"
    assert conv_node.attributes["strides"] == [2, 2]


def test_nn_ops_conv2d_nchw() -> None:
    """Tests the nn ops conv2d nchw functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode(
        "n",
        "Conv2D",
        inputs=["input", "weight"],
        attr={"data_format": "NCHW", "strides": [1, 1, 2, 2], "dilations": [1, 1, 1, 1]},
    )
    NN_OPS_MAPPING["Conv2D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Conv"
    assert builder.graph.nodes[-1].attributes["strides"] == [2, 2]
    w_trans_node = builder.graph.nodes[-2]
    assert w_trans_node.op_type == "Transpose"
    assert w_trans_node.attributes["perm"] == [3, 2, 0, 1]


def test_nn_ops_conv3d() -> None:
    """Tests the nn ops conv3d functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "Conv3D", inputs=["input", "weight"])
    NN_OPS_MAPPING["Conv3D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Conv"


def test_nn_ops_poolings() -> None:
    """Tests the nn ops poolings functionality."""
    builder = TFToONNXGraphBuilder()
    for op in ["MaxPool", "MaxPoolV2", "MaxPool3D"]:
        NN_OPS_MAPPING[op](builder, TFNode(f"n_{op}", op, inputs=["x"]))
        assert builder.graph.nodes[-1].op_type == "MaxPool"
    for op in ["AvgPool", "AvgPool3D"]:
        NN_OPS_MAPPING[op](builder, TFNode(f"n_{op}", op, inputs=["x"]))
        assert builder.graph.nodes[-1].op_type == "AveragePool"
    NN_OPS_MAPPING["GlobalMaxPool"](builder, TFNode("n", "GlobalMaxPool", inputs=["x"]))
    assert builder.graph.nodes[-1].op_type == "GlobalMaxPool"
    NN_OPS_MAPPING["GlobalAvgPool"](builder, TFNode("n", "GlobalAvgPool", inputs=["x"]))
    assert builder.graph.nodes[-1].op_type == "GlobalAveragePool"


def test_nn_ops_batch_norm() -> None:
    """Tests the nn ops batch norm functionality."""
    builder = TFToONNXGraphBuilder()
    for op in [
        "BatchNormWithGlobalNormalization",
        "FusedBatchNorm",
        "FusedBatchNormV2",
        "FusedBatchNormV3",
    ]:
        node = TFNode(
            f"n_{op}", op, inputs=["x", "scale", "bias", "mean", "var"], attr={"epsilon": 0.0001}
        )
        NN_OPS_MAPPING[op](builder, node)
        assert builder.graph.nodes[-1].op_type == "BatchNormalization"
        assert builder.graph.nodes[-1].attributes["epsilon"] == 0.0001


def test_nn_ops_customs_and_misc() -> None:
    """Tests the nn ops customs and misc functionality."""
    builder = TFToONNXGraphBuilder()
    mapping_checks = {
        "FractionalMaxPool": "Custom_FractionalMaxPool",
        "FractionalAvgPool": "Custom_FractionalAvgPool",
        "L2Loss": "Custom_L2Loss",
        "LocalResponseNormalization": "LRN",
        "Dropout": "Dropout",
        "TopK": "TopK",
        "TopKV2": "TopK",
        "InTopK": "Custom_InTopK",
        "InTopKV2": "Custom_InTopK",
        "NthElement": "Custom_NthElement",
        "DepthwiseConv2dNative": "Conv",
        "Conv2DBackpropInput": "ConvTranspose",
        "Conv3DBackpropInputV2": "ConvTranspose",
    }
    for tf_op, onnx_op in mapping_checks.items():
        node = TFNode(f"n_{tf_op}", tf_op, inputs=["x"])
        NN_OPS_MAPPING[tf_op](builder, node)
        assert builder.graph.nodes[-1].op_type == onnx_op
