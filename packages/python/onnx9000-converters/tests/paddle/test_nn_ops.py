"""Tests the nn ops module functionality."""

from onnx9000.converters.paddle.builder import PaddleToONNXGraphBuilder
from onnx9000.converters.paddle.nn_ops import NN_OPS_MAPPING
from onnx9000.converters.paddle.parsers import PaddleNode


def test_paddle_nn_ops_matmul() -> None:
    """Tests the paddle nn ops matmul functionality."""
    builder = PaddleToONNXGraphBuilder()
    n = PaddleNode(
        "n",
        "matmul",
        inputs={"X": ["a"], "Y": ["b"]},
        attrs={"transpose_X": True, "transpose_Y": True},
    )
    NN_OPS_MAPPING["matmul"](builder, n)
    assert builder.graph.nodes[-1].op_type == "MatMul"
    assert builder.graph.nodes[-2].op_type == "Transpose"
    assert builder.graph.nodes[-3].op_type == "Transpose"
    n2 = PaddleNode("n", "matmul", inputs={"X": ["a"], "Y": ["b"]})
    NN_OPS_MAPPING["matmul"](builder, n2)
    assert builder.graph.nodes[-1].op_type == "MatMul"


def test_paddle_nn_ops_linear() -> None:
    """Tests the paddle nn ops linear functionality."""
    builder = PaddleToONNXGraphBuilder()
    n = PaddleNode("n", "fc", inputs={"X": ["a"], "Y": ["w"], "Bias": ["b"]})
    NN_OPS_MAPPING["fc"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Add"
    assert builder.graph.nodes[-2].op_type == "MatMul"
    n2 = PaddleNode("n", "fc", inputs={"X": ["a"], "Y": ["w"]})
    NN_OPS_MAPPING["fc"](builder, n2)
    assert builder.graph.nodes[-1].op_type == "MatMul"


def test_paddle_nn_ops_conv() -> None:
    """Tests the paddle nn ops conv functionality."""
    builder = PaddleToONNXGraphBuilder()
    n = PaddleNode(
        "n", "conv2d", inputs={"Input": ["a"], "Filter": ["w"], "Bias": ["b"]}, attrs={"groups": 2}
    )
    NN_OPS_MAPPING["conv2d"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Conv"
    assert builder.graph.nodes[-1].attributes["group"] == 2


def test_paddle_nn_ops_pool() -> None:
    """Tests the paddle nn ops pool functionality."""
    builder = PaddleToONNXGraphBuilder()
    n = PaddleNode("n", "pool2d", inputs={"X": ["a"], "Y": ["b"]}, attrs={"global_pooling": True})
    NN_OPS_MAPPING["pool2d"](builder, n)
    assert builder.graph.nodes[-1].op_type == "GlobalMaxPool"
    n2 = PaddleNode(
        "n",
        "pool2d",
        inputs={"X": ["a"], "Y": ["b"]},
        attrs={"global_pooling": False, "pooling_type": "avg"},
    )
    NN_OPS_MAPPING["pool2d"](builder, n2)
    assert builder.graph.nodes[-1].op_type == "AveragePool"


def test_paddle_nn_ops_adaptive_pool() -> None:
    """Tests the paddle nn ops adaptive pool functionality."""
    builder = PaddleToONNXGraphBuilder()
    n = PaddleNode(
        "n", "adaptive_pool2d", inputs={"X": ["a"], "Y": ["b"]}, attrs={"pool_size": [1, 1]}
    )
    NN_OPS_MAPPING["adaptive_pool2d"](builder, n)
    assert builder.graph.nodes[-1].op_type == "GlobalMaxPool"
    n2 = PaddleNode(
        "n",
        "adaptive_pool2d",
        inputs={"X": ["a"], "Y": ["b"]},
        attrs={"pool_size": [2, 2], "pooling_type": "avg"},
    )
    NN_OPS_MAPPING["adaptive_pool2d"](builder, n2)
    assert builder.graph.nodes[-1].op_type == "AveragePool"
    n3 = PaddleNode(
        "n",
        "adaptive_pool2d",
        inputs={"X": ["a"], "Y": ["b"]},
        attrs={"pool_size": [2, 2], "pooling_type": "max"},
    )
    NN_OPS_MAPPING["adaptive_pool2d"](builder, n3)
    assert builder.graph.nodes[-1].op_type == "MaxPool"
    n4 = PaddleNode(
        "n",
        "adaptive_pool2d",
        inputs={"X": ["a"], "Y": ["b"]},
        attrs={"pool_size": [1, 1], "pooling_type": "avg"},
    )
    NN_OPS_MAPPING["adaptive_pool2d"](builder, n4)
    assert builder.graph.nodes[-1].op_type == "GlobalAveragePool"


def test_paddle_nn_ops_norms() -> None:
    """Tests the paddle nn ops norms functionality."""
    builder = PaddleToONNXGraphBuilder()
    n = PaddleNode(
        "n",
        "batch_norm",
        inputs={"X": ["a"], "Scale": ["s"], "Bias": ["b"], "Mean": ["m"], "Variance": ["v"]},
    )
    NN_OPS_MAPPING["batch_norm"](builder, n)
    assert builder.graph.nodes[-1].op_type == "BatchNormalization"
    n2 = PaddleNode("n", "layer_norm", inputs={"X": ["a"], "Scale": ["s"], "Bias": ["b"]})
    NN_OPS_MAPPING["layer_norm"](builder, n2)
    assert builder.graph.nodes[-1].op_type == "LayerNormalization"
    n3 = PaddleNode("n", "group_norm", inputs={"X": ["a"], "Y": ["b"]})
    NN_OPS_MAPPING["group_norm"](builder, n3)
    assert builder.graph.nodes[-1].op_type == "Reshape"
    n4 = PaddleNode("n", "instance_norm", inputs={"X": ["a"], "Y": ["b"]})
    NN_OPS_MAPPING["instance_norm"](builder, n4)
    assert builder.graph.nodes[-1].op_type == "InstanceNormalization"


def test_paddle_nn_ops_activations() -> None:
    """Tests the paddle nn ops activations functionality."""
    builder = PaddleToONNXGraphBuilder()
    n = PaddleNode("n", "relu6", inputs={"X": ["a"], "Y": ["b"]})
    NN_OPS_MAPPING["relu6"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Clip"
    n = PaddleNode("n", "leaky_relu", inputs={"X": ["a"], "Y": ["b"]})
    NN_OPS_MAPPING["leaky_relu"](builder, n)
    assert builder.graph.nodes[-1].op_type == "LeakyRelu"
    n = PaddleNode("n", "elu", inputs={"X": ["a"], "Y": ["b"]})
    NN_OPS_MAPPING["elu"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Elu"
    n = PaddleNode("n", "selu", inputs={"X": ["a"], "Y": ["b"]})
    NN_OPS_MAPPING["selu"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Selu"
    n = PaddleNode("n", "gelu", inputs={"X": ["a"], "Y": ["b"]}, attrs={"approximate": True})
    NN_OPS_MAPPING["gelu"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Mul"
    n = PaddleNode("n", "gelu", inputs={"X": ["a"], "Y": ["b"]}, attrs={"approximate": False})
    NN_OPS_MAPPING["gelu"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Mul"
    n = PaddleNode("n", "swish", inputs={"X": ["a"], "Y": ["b"]})
    NN_OPS_MAPPING["swish"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Mul"
    assert builder.graph.nodes[-2].op_type == "Sigmoid"
    n = PaddleNode("n", "sigmoid", inputs={"X": ["a"], "Y": ["b"]})
    NN_OPS_MAPPING["sigmoid"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Sigmoid"
    n = PaddleNode("n", "softplus", inputs={"X": ["a"], "Y": ["b"]})
    NN_OPS_MAPPING["softplus"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Softplus"
    n = PaddleNode("n", "softsign", inputs={"X": ["a"], "Y": ["b"]})
    NN_OPS_MAPPING["softsign"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Softsign"


def test_paddle_nn_ops_misc() -> None:
    """Tests the paddle nn ops misc functionality."""
    builder = PaddleToONNXGraphBuilder()
    for op in [
        "hard_swish",
        "hard_sigmoid",
        "softmax",
        "log_softmax",
        "dropout",
        "p_norm",
        "unpool",
        "mul",
    ]:
        n = PaddleNode("n", op, inputs={"X": ["a"], "Y": ["b"]})
        outs = NN_OPS_MAPPING[op](builder, n)
        assert len(outs) >= 1


def test_paddle_nn_ops_pad_rois() -> None:
    """Tests the paddle nn ops pad rois functionality."""
    builder = PaddleToONNXGraphBuilder()
    n = PaddleNode(
        "n", "pad", inputs={"X": ["a"], "Y": ["b"]}, attrs={"paddings": [1, 1], "pad_value": 0.0}
    )
    NN_OPS_MAPPING["pad"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Pad"
    n = PaddleNode("n", "roi_align", inputs={"X": ["a"], "ROIs": ["r"]})
    NN_OPS_MAPPING["roi_align"](builder, n)
    assert builder.graph.nodes[-1].op_type == "RoiAlign"
    n = PaddleNode("n", "roi_pool", inputs={"X": ["a"], "ROIs": ["r"]})
    NN_OPS_MAPPING["roi_pool"](builder, n)
    assert builder.graph.nodes[-1].op_type == "MaxRoiPool"
    n = PaddleNode("n", "deformable_conv", inputs={"X": ["a"], "Y": ["b"]})
    NN_OPS_MAPPING["deformable_conv"](builder, n)
    assert builder.graph.nodes[-1].op_type == "DeformConv"


def test_paddle_nn_ops_instance_norm_with_scale_bias() -> None:
    """Tests the paddle nn ops instance norm with scale bias functionality."""
    from onnx9000.converters.paddle.builder import PaddleToONNXGraphBuilder
    from onnx9000.converters.paddle.nn_ops import _map_instance_norm
    from onnx9000.converters.paddle.parsers import PaddleNode

    builder = PaddleToONNXGraphBuilder()
    n = PaddleNode("n", "instance_norm", inputs={"X": ["x"], "Scale": ["s"], "Bias": ["b"]})
    _map_instance_norm(builder, n)
    assert builder.graph.nodes[-1].op_type == "InstanceNormalization"
    assert builder.graph.nodes[-1].inputs == ["x", "s", "b"]


def test_nn_ops_custom() -> None:
    """Tests the nn ops custom functionality."""
    builder = PaddleToONNXGraphBuilder()
    n = PaddleNode("n", "bce_loss", inputs={"X": ["a"], "Y": ["b"]})
    NN_OPS_MAPPING["bce_loss"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Custom_Paddle_bce_loss"
    n2 = PaddleNode("n2", "mish", inputs={"X": ["a"], "Y": ["b"]})
    NN_OPS_MAPPING["mish"](builder, n2)
    assert builder.graph.nodes[-1].op_type == "Mish"
