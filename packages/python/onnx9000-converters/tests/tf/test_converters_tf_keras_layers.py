"""Tests the keras layers module functionality."""

from onnx9000.converters.tf.builder import TFToONNXGraphBuilder
from onnx9000.converters.tf.keras_layers import KERAS_LAYERS_MAPPING
from onnx9000.converters.tf.parsers import TFNode


def test_keras_layers_simple() -> None:
    """Tests the keras layers simple functionality."""
    builder = TFToONNXGraphBuilder()

    # Check dense specifically since it can generate MatMul or Add
    node = TFNode("n_dense", "Dense", inputs=["a", "b"])
    KERAS_LAYERS_MAPPING["Dense"](builder, node)
    assert builder.graph.nodes[-1].op_type == "MatMul"

    node3 = TFNode("n_dense3", "Dense", inputs=["a", "b", "c"])
    KERAS_LAYERS_MAPPING["Dense"](builder, node3)
    assert builder.graph.nodes[-1].op_type == "Add"

    direct_maps = {
        "Conv1D": "Conv",
        "Conv2D": "Conv",
        "Conv3D": "Conv",
        "SeparableConv1D": "Conv",
        "SeparableConv2D": "Conv",
        "DepthwiseConv2D": "Conv",
        "Conv2DTranspose": "ConvTranspose",
        "Conv3DTranspose": "ConvTranspose",
        "MaxPooling1D": "MaxPool",
        "MaxPooling2D": "MaxPool",
        "MaxPooling3D": "MaxPool",
        "AveragePooling1D": "AveragePool",
        "AveragePooling2D": "AveragePool",
        "AveragePooling3D": "AveragePool",
        "GlobalMaxPooling1D": "GlobalMaxPool",
        "GlobalMaxPooling2D": "GlobalMaxPool",
        "GlobalMaxPooling3D": "GlobalMaxPool",
        "GlobalAveragePooling1D": "GlobalAveragePool",
        "GlobalAveragePooling2D": "GlobalAveragePool",
        "GlobalAveragePooling3D": "GlobalAveragePool",
        "RepeatVector": "Tile",
        "Dot": "MatMul",
        "Embedding": "Gather",
        "BatchNormalization": "BatchNormalization",
        "LayerNormalization": "LayerNormalization",
        "Dropout": "Dropout",
        "Flatten": "Flatten",
        "Reshape": "Reshape",
        "Permute": "Transpose",
        "Concatenate": "Concat",
        "Average": "Mean",
        "Maximum": "Max",
        "Minimum": "Min",
        "Add": "Sum",
        "Subtract": "Sub",
        "Multiply": "Mul",
    }
    for op, expected in direct_maps.items():
        KERAS_LAYERS_MAPPING[op](builder, TFNode(f"n_{op}", op, inputs=["a", "b"]))
        assert builder.graph.nodes[-1].op_type == expected

    act_node = TFNode("n_act", "Activation", inputs=["a"], attr={"activation": b"sigmoid"})
    KERAS_LAYERS_MAPPING["Activation"](builder, act_node)
    assert builder.graph.nodes[-1].op_type == "Sigmoid"


def test_keras_layers_dimensional() -> None:
    """Tests the keras layers dimensional functionality."""
    builder = TFToONNXGraphBuilder()

    dim_map = {
        "SpatialDropout": "Dropout",
        "Cropping": "Slice",
        "UpSampling": "Resize",
        "ZeroPadding": "Pad",
    }

    for op_prefix, expected in dim_map.items():
        for dim in ["1D", "2D", "3D"]:
            op = f"{op_prefix}{dim}"
            KERAS_LAYERS_MAPPING[op](builder, TFNode("n", op, inputs=["a"]))
            assert builder.graph.nodes[-1].op_type == expected


def test_keras_layers_missing_initializers_stft() -> None:
    """Tests the keras layers initializers.STFT functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "initializers.STFT", inputs=["a"])

    KERAS_LAYERS_MAPPING["initializers.STFT"](builder, node)
    assert builder.graph.nodes[-1].op_type == "STFT"

    node_empty = TFNode("n2", "initializers.STFT", inputs=[])
    KERAS_LAYERS_MAPPING["initializers.STFT"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_initializers_stft_initializer() -> None:
    """Tests the keras layers initializers.STFTInitializer functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "initializers.STFTInitializer", inputs=["a"])

    KERAS_LAYERS_MAPPING["initializers.STFTInitializer"](builder, node)
    assert builder.graph.nodes[-1].op_type == "STFT"

    node_empty = TFNode("n2", "initializers.STFTInitializer", inputs=[])
    KERAS_LAYERS_MAPPING["initializers.STFTInitializer"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_initializers_stft_ext() -> None:
    """Tests the keras layers initializers.stft functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "initializers.stft", inputs=["a"])

    KERAS_LAYERS_MAPPING["initializers.stft"](builder, node)
    assert builder.graph.nodes[-1].op_type == "STFT"

    node_empty = TFNode("n2", "initializers.stft", inputs=[])
    KERAS_LAYERS_MAPPING["initializers.stft"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_activations_deserialize() -> None:
    """Tests the keras layers activations.deserialize functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "activations.deserialize", inputs=["a"])

    KERAS_LAYERS_MAPPING["activations.deserialize"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "activations.deserialize", inputs=[])
    KERAS_LAYERS_MAPPING["activations.deserialize"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_activations_get() -> None:
    """Tests the keras layers activations.get functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "activations.get", inputs=["a"])

    KERAS_LAYERS_MAPPING["activations.get"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "activations.get", inputs=[])
    KERAS_LAYERS_MAPPING["activations.get"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_activations_serialize() -> None:
    """Tests the keras layers activations.serialize functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "activations.serialize", inputs=["a"])

    KERAS_LAYERS_MAPPING["activations.serialize"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "activations.serialize", inputs=[])
    KERAS_LAYERS_MAPPING["activations.serialize"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_activations_celu() -> None:
    """Tests the keras layers activations.celu functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "activations.celu", inputs=["a"])

    KERAS_LAYERS_MAPPING["activations.celu"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "activations.celu", inputs=[])
    KERAS_LAYERS_MAPPING["activations.celu"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_activations_elu() -> None:
    """Tests the keras layers activations.elu functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "activations.elu", inputs=["a"])

    KERAS_LAYERS_MAPPING["activations.elu"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "activations.elu", inputs=[])
    KERAS_LAYERS_MAPPING["activations.elu"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_activations_exponential() -> None:
    """Tests the keras layers activations.exponential functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "activations.exponential", inputs=["a"])

    KERAS_LAYERS_MAPPING["activations.exponential"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "activations.exponential", inputs=[])
    KERAS_LAYERS_MAPPING["activations.exponential"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_activations_gelu() -> None:
    """Tests the keras layers activations.gelu functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "activations.gelu", inputs=["a"])

    KERAS_LAYERS_MAPPING["activations.gelu"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "activations.gelu", inputs=[])
    KERAS_LAYERS_MAPPING["activations.gelu"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_activations_glu() -> None:
    """Tests the keras layers activations.glu functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "activations.glu", inputs=["a"])

    KERAS_LAYERS_MAPPING["activations.glu"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "activations.glu", inputs=[])
    KERAS_LAYERS_MAPPING["activations.glu"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_activations_hard_shrink() -> None:
    """Tests the keras layers activations.hard_shrink functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "activations.hard_shrink", inputs=["a"])

    KERAS_LAYERS_MAPPING["activations.hard_shrink"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "activations.hard_shrink", inputs=[])
    KERAS_LAYERS_MAPPING["activations.hard_shrink"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_activations_hard_sigmoid() -> None:
    """Tests the keras layers activations.hard_sigmoid functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "activations.hard_sigmoid", inputs=["a"])

    KERAS_LAYERS_MAPPING["activations.hard_sigmoid"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "activations.hard_sigmoid", inputs=[])
    KERAS_LAYERS_MAPPING["activations.hard_sigmoid"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_activations_hard_silu() -> None:
    """Tests the keras layers activations.hard_silu functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "activations.hard_silu", inputs=["a"])

    KERAS_LAYERS_MAPPING["activations.hard_silu"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "activations.hard_silu", inputs=[])
    KERAS_LAYERS_MAPPING["activations.hard_silu"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_activations_hard_swish() -> None:
    """Tests the keras layers activations.hard_swish functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "activations.hard_swish", inputs=["a"])

    KERAS_LAYERS_MAPPING["activations.hard_swish"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "activations.hard_swish", inputs=[])
    KERAS_LAYERS_MAPPING["activations.hard_swish"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_activations_hard_tanh() -> None:
    """Tests the keras layers activations.hard_tanh functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "activations.hard_tanh", inputs=["a"])

    KERAS_LAYERS_MAPPING["activations.hard_tanh"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "activations.hard_tanh", inputs=[])
    KERAS_LAYERS_MAPPING["activations.hard_tanh"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_activations_leaky_relu() -> None:
    """Tests the keras layers activations.leaky_relu functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "activations.leaky_relu", inputs=["a"])

    KERAS_LAYERS_MAPPING["activations.leaky_relu"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "activations.leaky_relu", inputs=[])
    KERAS_LAYERS_MAPPING["activations.leaky_relu"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_activations_linear() -> None:
    """Tests the keras layers activations.linear functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "activations.linear", inputs=["a"])

    KERAS_LAYERS_MAPPING["activations.linear"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "activations.linear", inputs=[])
    KERAS_LAYERS_MAPPING["activations.linear"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_activations_log_sigmoid() -> None:
    """Tests the keras layers activations.log_sigmoid functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "activations.log_sigmoid", inputs=["a"])

    KERAS_LAYERS_MAPPING["activations.log_sigmoid"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "activations.log_sigmoid", inputs=[])
    KERAS_LAYERS_MAPPING["activations.log_sigmoid"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_activations_log_softmax() -> None:
    """Tests the keras layers activations.log_softmax functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "activations.log_softmax", inputs=["a"])

    KERAS_LAYERS_MAPPING["activations.log_softmax"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "activations.log_softmax", inputs=[])
    KERAS_LAYERS_MAPPING["activations.log_softmax"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_activations_mish() -> None:
    """Tests the keras layers activations.mish functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "activations.mish", inputs=["a"])

    KERAS_LAYERS_MAPPING["activations.mish"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "activations.mish", inputs=[])
    KERAS_LAYERS_MAPPING["activations.mish"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_activations_relu() -> None:
    """Tests the keras layers activations.relu functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "activations.relu", inputs=["a"])

    KERAS_LAYERS_MAPPING["activations.relu"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "activations.relu", inputs=[])
    KERAS_LAYERS_MAPPING["activations.relu"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_activations_relu6() -> None:
    """Tests the keras layers activations.relu6 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "activations.relu6", inputs=["a"])

    KERAS_LAYERS_MAPPING["activations.relu6"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "activations.relu6", inputs=[])
    KERAS_LAYERS_MAPPING["activations.relu6"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_activations_selu() -> None:
    """Tests the keras layers activations.selu functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "activations.selu", inputs=["a"])

    KERAS_LAYERS_MAPPING["activations.selu"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "activations.selu", inputs=[])
    KERAS_LAYERS_MAPPING["activations.selu"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_activations_sigmoid() -> None:
    """Tests the keras layers activations.sigmoid functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "activations.sigmoid", inputs=["a"])

    KERAS_LAYERS_MAPPING["activations.sigmoid"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "activations.sigmoid", inputs=[])
    KERAS_LAYERS_MAPPING["activations.sigmoid"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_activations_silu() -> None:
    """Tests the keras layers activations.silu functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "activations.silu", inputs=["a"])

    KERAS_LAYERS_MAPPING["activations.silu"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "activations.silu", inputs=[])
    KERAS_LAYERS_MAPPING["activations.silu"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_activations_swish() -> None:
    """Tests the keras layers activations.swish functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "activations.swish", inputs=["a"])

    KERAS_LAYERS_MAPPING["activations.swish"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "activations.swish", inputs=[])
    KERAS_LAYERS_MAPPING["activations.swish"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_activations_soft_shrink() -> None:
    """Tests the keras layers activations.soft_shrink functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "activations.soft_shrink", inputs=["a"])

    KERAS_LAYERS_MAPPING["activations.soft_shrink"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "activations.soft_shrink", inputs=[])
    KERAS_LAYERS_MAPPING["activations.soft_shrink"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_activations_softmax() -> None:
    """Tests the keras layers activations.softmax functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "activations.softmax", inputs=["a"])

    KERAS_LAYERS_MAPPING["activations.softmax"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "activations.softmax", inputs=[])
    KERAS_LAYERS_MAPPING["activations.softmax"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_activations_softplus() -> None:
    """Tests the keras layers activations.softplus functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "activations.softplus", inputs=["a"])

    KERAS_LAYERS_MAPPING["activations.softplus"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "activations.softplus", inputs=[])
    KERAS_LAYERS_MAPPING["activations.softplus"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_activations_softsign() -> None:
    """Tests the keras layers activations.softsign functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "activations.softsign", inputs=["a"])

    KERAS_LAYERS_MAPPING["activations.softsign"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "activations.softsign", inputs=[])
    KERAS_LAYERS_MAPPING["activations.softsign"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_activations_sparse_plus() -> None:
    """Tests the keras layers activations.sparse_plus functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "activations.sparse_plus", inputs=["a"])

    KERAS_LAYERS_MAPPING["activations.sparse_plus"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "activations.sparse_plus", inputs=[])
    KERAS_LAYERS_MAPPING["activations.sparse_plus"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_activations_sparse_sigmoid() -> None:
    """Tests the keras layers activations.sparse_sigmoid functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "activations.sparse_sigmoid", inputs=["a"])

    KERAS_LAYERS_MAPPING["activations.sparse_sigmoid"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "activations.sparse_sigmoid", inputs=[])
    KERAS_LAYERS_MAPPING["activations.sparse_sigmoid"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_activations_sparsemax() -> None:
    """Tests the keras layers activations.sparsemax functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "activations.sparsemax", inputs=["a"])

    KERAS_LAYERS_MAPPING["activations.sparsemax"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "activations.sparsemax", inputs=[])
    KERAS_LAYERS_MAPPING["activations.sparsemax"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_activations_squareplus() -> None:
    """Tests the keras layers activations.squareplus functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "activations.squareplus", inputs=["a"])

    KERAS_LAYERS_MAPPING["activations.squareplus"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "activations.squareplus", inputs=[])
    KERAS_LAYERS_MAPPING["activations.squareplus"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_activations_tanh() -> None:
    """Tests the keras layers activations.tanh functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "activations.tanh", inputs=["a"])

    KERAS_LAYERS_MAPPING["activations.tanh"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "activations.tanh", inputs=[])
    KERAS_LAYERS_MAPPING["activations.tanh"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_activations_tanh_shrink() -> None:
    """Tests the keras layers activations.tanh_shrink functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "activations.tanh_shrink", inputs=["a"])

    KERAS_LAYERS_MAPPING["activations.tanh_shrink"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "activations.tanh_shrink", inputs=[])
    KERAS_LAYERS_MAPPING["activations.tanh_shrink"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_activations_threshold() -> None:
    """Tests the keras layers activations.threshold functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "activations.threshold", inputs=["a"])

    KERAS_LAYERS_MAPPING["activations.threshold"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "activations.threshold", inputs=[])
    KERAS_LAYERS_MAPPING["activations.threshold"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_constraints_deserialize() -> None:
    """Tests the keras layers constraints.deserialize functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "constraints.deserialize", inputs=["a"])

    KERAS_LAYERS_MAPPING["constraints.deserialize"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "constraints.deserialize", inputs=[])
    KERAS_LAYERS_MAPPING["constraints.deserialize"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_constraints_get() -> None:
    """Tests the keras layers constraints.get functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "constraints.get", inputs=["a"])

    KERAS_LAYERS_MAPPING["constraints.get"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "constraints.get", inputs=[])
    KERAS_LAYERS_MAPPING["constraints.get"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_constraints_serialize() -> None:
    """Tests the keras layers constraints.serialize functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "constraints.serialize", inputs=["a"])

    KERAS_LAYERS_MAPPING["constraints.serialize"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "constraints.serialize", inputs=[])
    KERAS_LAYERS_MAPPING["constraints.serialize"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_constraints__constraint() -> None:
    """Tests the keras layers constraints.Constraint functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "constraints.Constraint", inputs=["a"])

    KERAS_LAYERS_MAPPING["constraints.Constraint"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "constraints.Constraint", inputs=[])
    KERAS_LAYERS_MAPPING["constraints.Constraint"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_constraints__max_norm() -> None:
    """Tests the keras layers constraints.MaxNorm functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "constraints.MaxNorm", inputs=["a"])

    KERAS_LAYERS_MAPPING["constraints.MaxNorm"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "constraints.MaxNorm", inputs=[])
    KERAS_LAYERS_MAPPING["constraints.MaxNorm"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_constraints_max_norm() -> None:
    """Tests the keras layers constraints.max_norm functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "constraints.max_norm", inputs=["a"])

    KERAS_LAYERS_MAPPING["constraints.max_norm"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "constraints.max_norm", inputs=[])
    KERAS_LAYERS_MAPPING["constraints.max_norm"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_constraints__min_max_norm() -> None:
    """Tests the keras layers constraints.MinMaxNorm functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "constraints.MinMaxNorm", inputs=["a"])

    KERAS_LAYERS_MAPPING["constraints.MinMaxNorm"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "constraints.MinMaxNorm", inputs=[])
    KERAS_LAYERS_MAPPING["constraints.MinMaxNorm"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_constraints_min_max_norm() -> None:
    """Tests the keras layers constraints.min_max_norm functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "constraints.min_max_norm", inputs=["a"])

    KERAS_LAYERS_MAPPING["constraints.min_max_norm"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "constraints.min_max_norm", inputs=[])
    KERAS_LAYERS_MAPPING["constraints.min_max_norm"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_constraints__non_neg() -> None:
    """Tests the keras layers constraints.NonNeg functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "constraints.NonNeg", inputs=["a"])

    KERAS_LAYERS_MAPPING["constraints.NonNeg"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "constraints.NonNeg", inputs=[])
    KERAS_LAYERS_MAPPING["constraints.NonNeg"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_constraints_non_neg() -> None:
    """Tests the keras layers constraints.non_neg functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "constraints.non_neg", inputs=["a"])

    KERAS_LAYERS_MAPPING["constraints.non_neg"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "constraints.non_neg", inputs=[])
    KERAS_LAYERS_MAPPING["constraints.non_neg"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_constraints__unit_norm() -> None:
    """Tests the keras layers constraints.UnitNorm functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "constraints.UnitNorm", inputs=["a"])

    KERAS_LAYERS_MAPPING["constraints.UnitNorm"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "constraints.UnitNorm", inputs=[])
    KERAS_LAYERS_MAPPING["constraints.UnitNorm"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_constraints_unit_norm() -> None:
    """Tests the keras layers constraints.unit_norm functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "constraints.unit_norm", inputs=["a"])

    KERAS_LAYERS_MAPPING["constraints.unit_norm"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "constraints.unit_norm", inputs=[])
    KERAS_LAYERS_MAPPING["constraints.unit_norm"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers_tfsm_layer() -> None:
    """Tests the keras layers layers.TFSMLayer functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.TFSMLayer", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.TFSMLayer"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.TFSMLayer", inputs=[])
    KERAS_LAYERS_MAPPING["layers.TFSMLayer"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers_deserialize() -> None:
    """Tests the keras layers layers.deserialize functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.deserialize", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.deserialize"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.deserialize", inputs=[])
    KERAS_LAYERS_MAPPING["layers.deserialize"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers_serialize() -> None:
    """Tests the keras layers layers.serialize functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.serialize", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.serialize"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.serialize", inputs=[])
    KERAS_LAYERS_MAPPING["layers.serialize"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__activation() -> None:
    """Tests the keras layers layers.Activation functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Activation", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Activation"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.Activation", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Activation"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers_elu() -> None:
    """Tests the keras layers layers.ELU functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.ELU", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.ELU"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.ELU", inputs=[])
    KERAS_LAYERS_MAPPING["layers.ELU"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__leaky_re_lu() -> None:
    """Tests the keras layers layers.LeakyReLU functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.LeakyReLU", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.LeakyReLU"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.LeakyReLU", inputs=[])
    KERAS_LAYERS_MAPPING["layers.LeakyReLU"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers_p_re_lu() -> None:
    """Tests the keras layers layers.PReLU functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.PReLU", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.PReLU"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.PReLU", inputs=[])
    KERAS_LAYERS_MAPPING["layers.PReLU"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__re_lu() -> None:
    """Tests the keras layers layers.ReLU functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.ReLU", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.ReLU"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.ReLU", inputs=[])
    KERAS_LAYERS_MAPPING["layers.ReLU"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__softmax() -> None:
    """Tests the keras layers layers.Softmax functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Softmax", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Softmax"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.Softmax", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Softmax"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__additive_attention() -> None:
    """Tests the keras layers layers.AdditiveAttention functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.AdditiveAttention", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.AdditiveAttention"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.AdditiveAttention", inputs=[])
    KERAS_LAYERS_MAPPING["layers.AdditiveAttention"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__attention() -> None:
    """Tests the keras layers layers.Attention functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Attention", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Attention"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.Attention", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Attention"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__group_query_attention() -> None:
    """Tests the keras layers layers.GroupQueryAttention functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.GroupQueryAttention", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.GroupQueryAttention"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.GroupQueryAttention", inputs=[])
    KERAS_LAYERS_MAPPING["layers.GroupQueryAttention"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__multi_head_attention() -> None:
    """Tests the keras layers layers.MultiHeadAttention functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.MultiHeadAttention", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.MultiHeadAttention"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.MultiHeadAttention", inputs=[])
    KERAS_LAYERS_MAPPING["layers.MultiHeadAttention"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__conv1_d() -> None:
    """Tests the keras layers layers.Conv1D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Conv1D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Conv1D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.Conv1D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Conv1D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__convolution1_d() -> None:
    """Tests the keras layers layers.Convolution1D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Convolution1D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Convolution1D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.Convolution1D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Convolution1D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__conv1_d_transpose() -> None:
    """Tests the keras layers layers.Conv1DTranspose functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Conv1DTranspose", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Conv1DTranspose"](builder, node)
    assert builder.graph.nodes[-1].op_type == "ConvTranspose"

    node_empty = TFNode("n2", "layers.Conv1DTranspose", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Conv1DTranspose"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__convolution1_d_transpose() -> None:
    """Tests the keras layers layers.Convolution1DTranspose functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Convolution1DTranspose", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Convolution1DTranspose"](builder, node)
    assert builder.graph.nodes[-1].op_type == "ConvTranspose"

    node_empty = TFNode("n2", "layers.Convolution1DTranspose", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Convolution1DTranspose"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__conv2_d() -> None:
    """Tests the keras layers layers.Conv2D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Conv2D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Conv2D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.Conv2D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Conv2D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__convolution2_d() -> None:
    """Tests the keras layers layers.Convolution2D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Convolution2D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Convolution2D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.Convolution2D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Convolution2D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__conv2_d_transpose() -> None:
    """Tests the keras layers layers.Conv2DTranspose functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Conv2DTranspose", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Conv2DTranspose"](builder, node)
    assert builder.graph.nodes[-1].op_type == "ConvTranspose"

    node_empty = TFNode("n2", "layers.Conv2DTranspose", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Conv2DTranspose"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__convolution2_d_transpose() -> None:
    """Tests the keras layers layers.Convolution2DTranspose functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Convolution2DTranspose", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Convolution2DTranspose"](builder, node)
    assert builder.graph.nodes[-1].op_type == "ConvTranspose"

    node_empty = TFNode("n2", "layers.Convolution2DTranspose", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Convolution2DTranspose"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__conv3_d() -> None:
    """Tests the keras layers layers.Conv3D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Conv3D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Conv3D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.Conv3D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Conv3D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__convolution3_d() -> None:
    """Tests the keras layers layers.Convolution3D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Convolution3D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Convolution3D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.Convolution3D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Convolution3D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__conv3_d_transpose() -> None:
    """Tests the keras layers layers.Conv3DTranspose functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Conv3DTranspose", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Conv3DTranspose"](builder, node)
    assert builder.graph.nodes[-1].op_type == "ConvTranspose"

    node_empty = TFNode("n2", "layers.Conv3DTranspose", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Conv3DTranspose"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__convolution3_d_transpose() -> None:
    """Tests the keras layers layers.Convolution3DTranspose functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Convolution3DTranspose", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Convolution3DTranspose"](builder, node)
    assert builder.graph.nodes[-1].op_type == "ConvTranspose"

    node_empty = TFNode("n2", "layers.Convolution3DTranspose", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Convolution3DTranspose"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__depthwise_conv1_d() -> None:
    """Tests the keras layers layers.DepthwiseConv1D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.DepthwiseConv1D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.DepthwiseConv1D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Conv"

    node_empty = TFNode("n2", "layers.DepthwiseConv1D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.DepthwiseConv1D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__depthwise_conv2_d() -> None:
    """Tests the keras layers layers.DepthwiseConv2D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.DepthwiseConv2D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.DepthwiseConv2D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Conv"

    node_empty = TFNode("n2", "layers.DepthwiseConv2D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.DepthwiseConv2D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__separable_conv1_d() -> None:
    """Tests the keras layers layers.SeparableConv1D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.SeparableConv1D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.SeparableConv1D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.SeparableConv1D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.SeparableConv1D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__separable_convolution1_d() -> None:
    """Tests the keras layers layers.SeparableConvolution1D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.SeparableConvolution1D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.SeparableConvolution1D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.SeparableConvolution1D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.SeparableConvolution1D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__separable_conv2_d() -> None:
    """Tests the keras layers layers.SeparableConv2D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.SeparableConv2D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.SeparableConv2D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.SeparableConv2D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.SeparableConv2D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__separable_convolution2_d() -> None:
    """Tests the keras layers layers.SeparableConvolution2D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.SeparableConvolution2D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.SeparableConvolution2D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.SeparableConvolution2D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.SeparableConvolution2D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__dense() -> None:
    """Tests the keras layers layers.Dense functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Dense", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Dense"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.Dense", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Dense"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__einsum_dense() -> None:
    """Tests the keras layers layers.EinsumDense functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.EinsumDense", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.EinsumDense"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Einsum"

    node_empty = TFNode("n2", "layers.EinsumDense", inputs=[])
    KERAS_LAYERS_MAPPING["layers.EinsumDense"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__embedding() -> None:
    """Tests the keras layers layers.Embedding functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Embedding", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Embedding"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Gather"

    node_empty = TFNode("n2", "layers.Embedding", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Embedding"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__identity() -> None:
    """Tests the keras layers layers.Identity functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Identity", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Identity"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.Identity", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Identity"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__input() -> None:
    """Tests the keras layers layers.Input functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Input", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Input"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.Input", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Input"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__input_layer() -> None:
    """Tests the keras layers layers.InputLayer functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.InputLayer", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.InputLayer"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.InputLayer", inputs=[])
    KERAS_LAYERS_MAPPING["layers.InputLayer"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__lambda() -> None:
    """Tests the keras layers layers.Lambda functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Lambda", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Lambda"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.Lambda", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Lambda"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__masking() -> None:
    """Tests the keras layers layers.Masking functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Masking", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Masking"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.Masking", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Masking"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__reversible_embedding() -> None:
    """Tests the keras layers layers.ReversibleEmbedding functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.ReversibleEmbedding", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.ReversibleEmbedding"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.ReversibleEmbedding", inputs=[])
    KERAS_LAYERS_MAPPING["layers.ReversibleEmbedding"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__wrapper() -> None:
    """Tests the keras layers layers.Wrapper functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Wrapper", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Wrapper"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.Wrapper", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Wrapper"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__input_spec() -> None:
    """Tests the keras layers layers.InputSpec functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.InputSpec", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.InputSpec"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.InputSpec", inputs=[])
    KERAS_LAYERS_MAPPING["layers.InputSpec"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__layer() -> None:
    """Tests the keras layers layers.Layer functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Layer", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Layer"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.Layer", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Layer"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__add() -> None:
    """Tests the keras layers layers.Add functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Add", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Add"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.Add", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Add"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers_add() -> None:
    """Tests the keras layers layers.add functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.add", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.add"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.add", inputs=[])
    KERAS_LAYERS_MAPPING["layers.add"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__average() -> None:
    """Tests the keras layers layers.Average functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Average", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Average"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.Average", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Average"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers_average() -> None:
    """Tests the keras layers layers.average functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.average", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.average"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.average", inputs=[])
    KERAS_LAYERS_MAPPING["layers.average"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__concatenate() -> None:
    """Tests the keras layers layers.Concatenate functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Concatenate", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Concatenate"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.Concatenate", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Concatenate"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers_concatenate() -> None:
    """Tests the keras layers layers.concatenate functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.concatenate", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.concatenate"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.concatenate", inputs=[])
    KERAS_LAYERS_MAPPING["layers.concatenate"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__dot() -> None:
    """Tests the keras layers layers.Dot functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Dot", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Dot"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.Dot", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Dot"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers_dot() -> None:
    """Tests the keras layers layers.dot functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.dot", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.dot"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.dot", inputs=[])
    KERAS_LAYERS_MAPPING["layers.dot"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__maximum() -> None:
    """Tests the keras layers layers.Maximum functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Maximum", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Maximum"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.Maximum", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Maximum"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers_maximum() -> None:
    """Tests the keras layers layers.maximum functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.maximum", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.maximum"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.maximum", inputs=[])
    KERAS_LAYERS_MAPPING["layers.maximum"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__minimum() -> None:
    """Tests the keras layers layers.Minimum functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Minimum", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Minimum"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.Minimum", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Minimum"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers_minimum() -> None:
    """Tests the keras layers layers.minimum functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.minimum", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.minimum"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.minimum", inputs=[])
    KERAS_LAYERS_MAPPING["layers.minimum"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__multiply() -> None:
    """Tests the keras layers layers.Multiply functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Multiply", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Multiply"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.Multiply", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Multiply"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers_multiply() -> None:
    """Tests the keras layers layers.multiply functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.multiply", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.multiply"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.multiply", inputs=[])
    KERAS_LAYERS_MAPPING["layers.multiply"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__subtract() -> None:
    """Tests the keras layers layers.Subtract functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Subtract", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Subtract"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.Subtract", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Subtract"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers_subtract() -> None:
    """Tests the keras layers layers.subtract functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.subtract", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.subtract"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.subtract", inputs=[])
    KERAS_LAYERS_MAPPING["layers.subtract"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__batch_normalization() -> None:
    """Tests the keras layers layers.BatchNormalization functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.BatchNormalization", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.BatchNormalization"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.BatchNormalization", inputs=[])
    KERAS_LAYERS_MAPPING["layers.BatchNormalization"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__group_normalization() -> None:
    """Tests the keras layers layers.GroupNormalization functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.GroupNormalization", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.GroupNormalization"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.GroupNormalization", inputs=[])
    KERAS_LAYERS_MAPPING["layers.GroupNormalization"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__layer_normalization() -> None:
    """Tests the keras layers layers.LayerNormalization functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.LayerNormalization", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.LayerNormalization"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.LayerNormalization", inputs=[])
    KERAS_LAYERS_MAPPING["layers.LayerNormalization"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers_rms_normalization() -> None:
    """Tests the keras layers layers.RMSNormalization functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.RMSNormalization", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.RMSNormalization"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.RMSNormalization", inputs=[])
    KERAS_LAYERS_MAPPING["layers.RMSNormalization"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__spectral_normalization() -> None:
    """Tests the keras layers layers.SpectralNormalization functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.SpectralNormalization", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.SpectralNormalization"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.SpectralNormalization", inputs=[])
    KERAS_LAYERS_MAPPING["layers.SpectralNormalization"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__unit_normalization() -> None:
    """Tests the keras layers layers.UnitNormalization functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.UnitNormalization", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.UnitNormalization"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.UnitNormalization", inputs=[])
    KERAS_LAYERS_MAPPING["layers.UnitNormalization"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__adaptive_average_pooling1_d() -> None:
    """Tests the keras layers layers.AdaptiveAveragePooling1D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.AdaptiveAveragePooling1D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.AdaptiveAveragePooling1D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.AdaptiveAveragePooling1D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.AdaptiveAveragePooling1D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__adaptive_average_pooling2_d() -> None:
    """Tests the keras layers layers.AdaptiveAveragePooling2D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.AdaptiveAveragePooling2D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.AdaptiveAveragePooling2D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.AdaptiveAveragePooling2D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.AdaptiveAveragePooling2D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__adaptive_average_pooling3_d() -> None:
    """Tests the keras layers layers.AdaptiveAveragePooling3D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.AdaptiveAveragePooling3D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.AdaptiveAveragePooling3D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.AdaptiveAveragePooling3D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.AdaptiveAveragePooling3D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__adaptive_max_pooling1_d() -> None:
    """Tests the keras layers layers.AdaptiveMaxPooling1D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.AdaptiveMaxPooling1D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.AdaptiveMaxPooling1D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.AdaptiveMaxPooling1D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.AdaptiveMaxPooling1D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__adaptive_max_pooling2_d() -> None:
    """Tests the keras layers layers.AdaptiveMaxPooling2D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.AdaptiveMaxPooling2D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.AdaptiveMaxPooling2D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.AdaptiveMaxPooling2D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.AdaptiveMaxPooling2D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__adaptive_max_pooling3_d() -> None:
    """Tests the keras layers layers.AdaptiveMaxPooling3D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.AdaptiveMaxPooling3D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.AdaptiveMaxPooling3D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.AdaptiveMaxPooling3D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.AdaptiveMaxPooling3D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__average_pooling1_d() -> None:
    """Tests the keras layers layers.AveragePooling1D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.AveragePooling1D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.AveragePooling1D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.AveragePooling1D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.AveragePooling1D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__avg_pool1_d() -> None:
    """Tests the keras layers layers.AvgPool1D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.AvgPool1D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.AvgPool1D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.AvgPool1D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.AvgPool1D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__average_pooling2_d() -> None:
    """Tests the keras layers layers.AveragePooling2D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.AveragePooling2D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.AveragePooling2D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.AveragePooling2D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.AveragePooling2D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__avg_pool2_d() -> None:
    """Tests the keras layers layers.AvgPool2D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.AvgPool2D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.AvgPool2D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.AvgPool2D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.AvgPool2D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__average_pooling3_d() -> None:
    """Tests the keras layers layers.AveragePooling3D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.AveragePooling3D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.AveragePooling3D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.AveragePooling3D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.AveragePooling3D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__avg_pool3_d() -> None:
    """Tests the keras layers layers.AvgPool3D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.AvgPool3D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.AvgPool3D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.AvgPool3D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.AvgPool3D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__global_average_pooling1_d() -> None:
    """Tests the keras layers layers.GlobalAveragePooling1D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.GlobalAveragePooling1D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.GlobalAveragePooling1D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.GlobalAveragePooling1D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.GlobalAveragePooling1D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__global_avg_pool1_d() -> None:
    """Tests the keras layers layers.GlobalAvgPool1D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.GlobalAvgPool1D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.GlobalAvgPool1D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.GlobalAvgPool1D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.GlobalAvgPool1D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__global_average_pooling2_d() -> None:
    """Tests the keras layers layers.GlobalAveragePooling2D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.GlobalAveragePooling2D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.GlobalAveragePooling2D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.GlobalAveragePooling2D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.GlobalAveragePooling2D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__global_avg_pool2_d() -> None:
    """Tests the keras layers layers.GlobalAvgPool2D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.GlobalAvgPool2D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.GlobalAvgPool2D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.GlobalAvgPool2D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.GlobalAvgPool2D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__global_average_pooling3_d() -> None:
    """Tests the keras layers layers.GlobalAveragePooling3D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.GlobalAveragePooling3D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.GlobalAveragePooling3D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.GlobalAveragePooling3D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.GlobalAveragePooling3D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__global_avg_pool3_d() -> None:
    """Tests the keras layers layers.GlobalAvgPool3D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.GlobalAvgPool3D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.GlobalAvgPool3D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.GlobalAvgPool3D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.GlobalAvgPool3D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__global_max_pool1_d() -> None:
    """Tests the keras layers layers.GlobalMaxPool1D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.GlobalMaxPool1D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.GlobalMaxPool1D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.GlobalMaxPool1D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.GlobalMaxPool1D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__global_max_pooling1_d() -> None:
    """Tests the keras layers layers.GlobalMaxPooling1D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.GlobalMaxPooling1D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.GlobalMaxPooling1D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.GlobalMaxPooling1D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.GlobalMaxPooling1D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__global_max_pool2_d() -> None:
    """Tests the keras layers layers.GlobalMaxPool2D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.GlobalMaxPool2D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.GlobalMaxPool2D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.GlobalMaxPool2D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.GlobalMaxPool2D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__global_max_pooling2_d() -> None:
    """Tests the keras layers layers.GlobalMaxPooling2D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.GlobalMaxPooling2D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.GlobalMaxPooling2D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.GlobalMaxPooling2D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.GlobalMaxPooling2D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__global_max_pool3_d() -> None:
    """Tests the keras layers layers.GlobalMaxPool3D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.GlobalMaxPool3D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.GlobalMaxPool3D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.GlobalMaxPool3D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.GlobalMaxPool3D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__global_max_pooling3_d() -> None:
    """Tests the keras layers layers.GlobalMaxPooling3D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.GlobalMaxPooling3D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.GlobalMaxPooling3D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.GlobalMaxPooling3D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.GlobalMaxPooling3D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__max_pool1_d() -> None:
    """Tests the keras layers layers.MaxPool1D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.MaxPool1D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.MaxPool1D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.MaxPool1D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.MaxPool1D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__max_pooling1_d() -> None:
    """Tests the keras layers layers.MaxPooling1D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.MaxPooling1D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.MaxPooling1D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.MaxPooling1D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.MaxPooling1D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__max_pool2_d() -> None:
    """Tests the keras layers layers.MaxPool2D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.MaxPool2D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.MaxPool2D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.MaxPool2D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.MaxPool2D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__max_pooling2_d() -> None:
    """Tests the keras layers layers.MaxPooling2D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.MaxPooling2D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.MaxPooling2D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.MaxPooling2D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.MaxPooling2D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__max_pool3_d() -> None:
    """Tests the keras layers layers.MaxPool3D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.MaxPool3D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.MaxPool3D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.MaxPool3D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.MaxPool3D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__max_pooling3_d() -> None:
    """Tests the keras layers layers.MaxPooling3D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.MaxPooling3D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.MaxPooling3D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.MaxPooling3D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.MaxPooling3D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__category_encoding() -> None:
    """Tests the keras layers layers.CategoryEncoding functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.CategoryEncoding", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.CategoryEncoding"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.CategoryEncoding", inputs=[])
    KERAS_LAYERS_MAPPING["layers.CategoryEncoding"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__discretization() -> None:
    """Tests the keras layers layers.Discretization functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Discretization", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Discretization"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.Discretization", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Discretization"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__hashed_crossing() -> None:
    """Tests the keras layers layers.HashedCrossing functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.HashedCrossing", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.HashedCrossing"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.HashedCrossing", inputs=[])
    KERAS_LAYERS_MAPPING["layers.HashedCrossing"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__hashing() -> None:
    """Tests the keras layers layers.Hashing functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Hashing", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Hashing"](builder, node)
    assert builder.graph.nodes[-1].op_type == "CategoryMapper"

    node_empty = TFNode("n2", "layers.Hashing", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Hashing"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__aug_mix() -> None:
    """Tests the keras layers layers.AugMix functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.AugMix", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.AugMix"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.AugMix", inputs=[])
    KERAS_LAYERS_MAPPING["layers.AugMix"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__auto_contrast() -> None:
    """Tests the keras layers layers.AutoContrast functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.AutoContrast", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.AutoContrast"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.AutoContrast", inputs=[])
    KERAS_LAYERS_MAPPING["layers.AutoContrast"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__center_crop() -> None:
    """Tests the keras layers layers.CenterCrop functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.CenterCrop", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.CenterCrop"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.CenterCrop", inputs=[])
    KERAS_LAYERS_MAPPING["layers.CenterCrop"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__cut_mix() -> None:
    """Tests the keras layers layers.CutMix functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.CutMix", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.CutMix"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.CutMix", inputs=[])
    KERAS_LAYERS_MAPPING["layers.CutMix"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__equalization() -> None:
    """Tests the keras layers layers.Equalization functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Equalization", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Equalization"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.Equalization", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Equalization"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__max_num_bounding_boxes() -> None:
    """Tests the keras layers layers.MaxNumBoundingBoxes functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.MaxNumBoundingBoxes", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.MaxNumBoundingBoxes"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.MaxNumBoundingBoxes", inputs=[])
    KERAS_LAYERS_MAPPING["layers.MaxNumBoundingBoxes"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__mix_up() -> None:
    """Tests the keras layers layers.MixUp functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.MixUp", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.MixUp"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.MixUp", inputs=[])
    KERAS_LAYERS_MAPPING["layers.MixUp"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__rand_augment() -> None:
    """Tests the keras layers layers.RandAugment functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.RandAugment", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.RandAugment"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.RandAugment", inputs=[])
    KERAS_LAYERS_MAPPING["layers.RandAugment"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__random_brightness() -> None:
    """Tests the keras layers layers.RandomBrightness functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.RandomBrightness", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.RandomBrightness"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.RandomBrightness", inputs=[])
    KERAS_LAYERS_MAPPING["layers.RandomBrightness"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__random_color_degeneration() -> None:
    """Tests the keras layers layers.RandomColorDegeneration functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.RandomColorDegeneration", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.RandomColorDegeneration"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.RandomColorDegeneration", inputs=[])
    KERAS_LAYERS_MAPPING["layers.RandomColorDegeneration"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__random_color_jitter() -> None:
    """Tests the keras layers layers.RandomColorJitter functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.RandomColorJitter", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.RandomColorJitter"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.RandomColorJitter", inputs=[])
    KERAS_LAYERS_MAPPING["layers.RandomColorJitter"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__random_contrast() -> None:
    """Tests the keras layers layers.RandomContrast functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.RandomContrast", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.RandomContrast"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.RandomContrast", inputs=[])
    KERAS_LAYERS_MAPPING["layers.RandomContrast"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__random_crop() -> None:
    """Tests the keras layers layers.RandomCrop functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.RandomCrop", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.RandomCrop"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.RandomCrop", inputs=[])
    KERAS_LAYERS_MAPPING["layers.RandomCrop"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__random_elastic_transform() -> None:
    """Tests the keras layers layers.RandomElasticTransform functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.RandomElasticTransform", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.RandomElasticTransform"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.RandomElasticTransform", inputs=[])
    KERAS_LAYERS_MAPPING["layers.RandomElasticTransform"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__random_erasing() -> None:
    """Tests the keras layers layers.RandomErasing functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.RandomErasing", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.RandomErasing"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.RandomErasing", inputs=[])
    KERAS_LAYERS_MAPPING["layers.RandomErasing"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__random_flip() -> None:
    """Tests the keras layers layers.RandomFlip functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.RandomFlip", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.RandomFlip"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.RandomFlip", inputs=[])
    KERAS_LAYERS_MAPPING["layers.RandomFlip"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__random_gaussian_blur() -> None:
    """Tests the keras layers layers.RandomGaussianBlur functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.RandomGaussianBlur", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.RandomGaussianBlur"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.RandomGaussianBlur", inputs=[])
    KERAS_LAYERS_MAPPING["layers.RandomGaussianBlur"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__random_grayscale() -> None:
    """Tests the keras layers layers.RandomGrayscale functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.RandomGrayscale", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.RandomGrayscale"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.RandomGrayscale", inputs=[])
    KERAS_LAYERS_MAPPING["layers.RandomGrayscale"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__random_hue() -> None:
    """Tests the keras layers layers.RandomHue functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.RandomHue", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.RandomHue"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.RandomHue", inputs=[])
    KERAS_LAYERS_MAPPING["layers.RandomHue"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__random_invert() -> None:
    """Tests the keras layers layers.RandomInvert functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.RandomInvert", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.RandomInvert"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.RandomInvert", inputs=[])
    KERAS_LAYERS_MAPPING["layers.RandomInvert"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__random_perspective() -> None:
    """Tests the keras layers layers.RandomPerspective functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.RandomPerspective", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.RandomPerspective"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.RandomPerspective", inputs=[])
    KERAS_LAYERS_MAPPING["layers.RandomPerspective"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__random_posterization() -> None:
    """Tests the keras layers layers.RandomPosterization functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.RandomPosterization", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.RandomPosterization"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.RandomPosterization", inputs=[])
    KERAS_LAYERS_MAPPING["layers.RandomPosterization"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__random_rotation() -> None:
    """Tests the keras layers layers.RandomRotation functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.RandomRotation", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.RandomRotation"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.RandomRotation", inputs=[])
    KERAS_LAYERS_MAPPING["layers.RandomRotation"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__random_saturation() -> None:
    """Tests the keras layers layers.RandomSaturation functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.RandomSaturation", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.RandomSaturation"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.RandomSaturation", inputs=[])
    KERAS_LAYERS_MAPPING["layers.RandomSaturation"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__random_sharpness() -> None:
    """Tests the keras layers layers.RandomSharpness functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.RandomSharpness", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.RandomSharpness"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.RandomSharpness", inputs=[])
    KERAS_LAYERS_MAPPING["layers.RandomSharpness"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__random_shear() -> None:
    """Tests the keras layers layers.RandomShear functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.RandomShear", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.RandomShear"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.RandomShear", inputs=[])
    KERAS_LAYERS_MAPPING["layers.RandomShear"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__random_translation() -> None:
    """Tests the keras layers layers.RandomTranslation functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.RandomTranslation", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.RandomTranslation"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.RandomTranslation", inputs=[])
    KERAS_LAYERS_MAPPING["layers.RandomTranslation"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__random_zoom() -> None:
    """Tests the keras layers layers.RandomZoom functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.RandomZoom", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.RandomZoom"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.RandomZoom", inputs=[])
    KERAS_LAYERS_MAPPING["layers.RandomZoom"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__resizing() -> None:
    """Tests the keras layers layers.Resizing functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Resizing", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Resizing"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.Resizing", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Resizing"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__solarization() -> None:
    """Tests the keras layers layers.Solarization functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Solarization", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Solarization"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.Solarization", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Solarization"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__integer_lookup() -> None:
    """Tests the keras layers layers.IntegerLookup functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.IntegerLookup", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.IntegerLookup"](builder, node)
    assert builder.graph.nodes[-1].op_type == "CategoryMapper"

    node_empty = TFNode("n2", "layers.IntegerLookup", inputs=[])
    KERAS_LAYERS_MAPPING["layers.IntegerLookup"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__mel_spectrogram() -> None:
    """Tests the keras layers layers.MelSpectrogram functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.MelSpectrogram", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.MelSpectrogram"](builder, node)
    assert builder.graph.nodes[-1].op_type == "MatMul"

    node_empty = TFNode("n2", "layers.MelSpectrogram", inputs=[])
    KERAS_LAYERS_MAPPING["layers.MelSpectrogram"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__normalization() -> None:
    """Tests the keras layers layers.Normalization functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Normalization", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Normalization"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.Normalization", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Normalization"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__pipeline() -> None:
    """Tests the keras layers layers.Pipeline functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Pipeline", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Pipeline"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.Pipeline", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Pipeline"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__rescaling() -> None:
    """Tests the keras layers layers.Rescaling functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Rescaling", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Rescaling"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.Rescaling", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Rescaling"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers_stft_spectrogram() -> None:
    """Tests the keras layers layers.STFTSpectrogram functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.STFTSpectrogram", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.STFTSpectrogram"](builder, node)
    assert builder.graph.nodes[-1].op_type == "STFT"

    node_empty = TFNode("n2", "layers.STFTSpectrogram", inputs=[])
    KERAS_LAYERS_MAPPING["layers.STFTSpectrogram"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__string_lookup() -> None:
    """Tests the keras layers layers.StringLookup functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.StringLookup", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.StringLookup"](builder, node)
    assert builder.graph.nodes[-1].op_type == "CategoryMapper"

    node_empty = TFNode("n2", "layers.StringLookup", inputs=[])
    KERAS_LAYERS_MAPPING["layers.StringLookup"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__text_vectorization() -> None:
    """Tests the keras layers layers.TextVectorization functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.TextVectorization", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.TextVectorization"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.TextVectorization", inputs=[])
    KERAS_LAYERS_MAPPING["layers.TextVectorization"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__activity_regularization() -> None:
    """Tests the keras layers layers.ActivityRegularization functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.ActivityRegularization", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.ActivityRegularization"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.ActivityRegularization", inputs=[])
    KERAS_LAYERS_MAPPING["layers.ActivityRegularization"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__alpha_dropout() -> None:
    """Tests the keras layers layers.AlphaDropout functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.AlphaDropout", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.AlphaDropout"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.AlphaDropout", inputs=[])
    KERAS_LAYERS_MAPPING["layers.AlphaDropout"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__dropout() -> None:
    """Tests the keras layers layers.Dropout functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Dropout", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Dropout"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.Dropout", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Dropout"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__gaussian_dropout() -> None:
    """Tests the keras layers layers.GaussianDropout functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.GaussianDropout", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.GaussianDropout"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.GaussianDropout", inputs=[])
    KERAS_LAYERS_MAPPING["layers.GaussianDropout"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__gaussian_noise() -> None:
    """Tests the keras layers layers.GaussianNoise functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.GaussianNoise", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.GaussianNoise"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.GaussianNoise", inputs=[])
    KERAS_LAYERS_MAPPING["layers.GaussianNoise"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__spatial_dropout1_d() -> None:
    """Tests the keras layers layers.SpatialDropout1D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.SpatialDropout1D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.SpatialDropout1D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.SpatialDropout1D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.SpatialDropout1D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__spatial_dropout2_d() -> None:
    """Tests the keras layers layers.SpatialDropout2D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.SpatialDropout2D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.SpatialDropout2D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.SpatialDropout2D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.SpatialDropout2D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__spatial_dropout3_d() -> None:
    """Tests the keras layers layers.SpatialDropout3D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.SpatialDropout3D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.SpatialDropout3D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.SpatialDropout3D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.SpatialDropout3D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__cropping1_d() -> None:
    """Tests the keras layers layers.Cropping1D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Cropping1D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Cropping1D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Slice"

    node_empty = TFNode("n2", "layers.Cropping1D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Cropping1D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__cropping2_d() -> None:
    """Tests the keras layers layers.Cropping2D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Cropping2D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Cropping2D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Slice"

    node_empty = TFNode("n2", "layers.Cropping2D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Cropping2D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__cropping3_d() -> None:
    """Tests the keras layers layers.Cropping3D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Cropping3D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Cropping3D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Slice"

    node_empty = TFNode("n2", "layers.Cropping3D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Cropping3D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__flatten() -> None:
    """Tests the keras layers layers.Flatten functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Flatten", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Flatten"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.Flatten", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Flatten"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__permute() -> None:
    """Tests the keras layers layers.Permute functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Permute", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Permute"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.Permute", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Permute"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__repeat_vector() -> None:
    """Tests the keras layers layers.RepeatVector functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.RepeatVector", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.RepeatVector"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Tile"

    node_empty = TFNode("n2", "layers.RepeatVector", inputs=[])
    KERAS_LAYERS_MAPPING["layers.RepeatVector"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__reshape() -> None:
    """Tests the keras layers layers.Reshape functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Reshape", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Reshape"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.Reshape", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Reshape"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__up_sampling1_d() -> None:
    """Tests the keras layers layers.UpSampling1D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.UpSampling1D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.UpSampling1D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Resize"

    node_empty = TFNode("n2", "layers.UpSampling1D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.UpSampling1D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__up_sampling2_d() -> None:
    """Tests the keras layers layers.UpSampling2D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.UpSampling2D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.UpSampling2D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Resize"

    node_empty = TFNode("n2", "layers.UpSampling2D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.UpSampling2D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__up_sampling3_d() -> None:
    """Tests the keras layers layers.UpSampling3D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.UpSampling3D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.UpSampling3D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Resize"

    node_empty = TFNode("n2", "layers.UpSampling3D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.UpSampling3D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__zero_padding1_d() -> None:
    """Tests the keras layers layers.ZeroPadding1D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.ZeroPadding1D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.ZeroPadding1D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Pad"

    node_empty = TFNode("n2", "layers.ZeroPadding1D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.ZeroPadding1D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__zero_padding2_d() -> None:
    """Tests the keras layers layers.ZeroPadding2D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.ZeroPadding2D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.ZeroPadding2D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Pad"

    node_empty = TFNode("n2", "layers.ZeroPadding2D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.ZeroPadding2D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__zero_padding3_d() -> None:
    """Tests the keras layers layers.ZeroPadding3D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.ZeroPadding3D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.ZeroPadding3D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Pad"

    node_empty = TFNode("n2", "layers.ZeroPadding3D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.ZeroPadding3D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__bidirectional() -> None:
    """Tests the keras layers layers.Bidirectional functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Bidirectional", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Bidirectional"](builder, node)
    assert len(builder.graph.nodes) > 0

    node_empty = TFNode("n2", "layers.Bidirectional", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Bidirectional"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__conv_lstm1_d() -> None:
    """Tests the keras layers layers.ConvLSTM1D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.ConvLSTM1D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.ConvLSTM1D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.ConvLSTM1D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.ConvLSTM1D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__conv_lstm2_d() -> None:
    """Tests the keras layers layers.ConvLSTM2D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.ConvLSTM2D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.ConvLSTM2D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.ConvLSTM2D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.ConvLSTM2D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__conv_lstm3_d() -> None:
    """Tests the keras layers layers.ConvLSTM3D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.ConvLSTM3D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.ConvLSTM3D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.ConvLSTM3D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.ConvLSTM3D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers_gru() -> None:
    """Tests the keras layers layers.GRU functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.GRU", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.GRU"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.GRU", inputs=[])
    KERAS_LAYERS_MAPPING["layers.GRU"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers_gru_cell() -> None:
    """Tests the keras layers layers.GRUCell functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.GRUCell", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.GRUCell"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.GRUCell", inputs=[])
    KERAS_LAYERS_MAPPING["layers.GRUCell"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers_lstm() -> None:
    """Tests the keras layers layers.LSTM functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.LSTM", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.LSTM"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.LSTM", inputs=[])
    KERAS_LAYERS_MAPPING["layers.LSTM"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers_lstm_cell() -> None:
    """Tests the keras layers layers.LSTMCell functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.LSTMCell", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.LSTMCell"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.LSTMCell", inputs=[])
    KERAS_LAYERS_MAPPING["layers.LSTMCell"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers_rnn() -> None:
    """Tests the keras layers layers.RNN functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.RNN", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.RNN"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.RNN", inputs=[])
    KERAS_LAYERS_MAPPING["layers.RNN"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__simple_rnn() -> None:
    """Tests the keras layers layers.SimpleRNN functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.SimpleRNN", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.SimpleRNN"](builder, node)
    assert len(builder.graph.nodes) > 0

    node_empty = TFNode("n2", "layers.SimpleRNN", inputs=[])
    KERAS_LAYERS_MAPPING["layers.SimpleRNN"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__simple_rnn_cell() -> None:
    """Tests the keras layers layers.SimpleRNNCell functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.SimpleRNNCell", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.SimpleRNNCell"](builder, node)
    assert len(builder.graph.nodes) > 0

    node_empty = TFNode("n2", "layers.SimpleRNNCell", inputs=[])
    KERAS_LAYERS_MAPPING["layers.SimpleRNNCell"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__stacked_rnn_cells() -> None:
    """Tests the keras layers layers.StackedRNNCells functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.StackedRNNCells", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.StackedRNNCells"](builder, node)
    assert len(builder.graph.nodes) > 0

    node_empty = TFNode("n2", "layers.StackedRNNCells", inputs=[])
    KERAS_LAYERS_MAPPING["layers.StackedRNNCells"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__time_distributed() -> None:
    """Tests the keras layers layers.TimeDistributed functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.TimeDistributed", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.TimeDistributed"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.TimeDistributed", inputs=[])
    KERAS_LAYERS_MAPPING["layers.TimeDistributed"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__flax_layer() -> None:
    """Tests the keras layers layers.FlaxLayer functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.FlaxLayer", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.FlaxLayer"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.FlaxLayer", inputs=[])
    KERAS_LAYERS_MAPPING["layers.FlaxLayer"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__jax_layer() -> None:
    """Tests the keras layers layers.JaxLayer functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.JaxLayer", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.JaxLayer"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.JaxLayer", inputs=[])
    KERAS_LAYERS_MAPPING["layers.JaxLayer"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__torch_module_wrapper() -> None:
    """Tests the keras layers layers.TorchModuleWrapper functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.TorchModuleWrapper", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.TorchModuleWrapper"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.TorchModuleWrapper", inputs=[])
    KERAS_LAYERS_MAPPING["layers.TorchModuleWrapper"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"
