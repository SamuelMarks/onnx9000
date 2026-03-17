"""Tests the keras layers module functionality."""

from onnx9000.converters.tf.builder import TFToONNXGraphBuilder
from onnx9000.converters.tf.keras_layers import KERAS_LAYERS_MAPPING
from onnx9000.converters.tf.parsers import TFNode


def test_keras_layers_simple() -> None:
    """Tests the keras layers simple functionality."""
    builder = TFToONNXGraphBuilder()
    custom_mapped = {
        "Dense": "Custom_KerasDense",
        "Conv1D": "Custom_KerasConv1D",
        "Conv2D": "Custom_KerasConv2D",
        "Conv3D": "Custom_KerasConv3D",
        "SeparableConv1D": "Custom_KerasSeparableConv1D",
        "SeparableConv2D": "Custom_KerasSeparableConv2D",
        "DepthwiseConv2D": "Custom_KerasDepthwiseConv2D",
        "Conv2DTranspose": "Custom_KerasConv2DTranspose",
        "Conv3DTranspose": "Custom_KerasConv3DTranspose",
        "MaxPooling1D": "Custom_KerasMaxPooling1D",
        "MaxPooling2D": "Custom_KerasMaxPooling2D",
        "MaxPooling3D": "Custom_KerasMaxPooling3D",
        "AveragePooling1D": "Custom_KerasAvgPooling1D",
        "AveragePooling2D": "Custom_KerasAvgPooling2D",
        "AveragePooling3D": "Custom_KerasAvgPooling3D",
        "GlobalMaxPooling1D": "Custom_KerasGlobalMaxPooling1D",
        "GlobalMaxPooling2D": "Custom_KerasGlobalMaxPooling2D",
        "GlobalMaxPooling3D": "Custom_KerasGlobalMaxPooling3D",
        "GlobalAveragePooling1D": "Custom_KerasGlobalAvgPooling1D",
        "GlobalAveragePooling2D": "Custom_KerasGlobalAvgPooling2D",
        "GlobalAveragePooling3D": "Custom_KerasGlobalAvgPooling3D",
        "RNN": "Custom_KerasRNN",
        "Bidirectional": "Custom_KerasBidirectional",
        "Activation": "Custom_KerasActivation",
        "RepeatVector": "Custom_KerasRepeatVector",
        "Dot": "Custom_KerasDot",
    }
    for op, expected in custom_mapped.items():
        KERAS_LAYERS_MAPPING[op](builder, TFNode("n", op, inputs=["a"]))
        assert builder.graph.nodes[-1].op_type == expected
    direct_maps = {
        "SimpleRNN": "RNN",
        "LSTM": "LSTM",
        "GRU": "GRU",
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
    for op, onnx_op in direct_maps.items():
        KERAS_LAYERS_MAPPING[op](builder, TFNode("n", op, inputs=["a"]))
        assert builder.graph.nodes[-1].op_type == onnx_op


def test_keras_layers_dimensional() -> None:
    """Tests the keras layers dimensional functionality."""
    builder = TFToONNXGraphBuilder()
    for op_prefix in ["SpatialDropout", "Cropping", "UpSampling", "ZeroPadding"]:
        for dim in ["1D", "2D", "3D"]:
            op = f"{op_prefix}{dim}"
            KERAS_LAYERS_MAPPING[op](builder, TFNode("n", op, inputs=["a"]))
            assert builder.graph.nodes[-1].op_type == f"Custom_Keras{op}"
