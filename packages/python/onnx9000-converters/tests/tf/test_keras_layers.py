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
        "RNN": "RNN",
        "Bidirectional": "RNN",
        "RepeatVector": "Tile",
        "Dot": "MatMul",
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


def test_keras_layers_missing_variable() -> None:
    """Tests the keras layers Variable functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "Variable", inputs=["a"])

    KERAS_LAYERS_MAPPING["Variable"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "Variable", inputs=[])
    KERAS_LAYERS_MAPPING["Variable"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_device() -> None:
    """Tests the keras layers device functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "device", inputs=["a"])

    KERAS_LAYERS_MAPPING["device"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "device", inputs=[])
    KERAS_LAYERS_MAPPING["device"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_name_scope() -> None:
    """Tests the keras layers name_scope functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "name_scope", inputs=["a"])

    KERAS_LAYERS_MAPPING["name_scope"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "name_scope", inputs=[])
    KERAS_LAYERS_MAPPING["name_scope"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_keras_tensor() -> None:
    """Tests the keras layers KerasTensor functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "KerasTensor", inputs=["a"])

    KERAS_LAYERS_MAPPING["KerasTensor"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "KerasTensor", inputs=[])
    KERAS_LAYERS_MAPPING["KerasTensor"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_remat_scope() -> None:
    """Tests the keras layers RematScope functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "RematScope", inputs=["a"])

    KERAS_LAYERS_MAPPING["RematScope"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "RematScope", inputs=[])
    KERAS_LAYERS_MAPPING["RematScope"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_remat() -> None:
    """Tests the keras layers remat functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "remat", inputs=["a"])

    KERAS_LAYERS_MAPPING["remat"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "remat", inputs=[])
    KERAS_LAYERS_MAPPING["remat"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_stateless_scope() -> None:
    """Tests the keras layers StatelessScope functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "StatelessScope", inputs=["a"])

    KERAS_LAYERS_MAPPING["StatelessScope"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "StatelessScope", inputs=[])
    KERAS_LAYERS_MAPPING["StatelessScope"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_symbolic_scope() -> None:
    """Tests the keras layers SymbolicScope functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "SymbolicScope", inputs=["a"])

    KERAS_LAYERS_MAPPING["SymbolicScope"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "SymbolicScope", inputs=[])
    KERAS_LAYERS_MAPPING["SymbolicScope"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_d_type_policy() -> None:
    """Tests the keras layers DTypePolicy functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "DTypePolicy", inputs=["a"])

    KERAS_LAYERS_MAPPING["DTypePolicy"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "DTypePolicy", inputs=[])
    KERAS_LAYERS_MAPPING["DTypePolicy"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_float_d_type_policy() -> None:
    """Tests the keras layers FloatDTypePolicy functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "FloatDTypePolicy", inputs=["a"])

    KERAS_LAYERS_MAPPING["FloatDTypePolicy"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "FloatDTypePolicy", inputs=[])
    KERAS_LAYERS_MAPPING["FloatDTypePolicy"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_initializer() -> None:
    """Tests the keras layers Initializer functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "Initializer", inputs=["a"])

    KERAS_LAYERS_MAPPING["Initializer"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "Initializer", inputs=[])
    KERAS_LAYERS_MAPPING["Initializer"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_input() -> None:
    """Tests the keras layers Input functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "Input", inputs=["a"])

    KERAS_LAYERS_MAPPING["Input"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "Input", inputs=[])
    KERAS_LAYERS_MAPPING["Input"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_input_spec() -> None:
    """Tests the keras layers InputSpec functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "InputSpec", inputs=["a"])

    KERAS_LAYERS_MAPPING["InputSpec"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "InputSpec", inputs=[])
    KERAS_LAYERS_MAPPING["InputSpec"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layer_ext() -> None:
    """Tests the keras layers Layer functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "Layer", inputs=["a"])

    KERAS_LAYERS_MAPPING["Layer"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "Layer", inputs=[])
    KERAS_LAYERS_MAPPING["Layer"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_loss() -> None:
    """Tests the keras layers Loss functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "Loss", inputs=["a"])

    KERAS_LAYERS_MAPPING["Loss"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "Loss", inputs=[])
    KERAS_LAYERS_MAPPING["Loss"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metric() -> None:
    """Tests the keras layers Metric functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "Metric", inputs=["a"])

    KERAS_LAYERS_MAPPING["Metric"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "Metric", inputs=[])
    KERAS_LAYERS_MAPPING["Metric"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_model() -> None:
    """Tests the keras layers Model functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "Model", inputs=["a"])

    KERAS_LAYERS_MAPPING["Model"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "Model", inputs=[])
    KERAS_LAYERS_MAPPING["Model"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_sequential() -> None:
    """Tests the keras layers Sequential functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "Sequential", inputs=["a"])

    KERAS_LAYERS_MAPPING["Sequential"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "Sequential", inputs=[])
    KERAS_LAYERS_MAPPING["Sequential"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_function() -> None:
    """Tests the keras layers Function functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "Function", inputs=["a"])

    KERAS_LAYERS_MAPPING["Function"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "Function", inputs=[])
    KERAS_LAYERS_MAPPING["Function"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_operation() -> None:
    """Tests the keras layers Operation functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "Operation", inputs=["a"])

    KERAS_LAYERS_MAPPING["Operation"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "Operation", inputs=[])
    KERAS_LAYERS_MAPPING["Operation"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_optimizer() -> None:
    """Tests the keras layers Optimizer functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "Optimizer", inputs=["a"])

    KERAS_LAYERS_MAPPING["Optimizer"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "Optimizer", inputs=[])
    KERAS_LAYERS_MAPPING["Optimizer"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_quantizer() -> None:
    """Tests the keras layers Quantizer functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "Quantizer", inputs=["a"])

    KERAS_LAYERS_MAPPING["Quantizer"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "Quantizer", inputs=[])
    KERAS_LAYERS_MAPPING["Quantizer"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_regularizer() -> None:
    """Tests the keras layers Regularizer functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "Regularizer", inputs=["a"])

    KERAS_LAYERS_MAPPING["Regularizer"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "Regularizer", inputs=[])
    KERAS_LAYERS_MAPPING["Regularizer"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_version() -> None:
    """Tests the keras layers version functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "version", inputs=["a"])

    KERAS_LAYERS_MAPPING["version"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "version", inputs=[])
    KERAS_LAYERS_MAPPING["version"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_saving__keras_file_editor() -> None:
    """Tests the keras layers saving.KerasFileEditor functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "saving.KerasFileEditor", inputs=["a"])

    KERAS_LAYERS_MAPPING["saving.KerasFileEditor"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "saving.KerasFileEditor", inputs=[])
    KERAS_LAYERS_MAPPING["saving.KerasFileEditor"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_saving__custom_object_scope() -> None:
    """Tests the keras layers saving.CustomObjectScope functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "saving.CustomObjectScope", inputs=["a"])

    KERAS_LAYERS_MAPPING["saving.CustomObjectScope"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "saving.CustomObjectScope", inputs=[])
    KERAS_LAYERS_MAPPING["saving.CustomObjectScope"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_saving_custom_object_scope() -> None:
    """Tests the keras layers saving.custom_object_scope functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "saving.custom_object_scope", inputs=["a"])

    KERAS_LAYERS_MAPPING["saving.custom_object_scope"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "saving.custom_object_scope", inputs=[])
    KERAS_LAYERS_MAPPING["saving.custom_object_scope"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_saving_get_custom_objects() -> None:
    """Tests the keras layers saving.get_custom_objects functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "saving.get_custom_objects", inputs=["a"])

    KERAS_LAYERS_MAPPING["saving.get_custom_objects"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "saving.get_custom_objects", inputs=[])
    KERAS_LAYERS_MAPPING["saving.get_custom_objects"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_saving_get_registered_name() -> None:
    """Tests the keras layers saving.get_registered_name functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "saving.get_registered_name", inputs=["a"])

    KERAS_LAYERS_MAPPING["saving.get_registered_name"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "saving.get_registered_name", inputs=[])
    KERAS_LAYERS_MAPPING["saving.get_registered_name"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_saving_get_registered_object() -> None:
    """Tests the keras layers saving.get_registered_object functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "saving.get_registered_object", inputs=["a"])

    KERAS_LAYERS_MAPPING["saving.get_registered_object"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "saving.get_registered_object", inputs=[])
    KERAS_LAYERS_MAPPING["saving.get_registered_object"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_saving_register_keras_serializable() -> None:
    """Tests the keras layers saving.register_keras_serializable functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "saving.register_keras_serializable", inputs=["a"])

    KERAS_LAYERS_MAPPING["saving.register_keras_serializable"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "saving.register_keras_serializable", inputs=[])
    KERAS_LAYERS_MAPPING["saving.register_keras_serializable"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_saving_load_model() -> None:
    """Tests the keras layers saving.load_model functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "saving.load_model", inputs=["a"])

    KERAS_LAYERS_MAPPING["saving.load_model"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "saving.load_model", inputs=[])
    KERAS_LAYERS_MAPPING["saving.load_model"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_saving_load_weights() -> None:
    """Tests the keras layers saving.load_weights functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "saving.load_weights", inputs=["a"])

    KERAS_LAYERS_MAPPING["saving.load_weights"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "saving.load_weights", inputs=[])
    KERAS_LAYERS_MAPPING["saving.load_weights"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_saving_save_model() -> None:
    """Tests the keras layers saving.save_model functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "saving.save_model", inputs=["a"])

    KERAS_LAYERS_MAPPING["saving.save_model"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "saving.save_model", inputs=[])
    KERAS_LAYERS_MAPPING["saving.save_model"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_saving_save_weights() -> None:
    """Tests the keras layers saving.save_weights functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "saving.save_weights", inputs=["a"])

    KERAS_LAYERS_MAPPING["saving.save_weights"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "saving.save_weights", inputs=[])
    KERAS_LAYERS_MAPPING["saving.save_weights"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_saving_deserialize_keras_object() -> None:
    """Tests the keras layers saving.deserialize_keras_object functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "saving.deserialize_keras_object", inputs=["a"])

    KERAS_LAYERS_MAPPING["saving.deserialize_keras_object"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "saving.deserialize_keras_object", inputs=[])
    KERAS_LAYERS_MAPPING["saving.deserialize_keras_object"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_saving_serialize_keras_object() -> None:
    """Tests the keras layers saving.serialize_keras_object functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "saving.serialize_keras_object", inputs=["a"])

    KERAS_LAYERS_MAPPING["saving.serialize_keras_object"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "saving.serialize_keras_object", inputs=[])
    KERAS_LAYERS_MAPPING["saving.serialize_keras_object"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_export__export_archive() -> None:
    """Tests the keras layers export.ExportArchive functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "export.ExportArchive", inputs=["a"])

    KERAS_LAYERS_MAPPING["export.ExportArchive"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "export.ExportArchive", inputs=[])
    KERAS_LAYERS_MAPPING["export.ExportArchive"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_preprocessing_image_dataset_from_directory() -> None:
    """Tests the keras layers preprocessing.image_dataset_from_directory functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "preprocessing.image_dataset_from_directory", inputs=["a"])

    KERAS_LAYERS_MAPPING["preprocessing.image_dataset_from_directory"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "preprocessing.image_dataset_from_directory", inputs=[])
    KERAS_LAYERS_MAPPING["preprocessing.image_dataset_from_directory"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_preprocessing_text_dataset_from_directory() -> None:
    """Tests the keras layers preprocessing.text_dataset_from_directory functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "preprocessing.text_dataset_from_directory", inputs=["a"])

    KERAS_LAYERS_MAPPING["preprocessing.text_dataset_from_directory"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "preprocessing.text_dataset_from_directory", inputs=[])
    KERAS_LAYERS_MAPPING["preprocessing.text_dataset_from_directory"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_preprocessing_timeseries_dataset_from_array() -> None:
    """Tests the keras layers preprocessing.timeseries_dataset_from_array functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "preprocessing.timeseries_dataset_from_array", inputs=["a"])

    KERAS_LAYERS_MAPPING["preprocessing.timeseries_dataset_from_array"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "preprocessing.timeseries_dataset_from_array", inputs=[])
    KERAS_LAYERS_MAPPING["preprocessing.timeseries_dataset_from_array"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_preprocessing_image_array_to_img() -> None:
    """Tests the keras layers preprocessing.image.array_to_img functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "preprocessing.image.array_to_img", inputs=["a"])

    KERAS_LAYERS_MAPPING["preprocessing.image.array_to_img"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "preprocessing.image.array_to_img", inputs=[])
    KERAS_LAYERS_MAPPING["preprocessing.image.array_to_img"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_preprocessing_image_img_to_array() -> None:
    """Tests the keras layers preprocessing.image.img_to_array functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "preprocessing.image.img_to_array", inputs=["a"])

    KERAS_LAYERS_MAPPING["preprocessing.image.img_to_array"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "preprocessing.image.img_to_array", inputs=[])
    KERAS_LAYERS_MAPPING["preprocessing.image.img_to_array"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_preprocessing_image_load_img() -> None:
    """Tests the keras layers preprocessing.image.load_img functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "preprocessing.image.load_img", inputs=["a"])

    KERAS_LAYERS_MAPPING["preprocessing.image.load_img"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "preprocessing.image.load_img", inputs=[])
    KERAS_LAYERS_MAPPING["preprocessing.image.load_img"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_preprocessing_image_save_img() -> None:
    """Tests the keras layers preprocessing.image.save_img functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "preprocessing.image.save_img", inputs=["a"])

    KERAS_LAYERS_MAPPING["preprocessing.image.save_img"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "preprocessing.image.save_img", inputs=[])
    KERAS_LAYERS_MAPPING["preprocessing.image.save_img"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_preprocessing_image_smart_resize() -> None:
    """Tests the keras layers preprocessing.image.smart_resize functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "preprocessing.image.smart_resize", inputs=["a"])

    KERAS_LAYERS_MAPPING["preprocessing.image.smart_resize"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "preprocessing.image.smart_resize", inputs=[])
    KERAS_LAYERS_MAPPING["preprocessing.image.smart_resize"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_preprocessing_sequence_pad_sequences() -> None:
    """Tests the keras layers preprocessing.sequence.pad_sequences functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "preprocessing.sequence.pad_sequences", inputs=["a"])

    KERAS_LAYERS_MAPPING["preprocessing.sequence.pad_sequences"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "preprocessing.sequence.pad_sequences", inputs=[])
    KERAS_LAYERS_MAPPING["preprocessing.sequence.pad_sequences"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_distillation__distillation_loss() -> None:
    """Tests the keras layers distillation.DistillationLoss functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "distillation.DistillationLoss", inputs=["a"])

    KERAS_LAYERS_MAPPING["distillation.DistillationLoss"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "distillation.DistillationLoss", inputs=[])
    KERAS_LAYERS_MAPPING["distillation.DistillationLoss"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_distillation__feature_distillation() -> None:
    """Tests the keras layers distillation.FeatureDistillation functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "distillation.FeatureDistillation", inputs=["a"])

    KERAS_LAYERS_MAPPING["distillation.FeatureDistillation"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "distillation.FeatureDistillation", inputs=[])
    KERAS_LAYERS_MAPPING["distillation.FeatureDistillation"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_distillation__logits_distillation() -> None:
    """Tests the keras layers distillation.LogitsDistillation functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "distillation.LogitsDistillation", inputs=["a"])

    KERAS_LAYERS_MAPPING["distillation.LogitsDistillation"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "distillation.LogitsDistillation", inputs=[])
    KERAS_LAYERS_MAPPING["distillation.LogitsDistillation"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_distillation__distiller() -> None:
    """Tests the keras layers distillation.Distiller functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "distillation.Distiller", inputs=["a"])

    KERAS_LAYERS_MAPPING["distillation.Distiller"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "distillation.Distiller", inputs=[])
    KERAS_LAYERS_MAPPING["distillation.Distiller"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_models_clone_model() -> None:
    """Tests the keras layers models.clone_model functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "models.clone_model", inputs=["a"])

    KERAS_LAYERS_MAPPING["models.clone_model"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "models.clone_model", inputs=[])
    KERAS_LAYERS_MAPPING["models.clone_model"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_models__model() -> None:
    """Tests the keras layers models.Model functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "models.Model", inputs=["a"])

    KERAS_LAYERS_MAPPING["models.Model"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "models.Model", inputs=[])
    KERAS_LAYERS_MAPPING["models.Model"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_models_model_from_json() -> None:
    """Tests the keras layers models.model_from_json functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "models.model_from_json", inputs=["a"])

    KERAS_LAYERS_MAPPING["models.model_from_json"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "models.model_from_json", inputs=[])
    KERAS_LAYERS_MAPPING["models.model_from_json"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_models__sequential() -> None:
    """Tests the keras layers models.Sequential functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "models.Sequential", inputs=["a"])

    KERAS_LAYERS_MAPPING["models.Sequential"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "models.Sequential", inputs=[])
    KERAS_LAYERS_MAPPING["models.Sequential"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_models_load_model() -> None:
    """Tests the keras layers models.load_model functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "models.load_model", inputs=["a"])

    KERAS_LAYERS_MAPPING["models.load_model"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "models.load_model", inputs=[])
    KERAS_LAYERS_MAPPING["models.load_model"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_models_save_model() -> None:
    """Tests the keras layers models.save_model functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "models.save_model", inputs=["a"])

    KERAS_LAYERS_MAPPING["models.save_model"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "models.save_model", inputs=[])
    KERAS_LAYERS_MAPPING["models.save_model"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications__conv_ne_xt_base() -> None:
    """Tests the keras layers applications.ConvNeXtBase functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.ConvNeXtBase", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.ConvNeXtBase"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.ConvNeXtBase", inputs=[])
    KERAS_LAYERS_MAPPING["applications.ConvNeXtBase"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications__conv_ne_xt_large() -> None:
    """Tests the keras layers applications.ConvNeXtLarge functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.ConvNeXtLarge", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.ConvNeXtLarge"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.ConvNeXtLarge", inputs=[])
    KERAS_LAYERS_MAPPING["applications.ConvNeXtLarge"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications__conv_ne_xt_small() -> None:
    """Tests the keras layers applications.ConvNeXtSmall functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.ConvNeXtSmall", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.ConvNeXtSmall"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.ConvNeXtSmall", inputs=[])
    KERAS_LAYERS_MAPPING["applications.ConvNeXtSmall"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications__conv_ne_xt_tiny() -> None:
    """Tests the keras layers applications.ConvNeXtTiny functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.ConvNeXtTiny", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.ConvNeXtTiny"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.ConvNeXtTiny", inputs=[])
    KERAS_LAYERS_MAPPING["applications.ConvNeXtTiny"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications__conv_ne_xt_x_large() -> None:
    """Tests the keras layers applications.ConvNeXtXLarge functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.ConvNeXtXLarge", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.ConvNeXtXLarge"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.ConvNeXtXLarge", inputs=[])
    KERAS_LAYERS_MAPPING["applications.ConvNeXtXLarge"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications__dense_net121() -> None:
    """Tests the keras layers applications.DenseNet121 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.DenseNet121", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.DenseNet121"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.DenseNet121", inputs=[])
    KERAS_LAYERS_MAPPING["applications.DenseNet121"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications__dense_net169() -> None:
    """Tests the keras layers applications.DenseNet169 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.DenseNet169", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.DenseNet169"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.DenseNet169", inputs=[])
    KERAS_LAYERS_MAPPING["applications.DenseNet169"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications__dense_net201() -> None:
    """Tests the keras layers applications.DenseNet201 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.DenseNet201", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.DenseNet201"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.DenseNet201", inputs=[])
    KERAS_LAYERS_MAPPING["applications.DenseNet201"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications__efficient_net_b0() -> None:
    """Tests the keras layers applications.EfficientNetB0 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.EfficientNetB0", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.EfficientNetB0"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.EfficientNetB0", inputs=[])
    KERAS_LAYERS_MAPPING["applications.EfficientNetB0"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications__efficient_net_b1() -> None:
    """Tests the keras layers applications.EfficientNetB1 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.EfficientNetB1", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.EfficientNetB1"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.EfficientNetB1", inputs=[])
    KERAS_LAYERS_MAPPING["applications.EfficientNetB1"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications__efficient_net_b2() -> None:
    """Tests the keras layers applications.EfficientNetB2 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.EfficientNetB2", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.EfficientNetB2"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.EfficientNetB2", inputs=[])
    KERAS_LAYERS_MAPPING["applications.EfficientNetB2"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications__efficient_net_b3() -> None:
    """Tests the keras layers applications.EfficientNetB3 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.EfficientNetB3", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.EfficientNetB3"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.EfficientNetB3", inputs=[])
    KERAS_LAYERS_MAPPING["applications.EfficientNetB3"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications__efficient_net_b4() -> None:
    """Tests the keras layers applications.EfficientNetB4 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.EfficientNetB4", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.EfficientNetB4"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.EfficientNetB4", inputs=[])
    KERAS_LAYERS_MAPPING["applications.EfficientNetB4"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications__efficient_net_b5() -> None:
    """Tests the keras layers applications.EfficientNetB5 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.EfficientNetB5", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.EfficientNetB5"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.EfficientNetB5", inputs=[])
    KERAS_LAYERS_MAPPING["applications.EfficientNetB5"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications__efficient_net_b6() -> None:
    """Tests the keras layers applications.EfficientNetB6 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.EfficientNetB6", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.EfficientNetB6"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.EfficientNetB6", inputs=[])
    KERAS_LAYERS_MAPPING["applications.EfficientNetB6"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications__efficient_net_b7() -> None:
    """Tests the keras layers applications.EfficientNetB7 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.EfficientNetB7", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.EfficientNetB7"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.EfficientNetB7", inputs=[])
    KERAS_LAYERS_MAPPING["applications.EfficientNetB7"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications__efficient_net_v2_b0() -> None:
    """Tests the keras layers applications.EfficientNetV2B0 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.EfficientNetV2B0", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.EfficientNetV2B0"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.EfficientNetV2B0", inputs=[])
    KERAS_LAYERS_MAPPING["applications.EfficientNetV2B0"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications__efficient_net_v2_b1() -> None:
    """Tests the keras layers applications.EfficientNetV2B1 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.EfficientNetV2B1", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.EfficientNetV2B1"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.EfficientNetV2B1", inputs=[])
    KERAS_LAYERS_MAPPING["applications.EfficientNetV2B1"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications__efficient_net_v2_b2() -> None:
    """Tests the keras layers applications.EfficientNetV2B2 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.EfficientNetV2B2", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.EfficientNetV2B2"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.EfficientNetV2B2", inputs=[])
    KERAS_LAYERS_MAPPING["applications.EfficientNetV2B2"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications__efficient_net_v2_b3() -> None:
    """Tests the keras layers applications.EfficientNetV2B3 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.EfficientNetV2B3", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.EfficientNetV2B3"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.EfficientNetV2B3", inputs=[])
    KERAS_LAYERS_MAPPING["applications.EfficientNetV2B3"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications__efficient_net_v2_l() -> None:
    """Tests the keras layers applications.EfficientNetV2L functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.EfficientNetV2L", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.EfficientNetV2L"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.EfficientNetV2L", inputs=[])
    KERAS_LAYERS_MAPPING["applications.EfficientNetV2L"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications__efficient_net_v2_m() -> None:
    """Tests the keras layers applications.EfficientNetV2M functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.EfficientNetV2M", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.EfficientNetV2M"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.EfficientNetV2M", inputs=[])
    KERAS_LAYERS_MAPPING["applications.EfficientNetV2M"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications__efficient_net_v2_s() -> None:
    """Tests the keras layers applications.EfficientNetV2S functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.EfficientNetV2S", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.EfficientNetV2S"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.EfficientNetV2S", inputs=[])
    KERAS_LAYERS_MAPPING["applications.EfficientNetV2S"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications__inception_res_net_v2() -> None:
    """Tests the keras layers applications.InceptionResNetV2 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.InceptionResNetV2", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.InceptionResNetV2"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.InceptionResNetV2", inputs=[])
    KERAS_LAYERS_MAPPING["applications.InceptionResNetV2"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications__inception_v3() -> None:
    """Tests the keras layers applications.InceptionV3 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.InceptionV3", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.InceptionV3"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.InceptionV3", inputs=[])
    KERAS_LAYERS_MAPPING["applications.InceptionV3"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications__mobile_net() -> None:
    """Tests the keras layers applications.MobileNet functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.MobileNet", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.MobileNet"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.MobileNet", inputs=[])
    KERAS_LAYERS_MAPPING["applications.MobileNet"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications__mobile_net_v2() -> None:
    """Tests the keras layers applications.MobileNetV2 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.MobileNetV2", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.MobileNetV2"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.MobileNetV2", inputs=[])
    KERAS_LAYERS_MAPPING["applications.MobileNetV2"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications__mobile_net_v3_large() -> None:
    """Tests the keras layers applications.MobileNetV3Large functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.MobileNetV3Large", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.MobileNetV3Large"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.MobileNetV3Large", inputs=[])
    KERAS_LAYERS_MAPPING["applications.MobileNetV3Large"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications__mobile_net_v3_small() -> None:
    """Tests the keras layers applications.MobileNetV3Small functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.MobileNetV3Small", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.MobileNetV3Small"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.MobileNetV3Small", inputs=[])
    KERAS_LAYERS_MAPPING["applications.MobileNetV3Small"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_nas_net_large() -> None:
    """Tests the keras layers applications.NASNetLarge functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.NASNetLarge", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.NASNetLarge"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.NASNetLarge", inputs=[])
    KERAS_LAYERS_MAPPING["applications.NASNetLarge"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_nas_net_mobile() -> None:
    """Tests the keras layers applications.NASNetMobile functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.NASNetMobile", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.NASNetMobile"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.NASNetMobile", inputs=[])
    KERAS_LAYERS_MAPPING["applications.NASNetMobile"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications__res_net50() -> None:
    """Tests the keras layers applications.ResNet50 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.ResNet50", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.ResNet50"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.ResNet50", inputs=[])
    KERAS_LAYERS_MAPPING["applications.ResNet50"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications__res_net101() -> None:
    """Tests the keras layers applications.ResNet101 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.ResNet101", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.ResNet101"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.ResNet101", inputs=[])
    KERAS_LAYERS_MAPPING["applications.ResNet101"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications__res_net152() -> None:
    """Tests the keras layers applications.ResNet152 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.ResNet152", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.ResNet152"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.ResNet152", inputs=[])
    KERAS_LAYERS_MAPPING["applications.ResNet152"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications__res_net50_v2() -> None:
    """Tests the keras layers applications.ResNet50V2 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.ResNet50V2", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.ResNet50V2"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.ResNet50V2", inputs=[])
    KERAS_LAYERS_MAPPING["applications.ResNet50V2"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications__res_net101_v2() -> None:
    """Tests the keras layers applications.ResNet101V2 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.ResNet101V2", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.ResNet101V2"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.ResNet101V2", inputs=[])
    KERAS_LAYERS_MAPPING["applications.ResNet101V2"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications__res_net152_v2() -> None:
    """Tests the keras layers applications.ResNet152V2 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.ResNet152V2", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.ResNet152V2"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.ResNet152V2", inputs=[])
    KERAS_LAYERS_MAPPING["applications.ResNet152V2"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_vgg16() -> None:
    """Tests the keras layers applications.VGG16 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.VGG16", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.VGG16"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.VGG16", inputs=[])
    KERAS_LAYERS_MAPPING["applications.VGG16"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_vgg19() -> None:
    """Tests the keras layers applications.VGG19 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.VGG19", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.VGG19"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.VGG19", inputs=[])
    KERAS_LAYERS_MAPPING["applications.VGG19"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications__xception() -> None:
    """Tests the keras layers applications.Xception functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.Xception", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.Xception"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.Xception", inputs=[])
    KERAS_LAYERS_MAPPING["applications.Xception"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_mobilenet_v3_decode_predictions() -> None:
    """Tests the keras layers applications.mobilenet_v3.decode_predictions functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.mobilenet_v3.decode_predictions", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.mobilenet_v3.decode_predictions"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.mobilenet_v3.decode_predictions", inputs=[])
    KERAS_LAYERS_MAPPING["applications.mobilenet_v3.decode_predictions"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_mobilenet_v3_preprocess_input() -> None:
    """Tests the keras layers applications.mobilenet_v3.preprocess_input functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.mobilenet_v3.preprocess_input", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.mobilenet_v3.preprocess_input"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.mobilenet_v3.preprocess_input", inputs=[])
    KERAS_LAYERS_MAPPING["applications.mobilenet_v3.preprocess_input"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_inception_resnet_v2__inception_res_net_v2() -> None:
    """Tests the keras layers applications.inception_resnet_v2.InceptionResNetV2 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.inception_resnet_v2.InceptionResNetV2", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.inception_resnet_v2.InceptionResNetV2"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.inception_resnet_v2.InceptionResNetV2", inputs=[])
    KERAS_LAYERS_MAPPING["applications.inception_resnet_v2.InceptionResNetV2"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_inception_resnet_v2_decode_predictions() -> None:
    """Tests the keras layers applications.inception_resnet_v2.decode_predictions functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.inception_resnet_v2.decode_predictions", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.inception_resnet_v2.decode_predictions"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.inception_resnet_v2.decode_predictions", inputs=[])
    KERAS_LAYERS_MAPPING["applications.inception_resnet_v2.decode_predictions"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_inception_resnet_v2_preprocess_input() -> None:
    """Tests the keras layers applications.inception_resnet_v2.preprocess_input functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.inception_resnet_v2.preprocess_input", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.inception_resnet_v2.preprocess_input"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.inception_resnet_v2.preprocess_input", inputs=[])
    KERAS_LAYERS_MAPPING["applications.inception_resnet_v2.preprocess_input"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_resnet_v2__res_net50_v2() -> None:
    """Tests the keras layers applications.resnet_v2.ResNet50V2 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.resnet_v2.ResNet50V2", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.resnet_v2.ResNet50V2"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.resnet_v2.ResNet50V2", inputs=[])
    KERAS_LAYERS_MAPPING["applications.resnet_v2.ResNet50V2"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_resnet_v2__res_net101_v2() -> None:
    """Tests the keras layers applications.resnet_v2.ResNet101V2 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.resnet_v2.ResNet101V2", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.resnet_v2.ResNet101V2"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.resnet_v2.ResNet101V2", inputs=[])
    KERAS_LAYERS_MAPPING["applications.resnet_v2.ResNet101V2"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_resnet_v2__res_net152_v2() -> None:
    """Tests the keras layers applications.resnet_v2.ResNet152V2 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.resnet_v2.ResNet152V2", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.resnet_v2.ResNet152V2"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.resnet_v2.ResNet152V2", inputs=[])
    KERAS_LAYERS_MAPPING["applications.resnet_v2.ResNet152V2"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_resnet_v2_decode_predictions() -> None:
    """Tests the keras layers applications.resnet_v2.decode_predictions functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.resnet_v2.decode_predictions", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.resnet_v2.decode_predictions"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.resnet_v2.decode_predictions", inputs=[])
    KERAS_LAYERS_MAPPING["applications.resnet_v2.decode_predictions"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_resnet_v2_preprocess_input() -> None:
    """Tests the keras layers applications.resnet_v2.preprocess_input functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.resnet_v2.preprocess_input", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.resnet_v2.preprocess_input"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.resnet_v2.preprocess_input", inputs=[])
    KERAS_LAYERS_MAPPING["applications.resnet_v2.preprocess_input"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_convnext__conv_ne_xt_base() -> None:
    """Tests the keras layers applications.convnext.ConvNeXtBase functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.convnext.ConvNeXtBase", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.convnext.ConvNeXtBase"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.convnext.ConvNeXtBase", inputs=[])
    KERAS_LAYERS_MAPPING["applications.convnext.ConvNeXtBase"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_convnext__conv_ne_xt_large() -> None:
    """Tests the keras layers applications.convnext.ConvNeXtLarge functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.convnext.ConvNeXtLarge", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.convnext.ConvNeXtLarge"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.convnext.ConvNeXtLarge", inputs=[])
    KERAS_LAYERS_MAPPING["applications.convnext.ConvNeXtLarge"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_convnext__conv_ne_xt_small() -> None:
    """Tests the keras layers applications.convnext.ConvNeXtSmall functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.convnext.ConvNeXtSmall", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.convnext.ConvNeXtSmall"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.convnext.ConvNeXtSmall", inputs=[])
    KERAS_LAYERS_MAPPING["applications.convnext.ConvNeXtSmall"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_convnext__conv_ne_xt_tiny() -> None:
    """Tests the keras layers applications.convnext.ConvNeXtTiny functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.convnext.ConvNeXtTiny", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.convnext.ConvNeXtTiny"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.convnext.ConvNeXtTiny", inputs=[])
    KERAS_LAYERS_MAPPING["applications.convnext.ConvNeXtTiny"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_convnext__conv_ne_xt_x_large() -> None:
    """Tests the keras layers applications.convnext.ConvNeXtXLarge functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.convnext.ConvNeXtXLarge", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.convnext.ConvNeXtXLarge"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.convnext.ConvNeXtXLarge", inputs=[])
    KERAS_LAYERS_MAPPING["applications.convnext.ConvNeXtXLarge"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_convnext_decode_predictions() -> None:
    """Tests the keras layers applications.convnext.decode_predictions functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.convnext.decode_predictions", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.convnext.decode_predictions"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.convnext.decode_predictions", inputs=[])
    KERAS_LAYERS_MAPPING["applications.convnext.decode_predictions"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_convnext_preprocess_input() -> None:
    """Tests the keras layers applications.convnext.preprocess_input functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.convnext.preprocess_input", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.convnext.preprocess_input"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.convnext.preprocess_input", inputs=[])
    KERAS_LAYERS_MAPPING["applications.convnext.preprocess_input"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_inception_v3__inception_v3() -> None:
    """Tests the keras layers applications.inception_v3.InceptionV3 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.inception_v3.InceptionV3", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.inception_v3.InceptionV3"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.inception_v3.InceptionV3", inputs=[])
    KERAS_LAYERS_MAPPING["applications.inception_v3.InceptionV3"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_inception_v3_decode_predictions() -> None:
    """Tests the keras layers applications.inception_v3.decode_predictions functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.inception_v3.decode_predictions", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.inception_v3.decode_predictions"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.inception_v3.decode_predictions", inputs=[])
    KERAS_LAYERS_MAPPING["applications.inception_v3.decode_predictions"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_inception_v3_preprocess_input() -> None:
    """Tests the keras layers applications.inception_v3.preprocess_input functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.inception_v3.preprocess_input", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.inception_v3.preprocess_input"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.inception_v3.preprocess_input", inputs=[])
    KERAS_LAYERS_MAPPING["applications.inception_v3.preprocess_input"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_xception__xception() -> None:
    """Tests the keras layers applications.xception.Xception functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.xception.Xception", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.xception.Xception"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.xception.Xception", inputs=[])
    KERAS_LAYERS_MAPPING["applications.xception.Xception"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_xception_decode_predictions() -> None:
    """Tests the keras layers applications.xception.decode_predictions functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.xception.decode_predictions", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.xception.decode_predictions"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.xception.decode_predictions", inputs=[])
    KERAS_LAYERS_MAPPING["applications.xception.decode_predictions"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_xception_preprocess_input() -> None:
    """Tests the keras layers applications.xception.preprocess_input functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.xception.preprocess_input", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.xception.preprocess_input"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.xception.preprocess_input", inputs=[])
    KERAS_LAYERS_MAPPING["applications.xception.preprocess_input"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_resnet50__res_net50() -> None:
    """Tests the keras layers applications.resnet50.ResNet50 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.resnet50.ResNet50", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.resnet50.ResNet50"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.resnet50.ResNet50", inputs=[])
    KERAS_LAYERS_MAPPING["applications.resnet50.ResNet50"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_resnet50_decode_predictions() -> None:
    """Tests the keras layers applications.resnet50.decode_predictions functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.resnet50.decode_predictions", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.resnet50.decode_predictions"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.resnet50.decode_predictions", inputs=[])
    KERAS_LAYERS_MAPPING["applications.resnet50.decode_predictions"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_resnet50_preprocess_input() -> None:
    """Tests the keras layers applications.resnet50.preprocess_input functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.resnet50.preprocess_input", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.resnet50.preprocess_input"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.resnet50.preprocess_input", inputs=[])
    KERAS_LAYERS_MAPPING["applications.resnet50.preprocess_input"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_vgg19_vgg19() -> None:
    """Tests the keras layers applications.vgg19.VGG19 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.vgg19.VGG19", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.vgg19.VGG19"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.vgg19.VGG19", inputs=[])
    KERAS_LAYERS_MAPPING["applications.vgg19.VGG19"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_vgg19_decode_predictions() -> None:
    """Tests the keras layers applications.vgg19.decode_predictions functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.vgg19.decode_predictions", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.vgg19.decode_predictions"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.vgg19.decode_predictions", inputs=[])
    KERAS_LAYERS_MAPPING["applications.vgg19.decode_predictions"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_vgg19_preprocess_input() -> None:
    """Tests the keras layers applications.vgg19.preprocess_input functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.vgg19.preprocess_input", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.vgg19.preprocess_input"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.vgg19.preprocess_input", inputs=[])
    KERAS_LAYERS_MAPPING["applications.vgg19.preprocess_input"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_resnet__res_net50() -> None:
    """Tests the keras layers applications.resnet.ResNet50 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.resnet.ResNet50", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.resnet.ResNet50"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.resnet.ResNet50", inputs=[])
    KERAS_LAYERS_MAPPING["applications.resnet.ResNet50"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_resnet__res_net101() -> None:
    """Tests the keras layers applications.resnet.ResNet101 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.resnet.ResNet101", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.resnet.ResNet101"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.resnet.ResNet101", inputs=[])
    KERAS_LAYERS_MAPPING["applications.resnet.ResNet101"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_resnet__res_net152() -> None:
    """Tests the keras layers applications.resnet.ResNet152 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.resnet.ResNet152", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.resnet.ResNet152"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.resnet.ResNet152", inputs=[])
    KERAS_LAYERS_MAPPING["applications.resnet.ResNet152"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_resnet_decode_predictions() -> None:
    """Tests the keras layers applications.resnet.decode_predictions functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.resnet.decode_predictions", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.resnet.decode_predictions"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.resnet.decode_predictions", inputs=[])
    KERAS_LAYERS_MAPPING["applications.resnet.decode_predictions"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_resnet_preprocess_input() -> None:
    """Tests the keras layers applications.resnet.preprocess_input functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.resnet.preprocess_input", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.resnet.preprocess_input"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.resnet.preprocess_input", inputs=[])
    KERAS_LAYERS_MAPPING["applications.resnet.preprocess_input"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_vgg16_vgg16() -> None:
    """Tests the keras layers applications.vgg16.VGG16 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.vgg16.VGG16", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.vgg16.VGG16"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.vgg16.VGG16", inputs=[])
    KERAS_LAYERS_MAPPING["applications.vgg16.VGG16"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_vgg16_decode_predictions() -> None:
    """Tests the keras layers applications.vgg16.decode_predictions functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.vgg16.decode_predictions", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.vgg16.decode_predictions"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.vgg16.decode_predictions", inputs=[])
    KERAS_LAYERS_MAPPING["applications.vgg16.decode_predictions"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_vgg16_preprocess_input() -> None:
    """Tests the keras layers applications.vgg16.preprocess_input functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.vgg16.preprocess_input", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.vgg16.preprocess_input"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.vgg16.preprocess_input", inputs=[])
    KERAS_LAYERS_MAPPING["applications.vgg16.preprocess_input"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_densenet__dense_net121() -> None:
    """Tests the keras layers applications.densenet.DenseNet121 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.densenet.DenseNet121", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.densenet.DenseNet121"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.densenet.DenseNet121", inputs=[])
    KERAS_LAYERS_MAPPING["applications.densenet.DenseNet121"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_densenet__dense_net169() -> None:
    """Tests the keras layers applications.densenet.DenseNet169 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.densenet.DenseNet169", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.densenet.DenseNet169"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.densenet.DenseNet169", inputs=[])
    KERAS_LAYERS_MAPPING["applications.densenet.DenseNet169"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_densenet__dense_net201() -> None:
    """Tests the keras layers applications.densenet.DenseNet201 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.densenet.DenseNet201", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.densenet.DenseNet201"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.densenet.DenseNet201", inputs=[])
    KERAS_LAYERS_MAPPING["applications.densenet.DenseNet201"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_densenet_decode_predictions() -> None:
    """Tests the keras layers applications.densenet.decode_predictions functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.densenet.decode_predictions", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.densenet.decode_predictions"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.densenet.decode_predictions", inputs=[])
    KERAS_LAYERS_MAPPING["applications.densenet.decode_predictions"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_densenet_preprocess_input() -> None:
    """Tests the keras layers applications.densenet.preprocess_input functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.densenet.preprocess_input", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.densenet.preprocess_input"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.densenet.preprocess_input", inputs=[])
    KERAS_LAYERS_MAPPING["applications.densenet.preprocess_input"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_nasnet_nas_net_large() -> None:
    """Tests the keras layers applications.nasnet.NASNetLarge functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.nasnet.NASNetLarge", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.nasnet.NASNetLarge"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.nasnet.NASNetLarge", inputs=[])
    KERAS_LAYERS_MAPPING["applications.nasnet.NASNetLarge"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_nasnet_nas_net_mobile() -> None:
    """Tests the keras layers applications.nasnet.NASNetMobile functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.nasnet.NASNetMobile", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.nasnet.NASNetMobile"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.nasnet.NASNetMobile", inputs=[])
    KERAS_LAYERS_MAPPING["applications.nasnet.NASNetMobile"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_nasnet_decode_predictions() -> None:
    """Tests the keras layers applications.nasnet.decode_predictions functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.nasnet.decode_predictions", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.nasnet.decode_predictions"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.nasnet.decode_predictions", inputs=[])
    KERAS_LAYERS_MAPPING["applications.nasnet.decode_predictions"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_nasnet_preprocess_input() -> None:
    """Tests the keras layers applications.nasnet.preprocess_input functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.nasnet.preprocess_input", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.nasnet.preprocess_input"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.nasnet.preprocess_input", inputs=[])
    KERAS_LAYERS_MAPPING["applications.nasnet.preprocess_input"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_mobilenet__mobile_net() -> None:
    """Tests the keras layers applications.mobilenet.MobileNet functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.mobilenet.MobileNet", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.mobilenet.MobileNet"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.mobilenet.MobileNet", inputs=[])
    KERAS_LAYERS_MAPPING["applications.mobilenet.MobileNet"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_mobilenet_decode_predictions() -> None:
    """Tests the keras layers applications.mobilenet.decode_predictions functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.mobilenet.decode_predictions", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.mobilenet.decode_predictions"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.mobilenet.decode_predictions", inputs=[])
    KERAS_LAYERS_MAPPING["applications.mobilenet.decode_predictions"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_mobilenet_preprocess_input() -> None:
    """Tests the keras layers applications.mobilenet.preprocess_input functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.mobilenet.preprocess_input", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.mobilenet.preprocess_input"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.mobilenet.preprocess_input", inputs=[])
    KERAS_LAYERS_MAPPING["applications.mobilenet.preprocess_input"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_imagenet_utils_decode_predictions() -> None:
    """Tests the keras layers applications.imagenet_utils.decode_predictions functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.imagenet_utils.decode_predictions", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.imagenet_utils.decode_predictions"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.imagenet_utils.decode_predictions", inputs=[])
    KERAS_LAYERS_MAPPING["applications.imagenet_utils.decode_predictions"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_imagenet_utils_preprocess_input() -> None:
    """Tests the keras layers applications.imagenet_utils.preprocess_input functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.imagenet_utils.preprocess_input", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.imagenet_utils.preprocess_input"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.imagenet_utils.preprocess_input", inputs=[])
    KERAS_LAYERS_MAPPING["applications.imagenet_utils.preprocess_input"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_efficientnet__efficient_net_b0() -> None:
    """Tests the keras layers applications.efficientnet.EfficientNetB0 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.efficientnet.EfficientNetB0", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.efficientnet.EfficientNetB0"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.efficientnet.EfficientNetB0", inputs=[])
    KERAS_LAYERS_MAPPING["applications.efficientnet.EfficientNetB0"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_efficientnet__efficient_net_b1() -> None:
    """Tests the keras layers applications.efficientnet.EfficientNetB1 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.efficientnet.EfficientNetB1", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.efficientnet.EfficientNetB1"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.efficientnet.EfficientNetB1", inputs=[])
    KERAS_LAYERS_MAPPING["applications.efficientnet.EfficientNetB1"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_efficientnet__efficient_net_b2() -> None:
    """Tests the keras layers applications.efficientnet.EfficientNetB2 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.efficientnet.EfficientNetB2", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.efficientnet.EfficientNetB2"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.efficientnet.EfficientNetB2", inputs=[])
    KERAS_LAYERS_MAPPING["applications.efficientnet.EfficientNetB2"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_efficientnet__efficient_net_b3() -> None:
    """Tests the keras layers applications.efficientnet.EfficientNetB3 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.efficientnet.EfficientNetB3", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.efficientnet.EfficientNetB3"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.efficientnet.EfficientNetB3", inputs=[])
    KERAS_LAYERS_MAPPING["applications.efficientnet.EfficientNetB3"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_efficientnet__efficient_net_b4() -> None:
    """Tests the keras layers applications.efficientnet.EfficientNetB4 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.efficientnet.EfficientNetB4", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.efficientnet.EfficientNetB4"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.efficientnet.EfficientNetB4", inputs=[])
    KERAS_LAYERS_MAPPING["applications.efficientnet.EfficientNetB4"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_efficientnet__efficient_net_b5() -> None:
    """Tests the keras layers applications.efficientnet.EfficientNetB5 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.efficientnet.EfficientNetB5", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.efficientnet.EfficientNetB5"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.efficientnet.EfficientNetB5", inputs=[])
    KERAS_LAYERS_MAPPING["applications.efficientnet.EfficientNetB5"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_efficientnet__efficient_net_b6() -> None:
    """Tests the keras layers applications.efficientnet.EfficientNetB6 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.efficientnet.EfficientNetB6", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.efficientnet.EfficientNetB6"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.efficientnet.EfficientNetB6", inputs=[])
    KERAS_LAYERS_MAPPING["applications.efficientnet.EfficientNetB6"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_efficientnet__efficient_net_b7() -> None:
    """Tests the keras layers applications.efficientnet.EfficientNetB7 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.efficientnet.EfficientNetB7", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.efficientnet.EfficientNetB7"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.efficientnet.EfficientNetB7", inputs=[])
    KERAS_LAYERS_MAPPING["applications.efficientnet.EfficientNetB7"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_efficientnet_decode_predictions() -> None:
    """Tests the keras layers applications.efficientnet.decode_predictions functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.efficientnet.decode_predictions", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.efficientnet.decode_predictions"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.efficientnet.decode_predictions", inputs=[])
    KERAS_LAYERS_MAPPING["applications.efficientnet.decode_predictions"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_efficientnet_preprocess_input() -> None:
    """Tests the keras layers applications.efficientnet.preprocess_input functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.efficientnet.preprocess_input", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.efficientnet.preprocess_input"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.efficientnet.preprocess_input", inputs=[])
    KERAS_LAYERS_MAPPING["applications.efficientnet.preprocess_input"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_efficientnet_v2__efficient_net_v2_b0() -> None:
    """Tests the keras layers applications.efficientnet_v2.EfficientNetV2B0 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.efficientnet_v2.EfficientNetV2B0", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.efficientnet_v2.EfficientNetV2B0"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.efficientnet_v2.EfficientNetV2B0", inputs=[])
    KERAS_LAYERS_MAPPING["applications.efficientnet_v2.EfficientNetV2B0"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_efficientnet_v2__efficient_net_v2_b1() -> None:
    """Tests the keras layers applications.efficientnet_v2.EfficientNetV2B1 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.efficientnet_v2.EfficientNetV2B1", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.efficientnet_v2.EfficientNetV2B1"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.efficientnet_v2.EfficientNetV2B1", inputs=[])
    KERAS_LAYERS_MAPPING["applications.efficientnet_v2.EfficientNetV2B1"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_efficientnet_v2__efficient_net_v2_b2() -> None:
    """Tests the keras layers applications.efficientnet_v2.EfficientNetV2B2 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.efficientnet_v2.EfficientNetV2B2", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.efficientnet_v2.EfficientNetV2B2"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.efficientnet_v2.EfficientNetV2B2", inputs=[])
    KERAS_LAYERS_MAPPING["applications.efficientnet_v2.EfficientNetV2B2"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_efficientnet_v2__efficient_net_v2_b3() -> None:
    """Tests the keras layers applications.efficientnet_v2.EfficientNetV2B3 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.efficientnet_v2.EfficientNetV2B3", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.efficientnet_v2.EfficientNetV2B3"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.efficientnet_v2.EfficientNetV2B3", inputs=[])
    KERAS_LAYERS_MAPPING["applications.efficientnet_v2.EfficientNetV2B3"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_efficientnet_v2__efficient_net_v2_l() -> None:
    """Tests the keras layers applications.efficientnet_v2.EfficientNetV2L functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.efficientnet_v2.EfficientNetV2L", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.efficientnet_v2.EfficientNetV2L"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.efficientnet_v2.EfficientNetV2L", inputs=[])
    KERAS_LAYERS_MAPPING["applications.efficientnet_v2.EfficientNetV2L"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_efficientnet_v2__efficient_net_v2_m() -> None:
    """Tests the keras layers applications.efficientnet_v2.EfficientNetV2M functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.efficientnet_v2.EfficientNetV2M", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.efficientnet_v2.EfficientNetV2M"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.efficientnet_v2.EfficientNetV2M", inputs=[])
    KERAS_LAYERS_MAPPING["applications.efficientnet_v2.EfficientNetV2M"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_efficientnet_v2__efficient_net_v2_s() -> None:
    """Tests the keras layers applications.efficientnet_v2.EfficientNetV2S functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.efficientnet_v2.EfficientNetV2S", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.efficientnet_v2.EfficientNetV2S"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.efficientnet_v2.EfficientNetV2S", inputs=[])
    KERAS_LAYERS_MAPPING["applications.efficientnet_v2.EfficientNetV2S"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_efficientnet_v2_decode_predictions() -> None:
    """Tests the keras layers applications.efficientnet_v2.decode_predictions functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.efficientnet_v2.decode_predictions", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.efficientnet_v2.decode_predictions"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.efficientnet_v2.decode_predictions", inputs=[])
    KERAS_LAYERS_MAPPING["applications.efficientnet_v2.decode_predictions"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_efficientnet_v2_preprocess_input() -> None:
    """Tests the keras layers applications.efficientnet_v2.preprocess_input functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.efficientnet_v2.preprocess_input", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.efficientnet_v2.preprocess_input"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.efficientnet_v2.preprocess_input", inputs=[])
    KERAS_LAYERS_MAPPING["applications.efficientnet_v2.preprocess_input"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_mobilenet_v2__mobile_net_v2() -> None:
    """Tests the keras layers applications.mobilenet_v2.MobileNetV2 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.mobilenet_v2.MobileNetV2", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.mobilenet_v2.MobileNetV2"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.mobilenet_v2.MobileNetV2", inputs=[])
    KERAS_LAYERS_MAPPING["applications.mobilenet_v2.MobileNetV2"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_mobilenet_v2_decode_predictions() -> None:
    """Tests the keras layers applications.mobilenet_v2.decode_predictions functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.mobilenet_v2.decode_predictions", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.mobilenet_v2.decode_predictions"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.mobilenet_v2.decode_predictions", inputs=[])
    KERAS_LAYERS_MAPPING["applications.mobilenet_v2.decode_predictions"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_applications_mobilenet_v2_preprocess_input() -> None:
    """Tests the keras layers applications.mobilenet_v2.preprocess_input functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "applications.mobilenet_v2.preprocess_input", inputs=["a"])

    KERAS_LAYERS_MAPPING["applications.mobilenet_v2.preprocess_input"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "applications.mobilenet_v2.preprocess_input", inputs=[])
    KERAS_LAYERS_MAPPING["applications.mobilenet_v2.preprocess_input"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_initializers_deserialize() -> None:
    """Tests the keras layers initializers.deserialize functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "initializers.deserialize", inputs=["a"])

    KERAS_LAYERS_MAPPING["initializers.deserialize"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "initializers.deserialize", inputs=[])
    KERAS_LAYERS_MAPPING["initializers.deserialize"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_initializers_get() -> None:
    """Tests the keras layers initializers.get functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "initializers.get", inputs=["a"])

    KERAS_LAYERS_MAPPING["initializers.get"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "initializers.get", inputs=[])
    KERAS_LAYERS_MAPPING["initializers.get"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_initializers_serialize() -> None:
    """Tests the keras layers initializers.serialize functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "initializers.serialize", inputs=["a"])

    KERAS_LAYERS_MAPPING["initializers.serialize"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "initializers.serialize", inputs=[])
    KERAS_LAYERS_MAPPING["initializers.serialize"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_initializers_stft() -> None:
    """Tests the keras layers initializers.STFT functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "initializers.STFT", inputs=["a"])

    KERAS_LAYERS_MAPPING["initializers.STFT"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "initializers.STFT", inputs=[])
    KERAS_LAYERS_MAPPING["initializers.STFT"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_initializers_stft_initializer() -> None:
    """Tests the keras layers initializers.STFTInitializer functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "initializers.STFTInitializer", inputs=["a"])

    KERAS_LAYERS_MAPPING["initializers.STFTInitializer"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "initializers.STFTInitializer", inputs=[])
    KERAS_LAYERS_MAPPING["initializers.STFTInitializer"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_initializers_stft_ext() -> None:
    """Tests the keras layers initializers.stft functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "initializers.stft", inputs=["a"])

    KERAS_LAYERS_MAPPING["initializers.stft"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "initializers.stft", inputs=[])
    KERAS_LAYERS_MAPPING["initializers.stft"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_initializers__constant() -> None:
    """Tests the keras layers initializers.Constant functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "initializers.Constant", inputs=["a"])

    KERAS_LAYERS_MAPPING["initializers.Constant"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "initializers.Constant", inputs=[])
    KERAS_LAYERS_MAPPING["initializers.Constant"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_initializers_constant() -> None:
    """Tests the keras layers initializers.constant functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "initializers.constant", inputs=["a"])

    KERAS_LAYERS_MAPPING["initializers.constant"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "initializers.constant", inputs=[])
    KERAS_LAYERS_MAPPING["initializers.constant"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_initializers__identity() -> None:
    """Tests the keras layers initializers.Identity functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "initializers.Identity", inputs=["a"])

    KERAS_LAYERS_MAPPING["initializers.Identity"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "initializers.Identity", inputs=[])
    KERAS_LAYERS_MAPPING["initializers.Identity"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_initializers__identity_initializer() -> None:
    """Tests the keras layers initializers.IdentityInitializer functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "initializers.IdentityInitializer", inputs=["a"])

    KERAS_LAYERS_MAPPING["initializers.IdentityInitializer"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "initializers.IdentityInitializer", inputs=[])
    KERAS_LAYERS_MAPPING["initializers.IdentityInitializer"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_initializers_identity() -> None:
    """Tests the keras layers initializers.identity functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "initializers.identity", inputs=["a"])

    KERAS_LAYERS_MAPPING["initializers.identity"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "initializers.identity", inputs=[])
    KERAS_LAYERS_MAPPING["initializers.identity"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_initializers__ones() -> None:
    """Tests the keras layers initializers.Ones functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "initializers.Ones", inputs=["a"])

    KERAS_LAYERS_MAPPING["initializers.Ones"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "initializers.Ones", inputs=[])
    KERAS_LAYERS_MAPPING["initializers.Ones"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_initializers_ones() -> None:
    """Tests the keras layers initializers.ones functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "initializers.ones", inputs=["a"])

    KERAS_LAYERS_MAPPING["initializers.ones"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "initializers.ones", inputs=[])
    KERAS_LAYERS_MAPPING["initializers.ones"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_initializers__zeros() -> None:
    """Tests the keras layers initializers.Zeros functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "initializers.Zeros", inputs=["a"])

    KERAS_LAYERS_MAPPING["initializers.Zeros"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "initializers.Zeros", inputs=[])
    KERAS_LAYERS_MAPPING["initializers.Zeros"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_initializers_zeros() -> None:
    """Tests the keras layers initializers.zeros functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "initializers.zeros", inputs=["a"])

    KERAS_LAYERS_MAPPING["initializers.zeros"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "initializers.zeros", inputs=[])
    KERAS_LAYERS_MAPPING["initializers.zeros"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_initializers__initializer() -> None:
    """Tests the keras layers initializers.Initializer functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "initializers.Initializer", inputs=["a"])

    KERAS_LAYERS_MAPPING["initializers.Initializer"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "initializers.Initializer", inputs=[])
    KERAS_LAYERS_MAPPING["initializers.Initializer"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_initializers__glorot_normal() -> None:
    """Tests the keras layers initializers.GlorotNormal functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "initializers.GlorotNormal", inputs=["a"])

    KERAS_LAYERS_MAPPING["initializers.GlorotNormal"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "initializers.GlorotNormal", inputs=[])
    KERAS_LAYERS_MAPPING["initializers.GlorotNormal"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_initializers_glorot_normal() -> None:
    """Tests the keras layers initializers.glorot_normal functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "initializers.glorot_normal", inputs=["a"])

    KERAS_LAYERS_MAPPING["initializers.glorot_normal"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "initializers.glorot_normal", inputs=[])
    KERAS_LAYERS_MAPPING["initializers.glorot_normal"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_initializers__glorot_uniform() -> None:
    """Tests the keras layers initializers.GlorotUniform functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "initializers.GlorotUniform", inputs=["a"])

    KERAS_LAYERS_MAPPING["initializers.GlorotUniform"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "initializers.GlorotUniform", inputs=[])
    KERAS_LAYERS_MAPPING["initializers.GlorotUniform"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_initializers_glorot_uniform() -> None:
    """Tests the keras layers initializers.glorot_uniform functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "initializers.glorot_uniform", inputs=["a"])

    KERAS_LAYERS_MAPPING["initializers.glorot_uniform"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "initializers.glorot_uniform", inputs=[])
    KERAS_LAYERS_MAPPING["initializers.glorot_uniform"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_initializers__he_normal() -> None:
    """Tests the keras layers initializers.HeNormal functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "initializers.HeNormal", inputs=["a"])

    KERAS_LAYERS_MAPPING["initializers.HeNormal"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "initializers.HeNormal", inputs=[])
    KERAS_LAYERS_MAPPING["initializers.HeNormal"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_initializers_he_normal() -> None:
    """Tests the keras layers initializers.he_normal functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "initializers.he_normal", inputs=["a"])

    KERAS_LAYERS_MAPPING["initializers.he_normal"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "initializers.he_normal", inputs=[])
    KERAS_LAYERS_MAPPING["initializers.he_normal"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_initializers__he_uniform() -> None:
    """Tests the keras layers initializers.HeUniform functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "initializers.HeUniform", inputs=["a"])

    KERAS_LAYERS_MAPPING["initializers.HeUniform"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "initializers.HeUniform", inputs=[])
    KERAS_LAYERS_MAPPING["initializers.HeUniform"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_initializers_he_uniform() -> None:
    """Tests the keras layers initializers.he_uniform functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "initializers.he_uniform", inputs=["a"])

    KERAS_LAYERS_MAPPING["initializers.he_uniform"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "initializers.he_uniform", inputs=[])
    KERAS_LAYERS_MAPPING["initializers.he_uniform"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_initializers__lecun_normal() -> None:
    """Tests the keras layers initializers.LecunNormal functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "initializers.LecunNormal", inputs=["a"])

    KERAS_LAYERS_MAPPING["initializers.LecunNormal"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "initializers.LecunNormal", inputs=[])
    KERAS_LAYERS_MAPPING["initializers.LecunNormal"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_initializers_lecun_normal() -> None:
    """Tests the keras layers initializers.lecun_normal functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "initializers.lecun_normal", inputs=["a"])

    KERAS_LAYERS_MAPPING["initializers.lecun_normal"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "initializers.lecun_normal", inputs=[])
    KERAS_LAYERS_MAPPING["initializers.lecun_normal"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_initializers__lecun_uniform() -> None:
    """Tests the keras layers initializers.LecunUniform functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "initializers.LecunUniform", inputs=["a"])

    KERAS_LAYERS_MAPPING["initializers.LecunUniform"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "initializers.LecunUniform", inputs=[])
    KERAS_LAYERS_MAPPING["initializers.LecunUniform"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_initializers_lecun_uniform() -> None:
    """Tests the keras layers initializers.lecun_uniform functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "initializers.lecun_uniform", inputs=["a"])

    KERAS_LAYERS_MAPPING["initializers.lecun_uniform"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "initializers.lecun_uniform", inputs=[])
    KERAS_LAYERS_MAPPING["initializers.lecun_uniform"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_initializers__orthogonal() -> None:
    """Tests the keras layers initializers.Orthogonal functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "initializers.Orthogonal", inputs=["a"])

    KERAS_LAYERS_MAPPING["initializers.Orthogonal"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "initializers.Orthogonal", inputs=[])
    KERAS_LAYERS_MAPPING["initializers.Orthogonal"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_initializers__orthogonal_initializer() -> None:
    """Tests the keras layers initializers.OrthogonalInitializer functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "initializers.OrthogonalInitializer", inputs=["a"])

    KERAS_LAYERS_MAPPING["initializers.OrthogonalInitializer"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "initializers.OrthogonalInitializer", inputs=[])
    KERAS_LAYERS_MAPPING["initializers.OrthogonalInitializer"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_initializers_orthogonal() -> None:
    """Tests the keras layers initializers.orthogonal functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "initializers.orthogonal", inputs=["a"])

    KERAS_LAYERS_MAPPING["initializers.orthogonal"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "initializers.orthogonal", inputs=[])
    KERAS_LAYERS_MAPPING["initializers.orthogonal"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_initializers__random_normal() -> None:
    """Tests the keras layers initializers.RandomNormal functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "initializers.RandomNormal", inputs=["a"])

    KERAS_LAYERS_MAPPING["initializers.RandomNormal"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "initializers.RandomNormal", inputs=[])
    KERAS_LAYERS_MAPPING["initializers.RandomNormal"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_initializers_random_normal() -> None:
    """Tests the keras layers initializers.random_normal functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "initializers.random_normal", inputs=["a"])

    KERAS_LAYERS_MAPPING["initializers.random_normal"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "initializers.random_normal", inputs=[])
    KERAS_LAYERS_MAPPING["initializers.random_normal"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_initializers__random_uniform() -> None:
    """Tests the keras layers initializers.RandomUniform functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "initializers.RandomUniform", inputs=["a"])

    KERAS_LAYERS_MAPPING["initializers.RandomUniform"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "initializers.RandomUniform", inputs=[])
    KERAS_LAYERS_MAPPING["initializers.RandomUniform"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_initializers_random_uniform() -> None:
    """Tests the keras layers initializers.random_uniform functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "initializers.random_uniform", inputs=["a"])

    KERAS_LAYERS_MAPPING["initializers.random_uniform"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "initializers.random_uniform", inputs=[])
    KERAS_LAYERS_MAPPING["initializers.random_uniform"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_initializers__truncated_normal() -> None:
    """Tests the keras layers initializers.TruncatedNormal functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "initializers.TruncatedNormal", inputs=["a"])

    KERAS_LAYERS_MAPPING["initializers.TruncatedNormal"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "initializers.TruncatedNormal", inputs=[])
    KERAS_LAYERS_MAPPING["initializers.TruncatedNormal"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_initializers_truncated_normal() -> None:
    """Tests the keras layers initializers.truncated_normal functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "initializers.truncated_normal", inputs=["a"])

    KERAS_LAYERS_MAPPING["initializers.truncated_normal"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "initializers.truncated_normal", inputs=[])
    KERAS_LAYERS_MAPPING["initializers.truncated_normal"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_initializers__variance_scaling() -> None:
    """Tests the keras layers initializers.VarianceScaling functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "initializers.VarianceScaling", inputs=["a"])

    KERAS_LAYERS_MAPPING["initializers.VarianceScaling"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "initializers.VarianceScaling", inputs=[])
    KERAS_LAYERS_MAPPING["initializers.VarianceScaling"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_initializers_variance_scaling() -> None:
    """Tests the keras layers initializers.variance_scaling functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "initializers.variance_scaling", inputs=["a"])

    KERAS_LAYERS_MAPPING["initializers.variance_scaling"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "initializers.variance_scaling", inputs=[])
    KERAS_LAYERS_MAPPING["initializers.variance_scaling"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_quantizers_deserialize() -> None:
    """Tests the keras layers quantizers.deserialize functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "quantizers.deserialize", inputs=["a"])

    KERAS_LAYERS_MAPPING["quantizers.deserialize"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "quantizers.deserialize", inputs=[])
    KERAS_LAYERS_MAPPING["quantizers.deserialize"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_quantizers_get() -> None:
    """Tests the keras layers quantizers.get functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "quantizers.get", inputs=["a"])

    KERAS_LAYERS_MAPPING["quantizers.get"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "quantizers.get", inputs=[])
    KERAS_LAYERS_MAPPING["quantizers.get"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_quantizers_serialize() -> None:
    """Tests the keras layers quantizers.serialize functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "quantizers.serialize", inputs=["a"])

    KERAS_LAYERS_MAPPING["quantizers.serialize"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "quantizers.serialize", inputs=[])
    KERAS_LAYERS_MAPPING["quantizers.serialize"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_quantizers_gptq_config() -> None:
    """Tests the keras layers quantizers.GPTQConfig functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "quantizers.GPTQConfig", inputs=["a"])

    KERAS_LAYERS_MAPPING["quantizers.GPTQConfig"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "quantizers.GPTQConfig", inputs=[])
    KERAS_LAYERS_MAPPING["quantizers.GPTQConfig"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_quantizers__float8_quantization_config() -> None:
    """Tests the keras layers quantizers.Float8QuantizationConfig functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "quantizers.Float8QuantizationConfig", inputs=["a"])

    KERAS_LAYERS_MAPPING["quantizers.Float8QuantizationConfig"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "quantizers.Float8QuantizationConfig", inputs=[])
    KERAS_LAYERS_MAPPING["quantizers.Float8QuantizationConfig"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_quantizers__int4_quantization_config() -> None:
    """Tests the keras layers quantizers.Int4QuantizationConfig functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "quantizers.Int4QuantizationConfig", inputs=["a"])

    KERAS_LAYERS_MAPPING["quantizers.Int4QuantizationConfig"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "quantizers.Int4QuantizationConfig", inputs=[])
    KERAS_LAYERS_MAPPING["quantizers.Int4QuantizationConfig"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_quantizers__int8_quantization_config() -> None:
    """Tests the keras layers quantizers.Int8QuantizationConfig functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "quantizers.Int8QuantizationConfig", inputs=["a"])

    KERAS_LAYERS_MAPPING["quantizers.Int8QuantizationConfig"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "quantizers.Int8QuantizationConfig", inputs=[])
    KERAS_LAYERS_MAPPING["quantizers.Int8QuantizationConfig"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_quantizers__quantization_config() -> None:
    """Tests the keras layers quantizers.QuantizationConfig functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "quantizers.QuantizationConfig", inputs=["a"])

    KERAS_LAYERS_MAPPING["quantizers.QuantizationConfig"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "quantizers.QuantizationConfig", inputs=[])
    KERAS_LAYERS_MAPPING["quantizers.QuantizationConfig"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_quantizers__abs_max_quantizer() -> None:
    """Tests the keras layers quantizers.AbsMaxQuantizer functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "quantizers.AbsMaxQuantizer", inputs=["a"])

    KERAS_LAYERS_MAPPING["quantizers.AbsMaxQuantizer"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "quantizers.AbsMaxQuantizer", inputs=[])
    KERAS_LAYERS_MAPPING["quantizers.AbsMaxQuantizer"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_quantizers__quantizer() -> None:
    """Tests the keras layers quantizers.Quantizer functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "quantizers.Quantizer", inputs=["a"])

    KERAS_LAYERS_MAPPING["quantizers.Quantizer"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "quantizers.Quantizer", inputs=[])
    KERAS_LAYERS_MAPPING["quantizers.Quantizer"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_quantizers_abs_max_quantize() -> None:
    """Tests the keras layers quantizers.abs_max_quantize functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "quantizers.abs_max_quantize", inputs=["a"])

    KERAS_LAYERS_MAPPING["quantizers.abs_max_quantize"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "quantizers.abs_max_quantize", inputs=[])
    KERAS_LAYERS_MAPPING["quantizers.abs_max_quantize"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_quantizers_compute_float8_amax_history() -> None:
    """Tests the keras layers quantizers.compute_float8_amax_history functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "quantizers.compute_float8_amax_history", inputs=["a"])

    KERAS_LAYERS_MAPPING["quantizers.compute_float8_amax_history"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "quantizers.compute_float8_amax_history", inputs=[])
    KERAS_LAYERS_MAPPING["quantizers.compute_float8_amax_history"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_quantizers_compute_float8_scale() -> None:
    """Tests the keras layers quantizers.compute_float8_scale functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "quantizers.compute_float8_scale", inputs=["a"])

    KERAS_LAYERS_MAPPING["quantizers.compute_float8_scale"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "quantizers.compute_float8_scale", inputs=[])
    KERAS_LAYERS_MAPPING["quantizers.compute_float8_scale"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_quantizers_fake_quant_with_min_max_vars() -> None:
    """Tests the keras layers quantizers.fake_quant_with_min_max_vars functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "quantizers.fake_quant_with_min_max_vars", inputs=["a"])

    KERAS_LAYERS_MAPPING["quantizers.fake_quant_with_min_max_vars"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "quantizers.fake_quant_with_min_max_vars", inputs=[])
    KERAS_LAYERS_MAPPING["quantizers.fake_quant_with_min_max_vars"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_quantizers_pack_int4() -> None:
    """Tests the keras layers quantizers.pack_int4 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "quantizers.pack_int4", inputs=["a"])

    KERAS_LAYERS_MAPPING["quantizers.pack_int4"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "quantizers.pack_int4", inputs=[])
    KERAS_LAYERS_MAPPING["quantizers.pack_int4"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_quantizers_quantize_and_dequantize() -> None:
    """Tests the keras layers quantizers.quantize_and_dequantize functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "quantizers.quantize_and_dequantize", inputs=["a"])

    KERAS_LAYERS_MAPPING["quantizers.quantize_and_dequantize"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "quantizers.quantize_and_dequantize", inputs=[])
    KERAS_LAYERS_MAPPING["quantizers.quantize_and_dequantize"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_quantizers_unpack_int4() -> None:
    """Tests the keras layers quantizers.unpack_int4 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "quantizers.unpack_int4", inputs=["a"])

    KERAS_LAYERS_MAPPING["quantizers.unpack_int4"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "quantizers.unpack_int4", inputs=[])
    KERAS_LAYERS_MAPPING["quantizers.unpack_int4"](builder, node_empty)
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


def test_keras_layers_missing_metrics_binary_crossentropy() -> None:
    """Tests the keras layers metrics.binary_crossentropy functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.binary_crossentropy", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.binary_crossentropy"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.binary_crossentropy", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.binary_crossentropy"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics_binary_focal_crossentropy() -> None:
    """Tests the keras layers metrics.binary_focal_crossentropy functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.binary_focal_crossentropy", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.binary_focal_crossentropy"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.binary_focal_crossentropy", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.binary_focal_crossentropy"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics_categorical_crossentropy() -> None:
    """Tests the keras layers metrics.categorical_crossentropy functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.categorical_crossentropy", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.categorical_crossentropy"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.categorical_crossentropy", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.categorical_crossentropy"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics_categorical_focal_crossentropy() -> None:
    """Tests the keras layers metrics.categorical_focal_crossentropy functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.categorical_focal_crossentropy", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.categorical_focal_crossentropy"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.categorical_focal_crossentropy", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.categorical_focal_crossentropy"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics_categorical_hinge() -> None:
    """Tests the keras layers metrics.categorical_hinge functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.categorical_hinge", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.categorical_hinge"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.categorical_hinge", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.categorical_hinge"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics_hinge() -> None:
    """Tests the keras layers metrics.hinge functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.hinge", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.hinge"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.hinge", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.hinge"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics_huber() -> None:
    """Tests the keras layers metrics.huber functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.huber", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.huber"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.huber", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.huber"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics_kl_divergence() -> None:
    """Tests the keras layers metrics.kl_divergence functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.kl_divergence", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.kl_divergence"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.kl_divergence", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.kl_divergence"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics_log_cosh() -> None:
    """Tests the keras layers metrics.log_cosh functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.log_cosh", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.log_cosh"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.log_cosh", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.log_cosh"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics_mean_absolute_error() -> None:
    """Tests the keras layers metrics.mean_absolute_error functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.mean_absolute_error", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.mean_absolute_error"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.mean_absolute_error", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.mean_absolute_error"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics_mean_absolute_percentage_error() -> None:
    """Tests the keras layers metrics.mean_absolute_percentage_error functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.mean_absolute_percentage_error", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.mean_absolute_percentage_error"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.mean_absolute_percentage_error", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.mean_absolute_percentage_error"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics_mean_squared_error() -> None:
    """Tests the keras layers metrics.mean_squared_error functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.mean_squared_error", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.mean_squared_error"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.mean_squared_error", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.mean_squared_error"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics_mean_squared_logarithmic_error() -> None:
    """Tests the keras layers metrics.mean_squared_logarithmic_error functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.mean_squared_logarithmic_error", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.mean_squared_logarithmic_error"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.mean_squared_logarithmic_error", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.mean_squared_logarithmic_error"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics_poisson() -> None:
    """Tests the keras layers metrics.poisson functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.poisson", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.poisson"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.poisson", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.poisson"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics_sparse_categorical_crossentropy() -> None:
    """Tests the keras layers metrics.sparse_categorical_crossentropy functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.sparse_categorical_crossentropy", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.sparse_categorical_crossentropy"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.sparse_categorical_crossentropy", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.sparse_categorical_crossentropy"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics_squared_hinge() -> None:
    """Tests the keras layers metrics.squared_hinge functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.squared_hinge", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.squared_hinge"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.squared_hinge", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.squared_hinge"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics_deserialize() -> None:
    """Tests the keras layers metrics.deserialize functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.deserialize", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.deserialize"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.deserialize", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.deserialize"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics_get() -> None:
    """Tests the keras layers metrics.get functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.get", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.get"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.get", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.get"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics_serialize() -> None:
    """Tests the keras layers metrics.serialize functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.serialize", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.serialize"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.serialize", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.serialize"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics__accuracy() -> None:
    """Tests the keras layers metrics.Accuracy functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.Accuracy", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.Accuracy"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.Accuracy", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.Accuracy"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics__binary_accuracy() -> None:
    """Tests the keras layers metrics.BinaryAccuracy functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.BinaryAccuracy", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.BinaryAccuracy"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.BinaryAccuracy", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.BinaryAccuracy"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics__categorical_accuracy() -> None:
    """Tests the keras layers metrics.CategoricalAccuracy functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.CategoricalAccuracy", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.CategoricalAccuracy"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.CategoricalAccuracy", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.CategoricalAccuracy"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics__sparse_categorical_accuracy() -> None:
    """Tests the keras layers metrics.SparseCategoricalAccuracy functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.SparseCategoricalAccuracy", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.SparseCategoricalAccuracy"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.SparseCategoricalAccuracy", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.SparseCategoricalAccuracy"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics__sparse_top_k_categorical_accuracy() -> None:
    """Tests the keras layers metrics.SparseTopKCategoricalAccuracy functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.SparseTopKCategoricalAccuracy", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.SparseTopKCategoricalAccuracy"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.SparseTopKCategoricalAccuracy", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.SparseTopKCategoricalAccuracy"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics__top_k_categorical_accuracy() -> None:
    """Tests the keras layers metrics.TopKCategoricalAccuracy functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.TopKCategoricalAccuracy", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.TopKCategoricalAccuracy"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.TopKCategoricalAccuracy", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.TopKCategoricalAccuracy"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics_binary_accuracy() -> None:
    """Tests the keras layers metrics.binary_accuracy functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.binary_accuracy", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.binary_accuracy"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.binary_accuracy", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.binary_accuracy"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics_categorical_accuracy() -> None:
    """Tests the keras layers metrics.categorical_accuracy functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.categorical_accuracy", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.categorical_accuracy"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.categorical_accuracy", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.categorical_accuracy"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics_sparse_categorical_accuracy() -> None:
    """Tests the keras layers metrics.sparse_categorical_accuracy functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.sparse_categorical_accuracy", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.sparse_categorical_accuracy"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.sparse_categorical_accuracy", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.sparse_categorical_accuracy"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics_sparse_top_k_categorical_accuracy() -> None:
    """Tests the keras layers metrics.sparse_top_k_categorical_accuracy functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.sparse_top_k_categorical_accuracy", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.sparse_top_k_categorical_accuracy"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.sparse_top_k_categorical_accuracy", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.sparse_top_k_categorical_accuracy"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics_top_k_categorical_accuracy() -> None:
    """Tests the keras layers metrics.top_k_categorical_accuracy functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.top_k_categorical_accuracy", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.top_k_categorical_accuracy"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.top_k_categorical_accuracy", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.top_k_categorical_accuracy"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics_auc() -> None:
    """Tests the keras layers metrics.AUC functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.AUC", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.AUC"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.AUC", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.AUC"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics__false_negatives() -> None:
    """Tests the keras layers metrics.FalseNegatives functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.FalseNegatives", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.FalseNegatives"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.FalseNegatives", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.FalseNegatives"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics__false_positives() -> None:
    """Tests the keras layers metrics.FalsePositives functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.FalsePositives", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.FalsePositives"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.FalsePositives", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.FalsePositives"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics__precision() -> None:
    """Tests the keras layers metrics.Precision functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.Precision", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.Precision"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.Precision", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.Precision"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics__precision_at_recall() -> None:
    """Tests the keras layers metrics.PrecisionAtRecall functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.PrecisionAtRecall", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.PrecisionAtRecall"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.PrecisionAtRecall", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.PrecisionAtRecall"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics__recall() -> None:
    """Tests the keras layers metrics.Recall functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.Recall", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.Recall"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.Recall", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.Recall"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics__recall_at_precision() -> None:
    """Tests the keras layers metrics.RecallAtPrecision functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.RecallAtPrecision", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.RecallAtPrecision"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.RecallAtPrecision", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.RecallAtPrecision"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics__sensitivity_at_specificity() -> None:
    """Tests the keras layers metrics.SensitivityAtSpecificity functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.SensitivityAtSpecificity", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.SensitivityAtSpecificity"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.SensitivityAtSpecificity", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.SensitivityAtSpecificity"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics__specificity_at_sensitivity() -> None:
    """Tests the keras layers metrics.SpecificityAtSensitivity functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.SpecificityAtSensitivity", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.SpecificityAtSensitivity"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.SpecificityAtSensitivity", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.SpecificityAtSensitivity"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics__true_negatives() -> None:
    """Tests the keras layers metrics.TrueNegatives functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.TrueNegatives", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.TrueNegatives"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.TrueNegatives", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.TrueNegatives"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics__true_positives() -> None:
    """Tests the keras layers metrics.TruePositives functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.TruePositives", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.TruePositives"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.TruePositives", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.TruePositives"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics__concordance_correlation() -> None:
    """Tests the keras layers metrics.ConcordanceCorrelation functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.ConcordanceCorrelation", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.ConcordanceCorrelation"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.ConcordanceCorrelation", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.ConcordanceCorrelation"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics__pearson_correlation() -> None:
    """Tests the keras layers metrics.PearsonCorrelation functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.PearsonCorrelation", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.PearsonCorrelation"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.PearsonCorrelation", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.PearsonCorrelation"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics_concordance_correlation() -> None:
    """Tests the keras layers metrics.concordance_correlation functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.concordance_correlation", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.concordance_correlation"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.concordance_correlation", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.concordance_correlation"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics_pearson_correlation() -> None:
    """Tests the keras layers metrics.pearson_correlation functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.pearson_correlation", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.pearson_correlation"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.pearson_correlation", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.pearson_correlation"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics_f1_score() -> None:
    """Tests the keras layers metrics.F1Score functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.F1Score", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.F1Score"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.F1Score", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.F1Score"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics_f_beta_score() -> None:
    """Tests the keras layers metrics.FBetaScore functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.FBetaScore", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.FBetaScore"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.FBetaScore", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.FBetaScore"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics__categorical_hinge() -> None:
    """Tests the keras layers metrics.CategoricalHinge functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.CategoricalHinge", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.CategoricalHinge"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.CategoricalHinge", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.CategoricalHinge"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics__hinge() -> None:
    """Tests the keras layers metrics.Hinge functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.Hinge", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.Hinge"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.Hinge", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.Hinge"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics__squared_hinge() -> None:
    """Tests the keras layers metrics.SquaredHinge functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.SquaredHinge", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.SquaredHinge"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.SquaredHinge", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.SquaredHinge"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics__binary_io_u() -> None:
    """Tests the keras layers metrics.BinaryIoU functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.BinaryIoU", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.BinaryIoU"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.BinaryIoU", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.BinaryIoU"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics__io_u() -> None:
    """Tests the keras layers metrics.IoU functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.IoU", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.IoU"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.IoU", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.IoU"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics__mean_io_u() -> None:
    """Tests the keras layers metrics.MeanIoU functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.MeanIoU", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.MeanIoU"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.MeanIoU", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.MeanIoU"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics__one_hot_io_u() -> None:
    """Tests the keras layers metrics.OneHotIoU functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.OneHotIoU", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.OneHotIoU"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.OneHotIoU", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.OneHotIoU"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics__one_hot_mean_io_u() -> None:
    """Tests the keras layers metrics.OneHotMeanIoU functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.OneHotMeanIoU", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.OneHotMeanIoU"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.OneHotMeanIoU", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.OneHotMeanIoU"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics__metric() -> None:
    """Tests the keras layers metrics.Metric functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.Metric", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.Metric"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.Metric", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.Metric"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics__binary_crossentropy() -> None:
    """Tests the keras layers metrics.BinaryCrossentropy functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.BinaryCrossentropy", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.BinaryCrossentropy"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.BinaryCrossentropy", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.BinaryCrossentropy"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics__categorical_crossentropy() -> None:
    """Tests the keras layers metrics.CategoricalCrossentropy functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.CategoricalCrossentropy", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.CategoricalCrossentropy"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.CategoricalCrossentropy", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.CategoricalCrossentropy"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics_kl_divergence_ext() -> None:
    """Tests the keras layers metrics.KLDivergence functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.KLDivergence", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.KLDivergence"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.KLDivergence", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.KLDivergence"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics__poisson() -> None:
    """Tests the keras layers metrics.Poisson functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.Poisson", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.Poisson"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.Poisson", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.Poisson"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics__sparse_categorical_crossentropy() -> None:
    """Tests the keras layers metrics.SparseCategoricalCrossentropy functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.SparseCategoricalCrossentropy", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.SparseCategoricalCrossentropy"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.SparseCategoricalCrossentropy", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.SparseCategoricalCrossentropy"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics__mean() -> None:
    """Tests the keras layers metrics.Mean functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.Mean", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.Mean"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.Mean", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.Mean"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics__mean_metric_wrapper() -> None:
    """Tests the keras layers metrics.MeanMetricWrapper functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.MeanMetricWrapper", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.MeanMetricWrapper"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.MeanMetricWrapper", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.MeanMetricWrapper"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics__sum() -> None:
    """Tests the keras layers metrics.Sum functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.Sum", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.Sum"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.Sum", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.Sum"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics__cosine_similarity() -> None:
    """Tests the keras layers metrics.CosineSimilarity functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.CosineSimilarity", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.CosineSimilarity"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.CosineSimilarity", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.CosineSimilarity"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics__log_cosh_error() -> None:
    """Tests the keras layers metrics.LogCoshError functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.LogCoshError", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.LogCoshError"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.LogCoshError", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.LogCoshError"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics__mean_absolute_error() -> None:
    """Tests the keras layers metrics.MeanAbsoluteError functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.MeanAbsoluteError", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.MeanAbsoluteError"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.MeanAbsoluteError", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.MeanAbsoluteError"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics__mean_absolute_percentage_error() -> None:
    """Tests the keras layers metrics.MeanAbsolutePercentageError functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.MeanAbsolutePercentageError", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.MeanAbsolutePercentageError"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.MeanAbsolutePercentageError", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.MeanAbsolutePercentageError"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics__mean_squared_error() -> None:
    """Tests the keras layers metrics.MeanSquaredError functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.MeanSquaredError", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.MeanSquaredError"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.MeanSquaredError", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.MeanSquaredError"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics__mean_squared_logarithmic_error() -> None:
    """Tests the keras layers metrics.MeanSquaredLogarithmicError functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.MeanSquaredLogarithmicError", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.MeanSquaredLogarithmicError"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.MeanSquaredLogarithmicError", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.MeanSquaredLogarithmicError"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics_r2_score() -> None:
    """Tests the keras layers metrics.R2Score functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.R2Score", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.R2Score"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.R2Score", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.R2Score"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_metrics__root_mean_squared_error() -> None:
    """Tests the keras layers metrics.RootMeanSquaredError functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "metrics.RootMeanSquaredError", inputs=["a"])

    KERAS_LAYERS_MAPPING["metrics.RootMeanSquaredError"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "metrics.RootMeanSquaredError", inputs=[])
    KERAS_LAYERS_MAPPING["metrics.RootMeanSquaredError"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_losses_deserialize() -> None:
    """Tests the keras layers losses.deserialize functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "losses.deserialize", inputs=["a"])

    KERAS_LAYERS_MAPPING["losses.deserialize"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "losses.deserialize", inputs=[])
    KERAS_LAYERS_MAPPING["losses.deserialize"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_losses_get() -> None:
    """Tests the keras layers losses.get functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "losses.get", inputs=["a"])

    KERAS_LAYERS_MAPPING["losses.get"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "losses.get", inputs=[])
    KERAS_LAYERS_MAPPING["losses.get"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_losses_serialize() -> None:
    """Tests the keras layers losses.serialize functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "losses.serialize", inputs=["a"])

    KERAS_LAYERS_MAPPING["losses.serialize"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "losses.serialize", inputs=[])
    KERAS_LAYERS_MAPPING["losses.serialize"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_losses__loss() -> None:
    """Tests the keras layers losses.Loss functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "losses.Loss", inputs=["a"])

    KERAS_LAYERS_MAPPING["losses.Loss"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "losses.Loss", inputs=[])
    KERAS_LAYERS_MAPPING["losses.Loss"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_losses_ctc() -> None:
    """Tests the keras layers losses.CTC functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "losses.CTC", inputs=["a"])

    KERAS_LAYERS_MAPPING["losses.CTC"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "losses.CTC", inputs=[])
    KERAS_LAYERS_MAPPING["losses.CTC"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_losses__binary_crossentropy() -> None:
    """Tests the keras layers losses.BinaryCrossentropy functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "losses.BinaryCrossentropy", inputs=["a"])

    KERAS_LAYERS_MAPPING["losses.BinaryCrossentropy"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "losses.BinaryCrossentropy", inputs=[])
    KERAS_LAYERS_MAPPING["losses.BinaryCrossentropy"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_losses__binary_focal_crossentropy() -> None:
    """Tests the keras layers losses.BinaryFocalCrossentropy functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "losses.BinaryFocalCrossentropy", inputs=["a"])

    KERAS_LAYERS_MAPPING["losses.BinaryFocalCrossentropy"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "losses.BinaryFocalCrossentropy", inputs=[])
    KERAS_LAYERS_MAPPING["losses.BinaryFocalCrossentropy"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_losses__categorical_crossentropy() -> None:
    """Tests the keras layers losses.CategoricalCrossentropy functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "losses.CategoricalCrossentropy", inputs=["a"])

    KERAS_LAYERS_MAPPING["losses.CategoricalCrossentropy"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "losses.CategoricalCrossentropy", inputs=[])
    KERAS_LAYERS_MAPPING["losses.CategoricalCrossentropy"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_losses__categorical_focal_crossentropy() -> None:
    """Tests the keras layers losses.CategoricalFocalCrossentropy functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "losses.CategoricalFocalCrossentropy", inputs=["a"])

    KERAS_LAYERS_MAPPING["losses.CategoricalFocalCrossentropy"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "losses.CategoricalFocalCrossentropy", inputs=[])
    KERAS_LAYERS_MAPPING["losses.CategoricalFocalCrossentropy"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_losses__categorical_generalized_cross_entropy() -> None:
    """Tests the keras layers losses.CategoricalGeneralizedCrossEntropy functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "losses.CategoricalGeneralizedCrossEntropy", inputs=["a"])

    KERAS_LAYERS_MAPPING["losses.CategoricalGeneralizedCrossEntropy"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "losses.CategoricalGeneralizedCrossEntropy", inputs=[])
    KERAS_LAYERS_MAPPING["losses.CategoricalGeneralizedCrossEntropy"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_losses__categorical_hinge() -> None:
    """Tests the keras layers losses.CategoricalHinge functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "losses.CategoricalHinge", inputs=["a"])

    KERAS_LAYERS_MAPPING["losses.CategoricalHinge"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "losses.CategoricalHinge", inputs=[])
    KERAS_LAYERS_MAPPING["losses.CategoricalHinge"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_losses__circle() -> None:
    """Tests the keras layers losses.Circle functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "losses.Circle", inputs=["a"])

    KERAS_LAYERS_MAPPING["losses.Circle"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "losses.Circle", inputs=[])
    KERAS_LAYERS_MAPPING["losses.Circle"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_losses__cosine_similarity() -> None:
    """Tests the keras layers losses.CosineSimilarity functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "losses.CosineSimilarity", inputs=["a"])

    KERAS_LAYERS_MAPPING["losses.CosineSimilarity"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "losses.CosineSimilarity", inputs=[])
    KERAS_LAYERS_MAPPING["losses.CosineSimilarity"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_losses__dice() -> None:
    """Tests the keras layers losses.Dice functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "losses.Dice", inputs=["a"])

    KERAS_LAYERS_MAPPING["losses.Dice"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "losses.Dice", inputs=[])
    KERAS_LAYERS_MAPPING["losses.Dice"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_losses__hinge() -> None:
    """Tests the keras layers losses.Hinge functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "losses.Hinge", inputs=["a"])

    KERAS_LAYERS_MAPPING["losses.Hinge"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "losses.Hinge", inputs=[])
    KERAS_LAYERS_MAPPING["losses.Hinge"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_losses__huber() -> None:
    """Tests the keras layers losses.Huber functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "losses.Huber", inputs=["a"])

    KERAS_LAYERS_MAPPING["losses.Huber"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "losses.Huber", inputs=[])
    KERAS_LAYERS_MAPPING["losses.Huber"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_losses_kl_divergence() -> None:
    """Tests the keras layers losses.KLDivergence functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "losses.KLDivergence", inputs=["a"])

    KERAS_LAYERS_MAPPING["losses.KLDivergence"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "losses.KLDivergence", inputs=[])
    KERAS_LAYERS_MAPPING["losses.KLDivergence"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_losses__log_cosh() -> None:
    """Tests the keras layers losses.LogCosh functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "losses.LogCosh", inputs=["a"])

    KERAS_LAYERS_MAPPING["losses.LogCosh"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "losses.LogCosh", inputs=[])
    KERAS_LAYERS_MAPPING["losses.LogCosh"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_losses__mean_absolute_error() -> None:
    """Tests the keras layers losses.MeanAbsoluteError functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "losses.MeanAbsoluteError", inputs=["a"])

    KERAS_LAYERS_MAPPING["losses.MeanAbsoluteError"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "losses.MeanAbsoluteError", inputs=[])
    KERAS_LAYERS_MAPPING["losses.MeanAbsoluteError"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_losses__mean_absolute_percentage_error() -> None:
    """Tests the keras layers losses.MeanAbsolutePercentageError functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "losses.MeanAbsolutePercentageError", inputs=["a"])

    KERAS_LAYERS_MAPPING["losses.MeanAbsolutePercentageError"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "losses.MeanAbsolutePercentageError", inputs=[])
    KERAS_LAYERS_MAPPING["losses.MeanAbsolutePercentageError"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_losses__mean_squared_error() -> None:
    """Tests the keras layers losses.MeanSquaredError functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "losses.MeanSquaredError", inputs=["a"])

    KERAS_LAYERS_MAPPING["losses.MeanSquaredError"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "losses.MeanSquaredError", inputs=[])
    KERAS_LAYERS_MAPPING["losses.MeanSquaredError"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_losses__mean_squared_logarithmic_error() -> None:
    """Tests the keras layers losses.MeanSquaredLogarithmicError functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "losses.MeanSquaredLogarithmicError", inputs=["a"])

    KERAS_LAYERS_MAPPING["losses.MeanSquaredLogarithmicError"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "losses.MeanSquaredLogarithmicError", inputs=[])
    KERAS_LAYERS_MAPPING["losses.MeanSquaredLogarithmicError"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_losses__poisson() -> None:
    """Tests the keras layers losses.Poisson functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "losses.Poisson", inputs=["a"])

    KERAS_LAYERS_MAPPING["losses.Poisson"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "losses.Poisson", inputs=[])
    KERAS_LAYERS_MAPPING["losses.Poisson"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_losses__sparse_categorical_crossentropy() -> None:
    """Tests the keras layers losses.SparseCategoricalCrossentropy functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "losses.SparseCategoricalCrossentropy", inputs=["a"])

    KERAS_LAYERS_MAPPING["losses.SparseCategoricalCrossentropy"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "losses.SparseCategoricalCrossentropy", inputs=[])
    KERAS_LAYERS_MAPPING["losses.SparseCategoricalCrossentropy"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_losses__squared_hinge() -> None:
    """Tests the keras layers losses.SquaredHinge functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "losses.SquaredHinge", inputs=["a"])

    KERAS_LAYERS_MAPPING["losses.SquaredHinge"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "losses.SquaredHinge", inputs=[])
    KERAS_LAYERS_MAPPING["losses.SquaredHinge"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_losses__tversky() -> None:
    """Tests the keras layers losses.Tversky functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "losses.Tversky", inputs=["a"])

    KERAS_LAYERS_MAPPING["losses.Tversky"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "losses.Tversky", inputs=[])
    KERAS_LAYERS_MAPPING["losses.Tversky"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_losses_binary_crossentropy() -> None:
    """Tests the keras layers losses.binary_crossentropy functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "losses.binary_crossentropy", inputs=["a"])

    KERAS_LAYERS_MAPPING["losses.binary_crossentropy"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "losses.binary_crossentropy", inputs=[])
    KERAS_LAYERS_MAPPING["losses.binary_crossentropy"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_losses_binary_focal_crossentropy() -> None:
    """Tests the keras layers losses.binary_focal_crossentropy functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "losses.binary_focal_crossentropy", inputs=["a"])

    KERAS_LAYERS_MAPPING["losses.binary_focal_crossentropy"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "losses.binary_focal_crossentropy", inputs=[])
    KERAS_LAYERS_MAPPING["losses.binary_focal_crossentropy"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_losses_categorical_crossentropy() -> None:
    """Tests the keras layers losses.categorical_crossentropy functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "losses.categorical_crossentropy", inputs=["a"])

    KERAS_LAYERS_MAPPING["losses.categorical_crossentropy"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "losses.categorical_crossentropy", inputs=[])
    KERAS_LAYERS_MAPPING["losses.categorical_crossentropy"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_losses_categorical_focal_crossentropy() -> None:
    """Tests the keras layers losses.categorical_focal_crossentropy functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "losses.categorical_focal_crossentropy", inputs=["a"])

    KERAS_LAYERS_MAPPING["losses.categorical_focal_crossentropy"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "losses.categorical_focal_crossentropy", inputs=[])
    KERAS_LAYERS_MAPPING["losses.categorical_focal_crossentropy"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_losses_categorical_generalized_cross_entropy() -> None:
    """Tests the keras layers losses.categorical_generalized_cross_entropy functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "losses.categorical_generalized_cross_entropy", inputs=["a"])

    KERAS_LAYERS_MAPPING["losses.categorical_generalized_cross_entropy"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "losses.categorical_generalized_cross_entropy", inputs=[])
    KERAS_LAYERS_MAPPING["losses.categorical_generalized_cross_entropy"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_losses_categorical_hinge() -> None:
    """Tests the keras layers losses.categorical_hinge functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "losses.categorical_hinge", inputs=["a"])

    KERAS_LAYERS_MAPPING["losses.categorical_hinge"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "losses.categorical_hinge", inputs=[])
    KERAS_LAYERS_MAPPING["losses.categorical_hinge"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_losses_circle() -> None:
    """Tests the keras layers losses.circle functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "losses.circle", inputs=["a"])

    KERAS_LAYERS_MAPPING["losses.circle"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "losses.circle", inputs=[])
    KERAS_LAYERS_MAPPING["losses.circle"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_losses_cosine_similarity() -> None:
    """Tests the keras layers losses.cosine_similarity functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "losses.cosine_similarity", inputs=["a"])

    KERAS_LAYERS_MAPPING["losses.cosine_similarity"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "losses.cosine_similarity", inputs=[])
    KERAS_LAYERS_MAPPING["losses.cosine_similarity"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_losses_ctc_ext() -> None:
    """Tests the keras layers losses.ctc functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "losses.ctc", inputs=["a"])

    KERAS_LAYERS_MAPPING["losses.ctc"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "losses.ctc", inputs=[])
    KERAS_LAYERS_MAPPING["losses.ctc"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_losses_dice() -> None:
    """Tests the keras layers losses.dice functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "losses.dice", inputs=["a"])

    KERAS_LAYERS_MAPPING["losses.dice"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "losses.dice", inputs=[])
    KERAS_LAYERS_MAPPING["losses.dice"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_losses_hinge() -> None:
    """Tests the keras layers losses.hinge functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "losses.hinge", inputs=["a"])

    KERAS_LAYERS_MAPPING["losses.hinge"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "losses.hinge", inputs=[])
    KERAS_LAYERS_MAPPING["losses.hinge"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_losses_huber() -> None:
    """Tests the keras layers losses.huber functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "losses.huber", inputs=["a"])

    KERAS_LAYERS_MAPPING["losses.huber"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "losses.huber", inputs=[])
    KERAS_LAYERS_MAPPING["losses.huber"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_losses_kl_divergence_ext() -> None:
    """Tests the keras layers losses.kl_divergence functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "losses.kl_divergence", inputs=["a"])

    KERAS_LAYERS_MAPPING["losses.kl_divergence"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "losses.kl_divergence", inputs=[])
    KERAS_LAYERS_MAPPING["losses.kl_divergence"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_losses_log_cosh() -> None:
    """Tests the keras layers losses.log_cosh functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "losses.log_cosh", inputs=["a"])

    KERAS_LAYERS_MAPPING["losses.log_cosh"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "losses.log_cosh", inputs=[])
    KERAS_LAYERS_MAPPING["losses.log_cosh"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_losses_mean_absolute_error() -> None:
    """Tests the keras layers losses.mean_absolute_error functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "losses.mean_absolute_error", inputs=["a"])

    KERAS_LAYERS_MAPPING["losses.mean_absolute_error"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "losses.mean_absolute_error", inputs=[])
    KERAS_LAYERS_MAPPING["losses.mean_absolute_error"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_losses_mean_absolute_percentage_error() -> None:
    """Tests the keras layers losses.mean_absolute_percentage_error functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "losses.mean_absolute_percentage_error", inputs=["a"])

    KERAS_LAYERS_MAPPING["losses.mean_absolute_percentage_error"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "losses.mean_absolute_percentage_error", inputs=[])
    KERAS_LAYERS_MAPPING["losses.mean_absolute_percentage_error"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_losses_mean_squared_error() -> None:
    """Tests the keras layers losses.mean_squared_error functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "losses.mean_squared_error", inputs=["a"])

    KERAS_LAYERS_MAPPING["losses.mean_squared_error"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "losses.mean_squared_error", inputs=[])
    KERAS_LAYERS_MAPPING["losses.mean_squared_error"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_losses_mean_squared_logarithmic_error() -> None:
    """Tests the keras layers losses.mean_squared_logarithmic_error functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "losses.mean_squared_logarithmic_error", inputs=["a"])

    KERAS_LAYERS_MAPPING["losses.mean_squared_logarithmic_error"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "losses.mean_squared_logarithmic_error", inputs=[])
    KERAS_LAYERS_MAPPING["losses.mean_squared_logarithmic_error"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_losses_poisson() -> None:
    """Tests the keras layers losses.poisson functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "losses.poisson", inputs=["a"])

    KERAS_LAYERS_MAPPING["losses.poisson"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "losses.poisson", inputs=[])
    KERAS_LAYERS_MAPPING["losses.poisson"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_losses_sparse_categorical_crossentropy() -> None:
    """Tests the keras layers losses.sparse_categorical_crossentropy functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "losses.sparse_categorical_crossentropy", inputs=["a"])

    KERAS_LAYERS_MAPPING["losses.sparse_categorical_crossentropy"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "losses.sparse_categorical_crossentropy", inputs=[])
    KERAS_LAYERS_MAPPING["losses.sparse_categorical_crossentropy"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_losses_squared_hinge() -> None:
    """Tests the keras layers losses.squared_hinge functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "losses.squared_hinge", inputs=["a"])

    KERAS_LAYERS_MAPPING["losses.squared_hinge"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "losses.squared_hinge", inputs=[])
    KERAS_LAYERS_MAPPING["losses.squared_hinge"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_losses_tversky() -> None:
    """Tests the keras layers losses.tversky functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "losses.tversky", inputs=["a"])

    KERAS_LAYERS_MAPPING["losses.tversky"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "losses.tversky", inputs=[])
    KERAS_LAYERS_MAPPING["losses.tversky"](builder, node_empty)
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


def test_keras_layers_missing_datasets_cifar10_load_data() -> None:
    """Tests the keras layers datasets.cifar10.load_data functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "datasets.cifar10.load_data", inputs=["a"])

    KERAS_LAYERS_MAPPING["datasets.cifar10.load_data"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "datasets.cifar10.load_data", inputs=[])
    KERAS_LAYERS_MAPPING["datasets.cifar10.load_data"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_datasets_fashion_mnist_load_data() -> None:
    """Tests the keras layers datasets.fashion_mnist.load_data functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "datasets.fashion_mnist.load_data", inputs=["a"])

    KERAS_LAYERS_MAPPING["datasets.fashion_mnist.load_data"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "datasets.fashion_mnist.load_data", inputs=[])
    KERAS_LAYERS_MAPPING["datasets.fashion_mnist.load_data"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_datasets_boston_housing_load_data() -> None:
    """Tests the keras layers datasets.boston_housing.load_data functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "datasets.boston_housing.load_data", inputs=["a"])

    KERAS_LAYERS_MAPPING["datasets.boston_housing.load_data"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "datasets.boston_housing.load_data", inputs=[])
    KERAS_LAYERS_MAPPING["datasets.boston_housing.load_data"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_datasets_california_housing_load_data() -> None:
    """Tests the keras layers datasets.california_housing.load_data functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "datasets.california_housing.load_data", inputs=["a"])

    KERAS_LAYERS_MAPPING["datasets.california_housing.load_data"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "datasets.california_housing.load_data", inputs=[])
    KERAS_LAYERS_MAPPING["datasets.california_housing.load_data"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_datasets_cifar100_load_data() -> None:
    """Tests the keras layers datasets.cifar100.load_data functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "datasets.cifar100.load_data", inputs=["a"])

    KERAS_LAYERS_MAPPING["datasets.cifar100.load_data"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "datasets.cifar100.load_data", inputs=[])
    KERAS_LAYERS_MAPPING["datasets.cifar100.load_data"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_datasets_mnist_load_data() -> None:
    """Tests the keras layers datasets.mnist.load_data functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "datasets.mnist.load_data", inputs=["a"])

    KERAS_LAYERS_MAPPING["datasets.mnist.load_data"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "datasets.mnist.load_data", inputs=[])
    KERAS_LAYERS_MAPPING["datasets.mnist.load_data"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_datasets_imdb_get_word_index() -> None:
    """Tests the keras layers datasets.imdb.get_word_index functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "datasets.imdb.get_word_index", inputs=["a"])

    KERAS_LAYERS_MAPPING["datasets.imdb.get_word_index"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "datasets.imdb.get_word_index", inputs=[])
    KERAS_LAYERS_MAPPING["datasets.imdb.get_word_index"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_datasets_imdb_load_data() -> None:
    """Tests the keras layers datasets.imdb.load_data functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "datasets.imdb.load_data", inputs=["a"])

    KERAS_LAYERS_MAPPING["datasets.imdb.load_data"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "datasets.imdb.load_data", inputs=[])
    KERAS_LAYERS_MAPPING["datasets.imdb.load_data"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_datasets_reuters_get_label_names() -> None:
    """Tests the keras layers datasets.reuters.get_label_names functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "datasets.reuters.get_label_names", inputs=["a"])

    KERAS_LAYERS_MAPPING["datasets.reuters.get_label_names"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "datasets.reuters.get_label_names", inputs=[])
    KERAS_LAYERS_MAPPING["datasets.reuters.get_label_names"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_datasets_reuters_get_word_index() -> None:
    """Tests the keras layers datasets.reuters.get_word_index functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "datasets.reuters.get_word_index", inputs=["a"])

    KERAS_LAYERS_MAPPING["datasets.reuters.get_word_index"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "datasets.reuters.get_word_index", inputs=[])
    KERAS_LAYERS_MAPPING["datasets.reuters.get_word_index"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_datasets_reuters_load_data() -> None:
    """Tests the keras layers datasets.reuters.load_data functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "datasets.reuters.load_data", inputs=["a"])

    KERAS_LAYERS_MAPPING["datasets.reuters.load_data"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "datasets.reuters.load_data", inputs=[])
    KERAS_LAYERS_MAPPING["datasets.reuters.load_data"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_dtype_policies_deserialize() -> None:
    """Tests the keras layers dtype_policies.deserialize functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "dtype_policies.deserialize", inputs=["a"])

    KERAS_LAYERS_MAPPING["dtype_policies.deserialize"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "dtype_policies.deserialize", inputs=[])
    KERAS_LAYERS_MAPPING["dtype_policies.deserialize"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_dtype_policies_get() -> None:
    """Tests the keras layers dtype_policies.get functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "dtype_policies.get", inputs=["a"])

    KERAS_LAYERS_MAPPING["dtype_policies.get"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "dtype_policies.get", inputs=[])
    KERAS_LAYERS_MAPPING["dtype_policies.get"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_dtype_policies_serialize() -> None:
    """Tests the keras layers dtype_policies.serialize functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "dtype_policies.serialize", inputs=["a"])

    KERAS_LAYERS_MAPPING["dtype_policies.serialize"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "dtype_policies.serialize", inputs=[])
    KERAS_LAYERS_MAPPING["dtype_policies.serialize"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_dtype_policies_d_type_policy() -> None:
    """Tests the keras layers dtype_policies.DTypePolicy functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "dtype_policies.DTypePolicy", inputs=["a"])

    KERAS_LAYERS_MAPPING["dtype_policies.DTypePolicy"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "dtype_policies.DTypePolicy", inputs=[])
    KERAS_LAYERS_MAPPING["dtype_policies.DTypePolicy"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_dtype_policies__float_d_type_policy() -> None:
    """Tests the keras layers dtype_policies.FloatDTypePolicy functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "dtype_policies.FloatDTypePolicy", inputs=["a"])

    KERAS_LAYERS_MAPPING["dtype_policies.FloatDTypePolicy"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "dtype_policies.FloatDTypePolicy", inputs=[])
    KERAS_LAYERS_MAPPING["dtype_policies.FloatDTypePolicy"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_dtype_policies_gptqd_type_policy() -> None:
    """Tests the keras layers dtype_policies.GPTQDTypePolicy functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "dtype_policies.GPTQDTypePolicy", inputs=["a"])

    KERAS_LAYERS_MAPPING["dtype_policies.GPTQDTypePolicy"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "dtype_policies.GPTQDTypePolicy", inputs=[])
    KERAS_LAYERS_MAPPING["dtype_policies.GPTQDTypePolicy"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_dtype_policies__quantized_d_type_policy() -> None:
    """Tests the keras layers dtype_policies.QuantizedDTypePolicy functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "dtype_policies.QuantizedDTypePolicy", inputs=["a"])

    KERAS_LAYERS_MAPPING["dtype_policies.QuantizedDTypePolicy"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "dtype_policies.QuantizedDTypePolicy", inputs=[])
    KERAS_LAYERS_MAPPING["dtype_policies.QuantizedDTypePolicy"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_dtype_policies__quantized_float8_d_type_policy() -> None:
    """Tests the keras layers dtype_policies.QuantizedFloat8DTypePolicy functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "dtype_policies.QuantizedFloat8DTypePolicy", inputs=["a"])

    KERAS_LAYERS_MAPPING["dtype_policies.QuantizedFloat8DTypePolicy"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "dtype_policies.QuantizedFloat8DTypePolicy", inputs=[])
    KERAS_LAYERS_MAPPING["dtype_policies.QuantizedFloat8DTypePolicy"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_dtype_policies_d_type_policy_map() -> None:
    """Tests the keras layers dtype_policies.DTypePolicyMap functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "dtype_policies.DTypePolicyMap", inputs=["a"])

    KERAS_LAYERS_MAPPING["dtype_policies.DTypePolicyMap"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "dtype_policies.DTypePolicyMap", inputs=[])
    KERAS_LAYERS_MAPPING["dtype_policies.DTypePolicyMap"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_mixed_precision_d_type_policy() -> None:
    """Tests the keras layers mixed_precision.DTypePolicy functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "mixed_precision.DTypePolicy", inputs=["a"])

    KERAS_LAYERS_MAPPING["mixed_precision.DTypePolicy"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "mixed_precision.DTypePolicy", inputs=[])
    KERAS_LAYERS_MAPPING["mixed_precision.DTypePolicy"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_mixed_precision__policy() -> None:
    """Tests the keras layers mixed_precision.Policy functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "mixed_precision.Policy", inputs=["a"])

    KERAS_LAYERS_MAPPING["mixed_precision.Policy"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "mixed_precision.Policy", inputs=[])
    KERAS_LAYERS_MAPPING["mixed_precision.Policy"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_mixed_precision_dtype_policy() -> None:
    """Tests the keras layers mixed_precision.dtype_policy functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "mixed_precision.dtype_policy", inputs=["a"])

    KERAS_LAYERS_MAPPING["mixed_precision.dtype_policy"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "mixed_precision.dtype_policy", inputs=[])
    KERAS_LAYERS_MAPPING["mixed_precision.dtype_policy"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_mixed_precision_global_policy() -> None:
    """Tests the keras layers mixed_precision.global_policy functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "mixed_precision.global_policy", inputs=["a"])

    KERAS_LAYERS_MAPPING["mixed_precision.global_policy"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "mixed_precision.global_policy", inputs=[])
    KERAS_LAYERS_MAPPING["mixed_precision.global_policy"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_mixed_precision_set_dtype_policy() -> None:
    """Tests the keras layers mixed_precision.set_dtype_policy functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "mixed_precision.set_dtype_policy", inputs=["a"])

    KERAS_LAYERS_MAPPING["mixed_precision.set_dtype_policy"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "mixed_precision.set_dtype_policy", inputs=[])
    KERAS_LAYERS_MAPPING["mixed_precision.set_dtype_policy"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_mixed_precision_set_global_policy() -> None:
    """Tests the keras layers mixed_precision.set_global_policy functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "mixed_precision.set_global_policy", inputs=["a"])

    KERAS_LAYERS_MAPPING["mixed_precision.set_global_policy"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "mixed_precision.set_global_policy", inputs=[])
    KERAS_LAYERS_MAPPING["mixed_precision.set_global_policy"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_mixed_precision__loss_scale_optimizer() -> None:
    """Tests the keras layers mixed_precision.LossScaleOptimizer functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "mixed_precision.LossScaleOptimizer", inputs=["a"])

    KERAS_LAYERS_MAPPING["mixed_precision.LossScaleOptimizer"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "mixed_precision.LossScaleOptimizer", inputs=[])
    KERAS_LAYERS_MAPPING["mixed_precision.LossScaleOptimizer"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_random_beta() -> None:
    """Tests the keras layers random.beta functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "random.beta", inputs=["a"])

    KERAS_LAYERS_MAPPING["random.beta"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "random.beta", inputs=[])
    KERAS_LAYERS_MAPPING["random.beta"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_random_binomial() -> None:
    """Tests the keras layers random.binomial functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "random.binomial", inputs=["a"])

    KERAS_LAYERS_MAPPING["random.binomial"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "random.binomial", inputs=[])
    KERAS_LAYERS_MAPPING["random.binomial"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_random_categorical() -> None:
    """Tests the keras layers random.categorical functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "random.categorical", inputs=["a"])

    KERAS_LAYERS_MAPPING["random.categorical"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "random.categorical", inputs=[])
    KERAS_LAYERS_MAPPING["random.categorical"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_random_dropout() -> None:
    """Tests the keras layers random.dropout functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "random.dropout", inputs=["a"])

    KERAS_LAYERS_MAPPING["random.dropout"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "random.dropout", inputs=[])
    KERAS_LAYERS_MAPPING["random.dropout"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_random_gamma() -> None:
    """Tests the keras layers random.gamma functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "random.gamma", inputs=["a"])

    KERAS_LAYERS_MAPPING["random.gamma"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "random.gamma", inputs=[])
    KERAS_LAYERS_MAPPING["random.gamma"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_random_normal() -> None:
    """Tests the keras layers random.normal functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "random.normal", inputs=["a"])

    KERAS_LAYERS_MAPPING["random.normal"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "random.normal", inputs=[])
    KERAS_LAYERS_MAPPING["random.normal"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_random_randint() -> None:
    """Tests the keras layers random.randint functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "random.randint", inputs=["a"])

    KERAS_LAYERS_MAPPING["random.randint"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "random.randint", inputs=[])
    KERAS_LAYERS_MAPPING["random.randint"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_random_shuffle() -> None:
    """Tests the keras layers random.shuffle functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "random.shuffle", inputs=["a"])

    KERAS_LAYERS_MAPPING["random.shuffle"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "random.shuffle", inputs=[])
    KERAS_LAYERS_MAPPING["random.shuffle"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_random_truncated_normal() -> None:
    """Tests the keras layers random.truncated_normal functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "random.truncated_normal", inputs=["a"])

    KERAS_LAYERS_MAPPING["random.truncated_normal"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "random.truncated_normal", inputs=[])
    KERAS_LAYERS_MAPPING["random.truncated_normal"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_random_uniform() -> None:
    """Tests the keras layers random.uniform functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "random.uniform", inputs=["a"])

    KERAS_LAYERS_MAPPING["random.uniform"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "random.uniform", inputs=[])
    KERAS_LAYERS_MAPPING["random.uniform"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_random__seed_generator() -> None:
    """Tests the keras layers random.SeedGenerator functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "random.SeedGenerator", inputs=["a"])

    KERAS_LAYERS_MAPPING["random.SeedGenerator"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "random.SeedGenerator", inputs=[])
    KERAS_LAYERS_MAPPING["random.SeedGenerator"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_config_disable_flash_attention() -> None:
    """Tests the keras layers config.disable_flash_attention functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "config.disable_flash_attention", inputs=["a"])

    KERAS_LAYERS_MAPPING["config.disable_flash_attention"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "config.disable_flash_attention", inputs=[])
    KERAS_LAYERS_MAPPING["config.disable_flash_attention"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_config_enable_flash_attention() -> None:
    """Tests the keras layers config.enable_flash_attention functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "config.enable_flash_attention", inputs=["a"])

    KERAS_LAYERS_MAPPING["config.enable_flash_attention"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "config.enable_flash_attention", inputs=[])
    KERAS_LAYERS_MAPPING["config.enable_flash_attention"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_config_epsilon() -> None:
    """Tests the keras layers config.epsilon functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "config.epsilon", inputs=["a"])

    KERAS_LAYERS_MAPPING["config.epsilon"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "config.epsilon", inputs=[])
    KERAS_LAYERS_MAPPING["config.epsilon"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_config_floatx() -> None:
    """Tests the keras layers config.floatx functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "config.floatx", inputs=["a"])

    KERAS_LAYERS_MAPPING["config.floatx"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "config.floatx", inputs=[])
    KERAS_LAYERS_MAPPING["config.floatx"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_config_image_data_format() -> None:
    """Tests the keras layers config.image_data_format functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "config.image_data_format", inputs=["a"])

    KERAS_LAYERS_MAPPING["config.image_data_format"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "config.image_data_format", inputs=[])
    KERAS_LAYERS_MAPPING["config.image_data_format"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_config_is_flash_attention_enabled() -> None:
    """Tests the keras layers config.is_flash_attention_enabled functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "config.is_flash_attention_enabled", inputs=["a"])

    KERAS_LAYERS_MAPPING["config.is_flash_attention_enabled"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "config.is_flash_attention_enabled", inputs=[])
    KERAS_LAYERS_MAPPING["config.is_flash_attention_enabled"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_config_is_nnx_enabled() -> None:
    """Tests the keras layers config.is_nnx_enabled functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "config.is_nnx_enabled", inputs=["a"])

    KERAS_LAYERS_MAPPING["config.is_nnx_enabled"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "config.is_nnx_enabled", inputs=[])
    KERAS_LAYERS_MAPPING["config.is_nnx_enabled"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_config_max_epochs() -> None:
    """Tests the keras layers config.max_epochs functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "config.max_epochs", inputs=["a"])

    KERAS_LAYERS_MAPPING["config.max_epochs"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "config.max_epochs", inputs=[])
    KERAS_LAYERS_MAPPING["config.max_epochs"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_config_max_steps_per_epoch() -> None:
    """Tests the keras layers config.max_steps_per_epoch functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "config.max_steps_per_epoch", inputs=["a"])

    KERAS_LAYERS_MAPPING["config.max_steps_per_epoch"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "config.max_steps_per_epoch", inputs=[])
    KERAS_LAYERS_MAPPING["config.max_steps_per_epoch"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_config_set_epsilon() -> None:
    """Tests the keras layers config.set_epsilon functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "config.set_epsilon", inputs=["a"])

    KERAS_LAYERS_MAPPING["config.set_epsilon"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "config.set_epsilon", inputs=[])
    KERAS_LAYERS_MAPPING["config.set_epsilon"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_config_set_floatx() -> None:
    """Tests the keras layers config.set_floatx functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "config.set_floatx", inputs=["a"])

    KERAS_LAYERS_MAPPING["config.set_floatx"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "config.set_floatx", inputs=[])
    KERAS_LAYERS_MAPPING["config.set_floatx"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_config_set_image_data_format() -> None:
    """Tests the keras layers config.set_image_data_format functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "config.set_image_data_format", inputs=["a"])

    KERAS_LAYERS_MAPPING["config.set_image_data_format"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "config.set_image_data_format", inputs=[])
    KERAS_LAYERS_MAPPING["config.set_image_data_format"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_config_set_max_epochs() -> None:
    """Tests the keras layers config.set_max_epochs functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "config.set_max_epochs", inputs=["a"])

    KERAS_LAYERS_MAPPING["config.set_max_epochs"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "config.set_max_epochs", inputs=[])
    KERAS_LAYERS_MAPPING["config.set_max_epochs"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_config_set_max_steps_per_epoch() -> None:
    """Tests the keras layers config.set_max_steps_per_epoch functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "config.set_max_steps_per_epoch", inputs=["a"])

    KERAS_LAYERS_MAPPING["config.set_max_steps_per_epoch"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "config.set_max_steps_per_epoch", inputs=[])
    KERAS_LAYERS_MAPPING["config.set_max_steps_per_epoch"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_config_dtype_policy() -> None:
    """Tests the keras layers config.dtype_policy functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "config.dtype_policy", inputs=["a"])

    KERAS_LAYERS_MAPPING["config.dtype_policy"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "config.dtype_policy", inputs=[])
    KERAS_LAYERS_MAPPING["config.dtype_policy"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_config_set_dtype_policy() -> None:
    """Tests the keras layers config.set_dtype_policy functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "config.set_dtype_policy", inputs=["a"])

    KERAS_LAYERS_MAPPING["config.set_dtype_policy"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "config.set_dtype_policy", inputs=[])
    KERAS_LAYERS_MAPPING["config.set_dtype_policy"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_config_enable_unsafe_deserialization() -> None:
    """Tests the keras layers config.enable_unsafe_deserialization functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "config.enable_unsafe_deserialization", inputs=["a"])

    KERAS_LAYERS_MAPPING["config.enable_unsafe_deserialization"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "config.enable_unsafe_deserialization", inputs=[])
    KERAS_LAYERS_MAPPING["config.enable_unsafe_deserialization"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_config_set_backend() -> None:
    """Tests the keras layers config.set_backend functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "config.set_backend", inputs=["a"])

    KERAS_LAYERS_MAPPING["config.set_backend"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "config.set_backend", inputs=[])
    KERAS_LAYERS_MAPPING["config.set_backend"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_config_disable_interactive_logging() -> None:
    """Tests the keras layers config.disable_interactive_logging functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "config.disable_interactive_logging", inputs=["a"])

    KERAS_LAYERS_MAPPING["config.disable_interactive_logging"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "config.disable_interactive_logging", inputs=[])
    KERAS_LAYERS_MAPPING["config.disable_interactive_logging"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_config_enable_interactive_logging() -> None:
    """Tests the keras layers config.enable_interactive_logging functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "config.enable_interactive_logging", inputs=["a"])

    KERAS_LAYERS_MAPPING["config.enable_interactive_logging"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "config.enable_interactive_logging", inputs=[])
    KERAS_LAYERS_MAPPING["config.enable_interactive_logging"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_config_is_interactive_logging_enabled() -> None:
    """Tests the keras layers config.is_interactive_logging_enabled functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "config.is_interactive_logging_enabled", inputs=["a"])

    KERAS_LAYERS_MAPPING["config.is_interactive_logging_enabled"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "config.is_interactive_logging_enabled", inputs=[])
    KERAS_LAYERS_MAPPING["config.is_interactive_logging_enabled"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_config_disable_traceback_filtering() -> None:
    """Tests the keras layers config.disable_traceback_filtering functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "config.disable_traceback_filtering", inputs=["a"])

    KERAS_LAYERS_MAPPING["config.disable_traceback_filtering"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "config.disable_traceback_filtering", inputs=[])
    KERAS_LAYERS_MAPPING["config.disable_traceback_filtering"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_config_enable_traceback_filtering() -> None:
    """Tests the keras layers config.enable_traceback_filtering functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "config.enable_traceback_filtering", inputs=["a"])

    KERAS_LAYERS_MAPPING["config.enable_traceback_filtering"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "config.enable_traceback_filtering", inputs=[])
    KERAS_LAYERS_MAPPING["config.enable_traceback_filtering"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_config_is_traceback_filtering_enabled() -> None:
    """Tests the keras layers config.is_traceback_filtering_enabled functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "config.is_traceback_filtering_enabled", inputs=["a"])

    KERAS_LAYERS_MAPPING["config.is_traceback_filtering_enabled"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "config.is_traceback_filtering_enabled", inputs=[])
    KERAS_LAYERS_MAPPING["config.is_traceback_filtering_enabled"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_distribution__data_parallel() -> None:
    """Tests the keras layers distribution.DataParallel functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "distribution.DataParallel", inputs=["a"])

    KERAS_LAYERS_MAPPING["distribution.DataParallel"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "distribution.DataParallel", inputs=[])
    KERAS_LAYERS_MAPPING["distribution.DataParallel"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_distribution__device_mesh() -> None:
    """Tests the keras layers distribution.DeviceMesh functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "distribution.DeviceMesh", inputs=["a"])

    KERAS_LAYERS_MAPPING["distribution.DeviceMesh"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "distribution.DeviceMesh", inputs=[])
    KERAS_LAYERS_MAPPING["distribution.DeviceMesh"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_distribution__layout_map() -> None:
    """Tests the keras layers distribution.LayoutMap functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "distribution.LayoutMap", inputs=["a"])

    KERAS_LAYERS_MAPPING["distribution.LayoutMap"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "distribution.LayoutMap", inputs=[])
    KERAS_LAYERS_MAPPING["distribution.LayoutMap"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_distribution__model_parallel() -> None:
    """Tests the keras layers distribution.ModelParallel functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "distribution.ModelParallel", inputs=["a"])

    KERAS_LAYERS_MAPPING["distribution.ModelParallel"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "distribution.ModelParallel", inputs=[])
    KERAS_LAYERS_MAPPING["distribution.ModelParallel"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_distribution__tensor_layout() -> None:
    """Tests the keras layers distribution.TensorLayout functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "distribution.TensorLayout", inputs=["a"])

    KERAS_LAYERS_MAPPING["distribution.TensorLayout"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "distribution.TensorLayout", inputs=[])
    KERAS_LAYERS_MAPPING["distribution.TensorLayout"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_distribution_distribute_tensor() -> None:
    """Tests the keras layers distribution.distribute_tensor functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "distribution.distribute_tensor", inputs=["a"])

    KERAS_LAYERS_MAPPING["distribution.distribute_tensor"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "distribution.distribute_tensor", inputs=[])
    KERAS_LAYERS_MAPPING["distribution.distribute_tensor"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_distribution_distribution() -> None:
    """Tests the keras layers distribution.distribution functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "distribution.distribution", inputs=["a"])

    KERAS_LAYERS_MAPPING["distribution.distribution"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "distribution.distribution", inputs=[])
    KERAS_LAYERS_MAPPING["distribution.distribution"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_distribution_get_device_count() -> None:
    """Tests the keras layers distribution.get_device_count functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "distribution.get_device_count", inputs=["a"])

    KERAS_LAYERS_MAPPING["distribution.get_device_count"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "distribution.get_device_count", inputs=[])
    KERAS_LAYERS_MAPPING["distribution.get_device_count"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_distribution_initialize() -> None:
    """Tests the keras layers distribution.initialize functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "distribution.initialize", inputs=["a"])

    KERAS_LAYERS_MAPPING["distribution.initialize"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "distribution.initialize", inputs=[])
    KERAS_LAYERS_MAPPING["distribution.initialize"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_distribution_list_devices() -> None:
    """Tests the keras layers distribution.list_devices functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "distribution.list_devices", inputs=["a"])

    KERAS_LAYERS_MAPPING["distribution.list_devices"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "distribution.list_devices", inputs=[])
    KERAS_LAYERS_MAPPING["distribution.list_devices"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_distribution_set_distribution() -> None:
    """Tests the keras layers distribution.set_distribution functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "distribution.set_distribution", inputs=["a"])

    KERAS_LAYERS_MAPPING["distribution.set_distribution"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "distribution.set_distribution", inputs=[])
    KERAS_LAYERS_MAPPING["distribution.set_distribution"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_visualization_draw_bounding_boxes() -> None:
    """Tests the keras layers visualization.draw_bounding_boxes functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "visualization.draw_bounding_boxes", inputs=["a"])

    KERAS_LAYERS_MAPPING["visualization.draw_bounding_boxes"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "visualization.draw_bounding_boxes", inputs=[])
    KERAS_LAYERS_MAPPING["visualization.draw_bounding_boxes"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_visualization_draw_segmentation_masks() -> None:
    """Tests the keras layers visualization.draw_segmentation_masks functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "visualization.draw_segmentation_masks", inputs=["a"])

    KERAS_LAYERS_MAPPING["visualization.draw_segmentation_masks"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "visualization.draw_segmentation_masks", inputs=[])
    KERAS_LAYERS_MAPPING["visualization.draw_segmentation_masks"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_visualization_plot_bounding_box_gallery() -> None:
    """Tests the keras layers visualization.plot_bounding_box_gallery functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "visualization.plot_bounding_box_gallery", inputs=["a"])

    KERAS_LAYERS_MAPPING["visualization.plot_bounding_box_gallery"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "visualization.plot_bounding_box_gallery", inputs=[])
    KERAS_LAYERS_MAPPING["visualization.plot_bounding_box_gallery"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_visualization_plot_image_gallery() -> None:
    """Tests the keras layers visualization.plot_image_gallery functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "visualization.plot_image_gallery", inputs=["a"])

    KERAS_LAYERS_MAPPING["visualization.plot_image_gallery"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "visualization.plot_image_gallery", inputs=[])
    KERAS_LAYERS_MAPPING["visualization.plot_image_gallery"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_visualization_plot_segmentation_mask_gallery() -> None:
    """Tests the keras layers visualization.plot_segmentation_mask_gallery functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "visualization.plot_segmentation_mask_gallery", inputs=["a"])

    KERAS_LAYERS_MAPPING["visualization.plot_segmentation_mask_gallery"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "visualization.plot_segmentation_mask_gallery", inputs=[])
    KERAS_LAYERS_MAPPING["visualization.plot_segmentation_mask_gallery"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_wrappers_sk_learn_classifier() -> None:
    """Tests the keras layers wrappers.SKLearnClassifier functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "wrappers.SKLearnClassifier", inputs=["a"])

    KERAS_LAYERS_MAPPING["wrappers.SKLearnClassifier"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "wrappers.SKLearnClassifier", inputs=[])
    KERAS_LAYERS_MAPPING["wrappers.SKLearnClassifier"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_wrappers_sk_learn_regressor() -> None:
    """Tests the keras layers wrappers.SKLearnRegressor functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "wrappers.SKLearnRegressor", inputs=["a"])

    KERAS_LAYERS_MAPPING["wrappers.SKLearnRegressor"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "wrappers.SKLearnRegressor", inputs=[])
    KERAS_LAYERS_MAPPING["wrappers.SKLearnRegressor"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_wrappers_sk_learn_transformer() -> None:
    """Tests the keras layers wrappers.SKLearnTransformer functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "wrappers.SKLearnTransformer", inputs=["a"])

    KERAS_LAYERS_MAPPING["wrappers.SKLearnTransformer"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "wrappers.SKLearnTransformer", inputs=[])
    KERAS_LAYERS_MAPPING["wrappers.SKLearnTransformer"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_regularizers_deserialize() -> None:
    """Tests the keras layers regularizers.deserialize functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "regularizers.deserialize", inputs=["a"])

    KERAS_LAYERS_MAPPING["regularizers.deserialize"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "regularizers.deserialize", inputs=[])
    KERAS_LAYERS_MAPPING["regularizers.deserialize"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_regularizers_get() -> None:
    """Tests the keras layers regularizers.get functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "regularizers.get", inputs=["a"])

    KERAS_LAYERS_MAPPING["regularizers.get"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "regularizers.get", inputs=[])
    KERAS_LAYERS_MAPPING["regularizers.get"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_regularizers_serialize() -> None:
    """Tests the keras layers regularizers.serialize functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "regularizers.serialize", inputs=["a"])

    KERAS_LAYERS_MAPPING["regularizers.serialize"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "regularizers.serialize", inputs=[])
    KERAS_LAYERS_MAPPING["regularizers.serialize"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_regularizers_l1() -> None:
    """Tests the keras layers regularizers.L1 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "regularizers.L1", inputs=["a"])

    KERAS_LAYERS_MAPPING["regularizers.L1"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "regularizers.L1", inputs=[])
    KERAS_LAYERS_MAPPING["regularizers.L1"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_regularizers_l1_ext() -> None:
    """Tests the keras layers regularizers.l1 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "regularizers.l1", inputs=["a"])

    KERAS_LAYERS_MAPPING["regularizers.l1"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "regularizers.l1", inputs=[])
    KERAS_LAYERS_MAPPING["regularizers.l1"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_regularizers_l1_l2() -> None:
    """Tests the keras layers regularizers.L1L2 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "regularizers.L1L2", inputs=["a"])

    KERAS_LAYERS_MAPPING["regularizers.L1L2"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "regularizers.L1L2", inputs=[])
    KERAS_LAYERS_MAPPING["regularizers.L1L2"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_regularizers_l1_l2_ext() -> None:
    """Tests the keras layers regularizers.l1_l2 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "regularizers.l1_l2", inputs=["a"])

    KERAS_LAYERS_MAPPING["regularizers.l1_l2"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "regularizers.l1_l2", inputs=[])
    KERAS_LAYERS_MAPPING["regularizers.l1_l2"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_regularizers_l2() -> None:
    """Tests the keras layers regularizers.L2 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "regularizers.L2", inputs=["a"])

    KERAS_LAYERS_MAPPING["regularizers.L2"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "regularizers.L2", inputs=[])
    KERAS_LAYERS_MAPPING["regularizers.L2"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_regularizers_l2_ext() -> None:
    """Tests the keras layers regularizers.l2 functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "regularizers.l2", inputs=["a"])

    KERAS_LAYERS_MAPPING["regularizers.l2"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "regularizers.l2", inputs=[])
    KERAS_LAYERS_MAPPING["regularizers.l2"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_regularizers__orthogonal_regularizer() -> None:
    """Tests the keras layers regularizers.OrthogonalRegularizer functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "regularizers.OrthogonalRegularizer", inputs=["a"])

    KERAS_LAYERS_MAPPING["regularizers.OrthogonalRegularizer"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "regularizers.OrthogonalRegularizer", inputs=[])
    KERAS_LAYERS_MAPPING["regularizers.OrthogonalRegularizer"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_regularizers_orthogonal_regularizer() -> None:
    """Tests the keras layers regularizers.orthogonal_regularizer functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "regularizers.orthogonal_regularizer", inputs=["a"])

    KERAS_LAYERS_MAPPING["regularizers.orthogonal_regularizer"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "regularizers.orthogonal_regularizer", inputs=[])
    KERAS_LAYERS_MAPPING["regularizers.orthogonal_regularizer"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_regularizers__regularizer() -> None:
    """Tests the keras layers regularizers.Regularizer functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "regularizers.Regularizer", inputs=["a"])

    KERAS_LAYERS_MAPPING["regularizers.Regularizer"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "regularizers.Regularizer", inputs=[])
    KERAS_LAYERS_MAPPING["regularizers.Regularizer"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_callbacks__backup_and_restore() -> None:
    """Tests the keras layers callbacks.BackupAndRestore functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "callbacks.BackupAndRestore", inputs=["a"])

    KERAS_LAYERS_MAPPING["callbacks.BackupAndRestore"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "callbacks.BackupAndRestore", inputs=[])
    KERAS_LAYERS_MAPPING["callbacks.BackupAndRestore"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_callbacks__callback() -> None:
    """Tests the keras layers callbacks.Callback functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "callbacks.Callback", inputs=["a"])

    KERAS_LAYERS_MAPPING["callbacks.Callback"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "callbacks.Callback", inputs=[])
    KERAS_LAYERS_MAPPING["callbacks.Callback"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_callbacks__callback_list() -> None:
    """Tests the keras layers callbacks.CallbackList functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "callbacks.CallbackList", inputs=["a"])

    KERAS_LAYERS_MAPPING["callbacks.CallbackList"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "callbacks.CallbackList", inputs=[])
    KERAS_LAYERS_MAPPING["callbacks.CallbackList"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_callbacks_csv_logger() -> None:
    """Tests the keras layers callbacks.CSVLogger functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "callbacks.CSVLogger", inputs=["a"])

    KERAS_LAYERS_MAPPING["callbacks.CSVLogger"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "callbacks.CSVLogger", inputs=[])
    KERAS_LAYERS_MAPPING["callbacks.CSVLogger"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_callbacks__early_stopping() -> None:
    """Tests the keras layers callbacks.EarlyStopping functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "callbacks.EarlyStopping", inputs=["a"])

    KERAS_LAYERS_MAPPING["callbacks.EarlyStopping"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "callbacks.EarlyStopping", inputs=[])
    KERAS_LAYERS_MAPPING["callbacks.EarlyStopping"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_callbacks__history() -> None:
    """Tests the keras layers callbacks.History functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "callbacks.History", inputs=["a"])

    KERAS_LAYERS_MAPPING["callbacks.History"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "callbacks.History", inputs=[])
    KERAS_LAYERS_MAPPING["callbacks.History"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_callbacks__lambda_callback() -> None:
    """Tests the keras layers callbacks.LambdaCallback functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "callbacks.LambdaCallback", inputs=["a"])

    KERAS_LAYERS_MAPPING["callbacks.LambdaCallback"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "callbacks.LambdaCallback", inputs=[])
    KERAS_LAYERS_MAPPING["callbacks.LambdaCallback"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_callbacks__learning_rate_scheduler() -> None:
    """Tests the keras layers callbacks.LearningRateScheduler functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "callbacks.LearningRateScheduler", inputs=["a"])

    KERAS_LAYERS_MAPPING["callbacks.LearningRateScheduler"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "callbacks.LearningRateScheduler", inputs=[])
    KERAS_LAYERS_MAPPING["callbacks.LearningRateScheduler"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_callbacks__model_checkpoint() -> None:
    """Tests the keras layers callbacks.ModelCheckpoint functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "callbacks.ModelCheckpoint", inputs=["a"])

    KERAS_LAYERS_MAPPING["callbacks.ModelCheckpoint"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "callbacks.ModelCheckpoint", inputs=[])
    KERAS_LAYERS_MAPPING["callbacks.ModelCheckpoint"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_callbacks__progbar_logger() -> None:
    """Tests the keras layers callbacks.ProgbarLogger functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "callbacks.ProgbarLogger", inputs=["a"])

    KERAS_LAYERS_MAPPING["callbacks.ProgbarLogger"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "callbacks.ProgbarLogger", inputs=[])
    KERAS_LAYERS_MAPPING["callbacks.ProgbarLogger"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_callbacks__reduce_lr_on_plateau() -> None:
    """Tests the keras layers callbacks.ReduceLROnPlateau functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "callbacks.ReduceLROnPlateau", inputs=["a"])

    KERAS_LAYERS_MAPPING["callbacks.ReduceLROnPlateau"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "callbacks.ReduceLROnPlateau", inputs=[])
    KERAS_LAYERS_MAPPING["callbacks.ReduceLROnPlateau"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_callbacks__remote_monitor() -> None:
    """Tests the keras layers callbacks.RemoteMonitor functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "callbacks.RemoteMonitor", inputs=["a"])

    KERAS_LAYERS_MAPPING["callbacks.RemoteMonitor"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "callbacks.RemoteMonitor", inputs=[])
    KERAS_LAYERS_MAPPING["callbacks.RemoteMonitor"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_callbacks__swap_ema_weights() -> None:
    """Tests the keras layers callbacks.SwapEMAWeights functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "callbacks.SwapEMAWeights", inputs=["a"])

    KERAS_LAYERS_MAPPING["callbacks.SwapEMAWeights"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "callbacks.SwapEMAWeights", inputs=[])
    KERAS_LAYERS_MAPPING["callbacks.SwapEMAWeights"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_callbacks__tensor_board() -> None:
    """Tests the keras layers callbacks.TensorBoard functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "callbacks.TensorBoard", inputs=["a"])

    KERAS_LAYERS_MAPPING["callbacks.TensorBoard"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "callbacks.TensorBoard", inputs=[])
    KERAS_LAYERS_MAPPING["callbacks.TensorBoard"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_callbacks__terminate_on_na_n() -> None:
    """Tests the keras layers callbacks.TerminateOnNaN functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "callbacks.TerminateOnNaN", inputs=["a"])

    KERAS_LAYERS_MAPPING["callbacks.TerminateOnNaN"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "callbacks.TerminateOnNaN", inputs=[])
    KERAS_LAYERS_MAPPING["callbacks.TerminateOnNaN"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_optimizers_deserialize() -> None:
    """Tests the keras layers optimizers.deserialize functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "optimizers.deserialize", inputs=["a"])

    KERAS_LAYERS_MAPPING["optimizers.deserialize"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "optimizers.deserialize", inputs=[])
    KERAS_LAYERS_MAPPING["optimizers.deserialize"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_optimizers_get() -> None:
    """Tests the keras layers optimizers.get functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "optimizers.get", inputs=["a"])

    KERAS_LAYERS_MAPPING["optimizers.get"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "optimizers.get", inputs=[])
    KERAS_LAYERS_MAPPING["optimizers.get"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_optimizers_serialize() -> None:
    """Tests the keras layers optimizers.serialize functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "optimizers.serialize", inputs=["a"])

    KERAS_LAYERS_MAPPING["optimizers.serialize"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "optimizers.serialize", inputs=[])
    KERAS_LAYERS_MAPPING["optimizers.serialize"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_optimizers__adadelta() -> None:
    """Tests the keras layers optimizers.Adadelta functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "optimizers.Adadelta", inputs=["a"])

    KERAS_LAYERS_MAPPING["optimizers.Adadelta"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "optimizers.Adadelta", inputs=[])
    KERAS_LAYERS_MAPPING["optimizers.Adadelta"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_optimizers__adafactor() -> None:
    """Tests the keras layers optimizers.Adafactor functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "optimizers.Adafactor", inputs=["a"])

    KERAS_LAYERS_MAPPING["optimizers.Adafactor"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "optimizers.Adafactor", inputs=[])
    KERAS_LAYERS_MAPPING["optimizers.Adafactor"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_optimizers__adagrad() -> None:
    """Tests the keras layers optimizers.Adagrad functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "optimizers.Adagrad", inputs=["a"])

    KERAS_LAYERS_MAPPING["optimizers.Adagrad"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "optimizers.Adagrad", inputs=[])
    KERAS_LAYERS_MAPPING["optimizers.Adagrad"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_optimizers__adam() -> None:
    """Tests the keras layers optimizers.Adam functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "optimizers.Adam", inputs=["a"])

    KERAS_LAYERS_MAPPING["optimizers.Adam"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "optimizers.Adam", inputs=[])
    KERAS_LAYERS_MAPPING["optimizers.Adam"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_optimizers__adamax() -> None:
    """Tests the keras layers optimizers.Adamax functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "optimizers.Adamax", inputs=["a"])

    KERAS_LAYERS_MAPPING["optimizers.Adamax"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "optimizers.Adamax", inputs=[])
    KERAS_LAYERS_MAPPING["optimizers.Adamax"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_optimizers__adam_w() -> None:
    """Tests the keras layers optimizers.AdamW functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "optimizers.AdamW", inputs=["a"])

    KERAS_LAYERS_MAPPING["optimizers.AdamW"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "optimizers.AdamW", inputs=[])
    KERAS_LAYERS_MAPPING["optimizers.AdamW"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_optimizers__ftrl() -> None:
    """Tests the keras layers optimizers.Ftrl functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "optimizers.Ftrl", inputs=["a"])

    KERAS_LAYERS_MAPPING["optimizers.Ftrl"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "optimizers.Ftrl", inputs=[])
    KERAS_LAYERS_MAPPING["optimizers.Ftrl"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_optimizers__lamb() -> None:
    """Tests the keras layers optimizers.Lamb functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "optimizers.Lamb", inputs=["a"])

    KERAS_LAYERS_MAPPING["optimizers.Lamb"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "optimizers.Lamb", inputs=[])
    KERAS_LAYERS_MAPPING["optimizers.Lamb"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_optimizers__lion() -> None:
    """Tests the keras layers optimizers.Lion functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "optimizers.Lion", inputs=["a"])

    KERAS_LAYERS_MAPPING["optimizers.Lion"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "optimizers.Lion", inputs=[])
    KERAS_LAYERS_MAPPING["optimizers.Lion"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_optimizers__loss_scale_optimizer() -> None:
    """Tests the keras layers optimizers.LossScaleOptimizer functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "optimizers.LossScaleOptimizer", inputs=["a"])

    KERAS_LAYERS_MAPPING["optimizers.LossScaleOptimizer"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "optimizers.LossScaleOptimizer", inputs=[])
    KERAS_LAYERS_MAPPING["optimizers.LossScaleOptimizer"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_optimizers__muon() -> None:
    """Tests the keras layers optimizers.Muon functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "optimizers.Muon", inputs=["a"])

    KERAS_LAYERS_MAPPING["optimizers.Muon"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "optimizers.Muon", inputs=[])
    KERAS_LAYERS_MAPPING["optimizers.Muon"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_optimizers__nadam() -> None:
    """Tests the keras layers optimizers.Nadam functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "optimizers.Nadam", inputs=["a"])

    KERAS_LAYERS_MAPPING["optimizers.Nadam"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "optimizers.Nadam", inputs=[])
    KERAS_LAYERS_MAPPING["optimizers.Nadam"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_optimizers__optimizer() -> None:
    """Tests the keras layers optimizers.Optimizer functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "optimizers.Optimizer", inputs=["a"])

    KERAS_LAYERS_MAPPING["optimizers.Optimizer"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "optimizers.Optimizer", inputs=[])
    KERAS_LAYERS_MAPPING["optimizers.Optimizer"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_optimizers_rm_sprop() -> None:
    """Tests the keras layers optimizers.RMSprop functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "optimizers.RMSprop", inputs=["a"])

    KERAS_LAYERS_MAPPING["optimizers.RMSprop"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "optimizers.RMSprop", inputs=[])
    KERAS_LAYERS_MAPPING["optimizers.RMSprop"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_optimizers_sgd() -> None:
    """Tests the keras layers optimizers.SGD functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "optimizers.SGD", inputs=["a"])

    KERAS_LAYERS_MAPPING["optimizers.SGD"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "optimizers.SGD", inputs=[])
    KERAS_LAYERS_MAPPING["optimizers.SGD"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_optimizers_schedules__cosine_decay() -> None:
    """Tests the keras layers optimizers.schedules.CosineDecay functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "optimizers.schedules.CosineDecay", inputs=["a"])

    KERAS_LAYERS_MAPPING["optimizers.schedules.CosineDecay"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "optimizers.schedules.CosineDecay", inputs=[])
    KERAS_LAYERS_MAPPING["optimizers.schedules.CosineDecay"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_optimizers_schedules__cosine_decay_restarts() -> None:
    """Tests the keras layers optimizers.schedules.CosineDecayRestarts functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "optimizers.schedules.CosineDecayRestarts", inputs=["a"])

    KERAS_LAYERS_MAPPING["optimizers.schedules.CosineDecayRestarts"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "optimizers.schedules.CosineDecayRestarts", inputs=[])
    KERAS_LAYERS_MAPPING["optimizers.schedules.CosineDecayRestarts"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_optimizers_schedules__exponential_decay() -> None:
    """Tests the keras layers optimizers.schedules.ExponentialDecay functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "optimizers.schedules.ExponentialDecay", inputs=["a"])

    KERAS_LAYERS_MAPPING["optimizers.schedules.ExponentialDecay"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "optimizers.schedules.ExponentialDecay", inputs=[])
    KERAS_LAYERS_MAPPING["optimizers.schedules.ExponentialDecay"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_optimizers_schedules__inverse_time_decay() -> None:
    """Tests the keras layers optimizers.schedules.InverseTimeDecay functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "optimizers.schedules.InverseTimeDecay", inputs=["a"])

    KERAS_LAYERS_MAPPING["optimizers.schedules.InverseTimeDecay"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "optimizers.schedules.InverseTimeDecay", inputs=[])
    KERAS_LAYERS_MAPPING["optimizers.schedules.InverseTimeDecay"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_optimizers_schedules__learning_rate_schedule() -> None:
    """Tests the keras layers optimizers.schedules.LearningRateSchedule functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "optimizers.schedules.LearningRateSchedule", inputs=["a"])

    KERAS_LAYERS_MAPPING["optimizers.schedules.LearningRateSchedule"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "optimizers.schedules.LearningRateSchedule", inputs=[])
    KERAS_LAYERS_MAPPING["optimizers.schedules.LearningRateSchedule"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_optimizers_schedules__piecewise_constant_decay() -> None:
    """Tests the keras layers optimizers.schedules.PiecewiseConstantDecay functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "optimizers.schedules.PiecewiseConstantDecay", inputs=["a"])

    KERAS_LAYERS_MAPPING["optimizers.schedules.PiecewiseConstantDecay"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "optimizers.schedules.PiecewiseConstantDecay", inputs=[])
    KERAS_LAYERS_MAPPING["optimizers.schedules.PiecewiseConstantDecay"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_optimizers_schedules__polynomial_decay() -> None:
    """Tests the keras layers optimizers.schedules.PolynomialDecay functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "optimizers.schedules.PolynomialDecay", inputs=["a"])

    KERAS_LAYERS_MAPPING["optimizers.schedules.PolynomialDecay"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "optimizers.schedules.PolynomialDecay", inputs=[])
    KERAS_LAYERS_MAPPING["optimizers.schedules.PolynomialDecay"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_optimizers_schedules_deserialize() -> None:
    """Tests the keras layers optimizers.schedules.deserialize functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "optimizers.schedules.deserialize", inputs=["a"])

    KERAS_LAYERS_MAPPING["optimizers.schedules.deserialize"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "optimizers.schedules.deserialize", inputs=[])
    KERAS_LAYERS_MAPPING["optimizers.schedules.deserialize"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_optimizers_schedules_serialize() -> None:
    """Tests the keras layers optimizers.schedules.serialize functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "optimizers.schedules.serialize", inputs=["a"])

    KERAS_LAYERS_MAPPING["optimizers.schedules.serialize"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "optimizers.schedules.serialize", inputs=[])
    KERAS_LAYERS_MAPPING["optimizers.schedules.serialize"](builder, node_empty)
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
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.Conv1DTranspose", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Conv1DTranspose"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__convolution1_d_transpose() -> None:
    """Tests the keras layers layers.Convolution1DTranspose functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Convolution1DTranspose", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Convolution1DTranspose"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

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
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.Conv2DTranspose", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Conv2DTranspose"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__convolution2_d_transpose() -> None:
    """Tests the keras layers layers.Convolution2DTranspose functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Convolution2DTranspose", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Convolution2DTranspose"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

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
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.Conv3DTranspose", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Conv3DTranspose"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__convolution3_d_transpose() -> None:
    """Tests the keras layers layers.Convolution3DTranspose functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Convolution3DTranspose", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Convolution3DTranspose"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.Convolution3DTranspose", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Convolution3DTranspose"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__depthwise_conv1_d() -> None:
    """Tests the keras layers layers.DepthwiseConv1D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.DepthwiseConv1D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.DepthwiseConv1D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.DepthwiseConv1D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.DepthwiseConv1D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__depthwise_conv2_d() -> None:
    """Tests the keras layers layers.DepthwiseConv2D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.DepthwiseConv2D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.DepthwiseConv2D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

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
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.EinsumDense", inputs=[])
    KERAS_LAYERS_MAPPING["layers.EinsumDense"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__embedding() -> None:
    """Tests the keras layers layers.Embedding functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Embedding", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Embedding"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

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
    assert builder.graph.nodes[-1].op_type == "Identity"

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
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.IntegerLookup", inputs=[])
    KERAS_LAYERS_MAPPING["layers.IntegerLookup"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__mel_spectrogram() -> None:
    """Tests the keras layers layers.MelSpectrogram functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.MelSpectrogram", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.MelSpectrogram"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

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
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.STFTSpectrogram", inputs=[])
    KERAS_LAYERS_MAPPING["layers.STFTSpectrogram"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__string_lookup() -> None:
    """Tests the keras layers layers.StringLookup functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.StringLookup", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.StringLookup"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

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
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.Cropping1D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Cropping1D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__cropping2_d() -> None:
    """Tests the keras layers layers.Cropping2D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Cropping2D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Cropping2D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.Cropping2D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.Cropping2D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__cropping3_d() -> None:
    """Tests the keras layers layers.Cropping3D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Cropping3D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Cropping3D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

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
    assert builder.graph.nodes[-1].op_type == "Identity"

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
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.UpSampling1D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.UpSampling1D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__up_sampling2_d() -> None:
    """Tests the keras layers layers.UpSampling2D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.UpSampling2D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.UpSampling2D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.UpSampling2D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.UpSampling2D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__up_sampling3_d() -> None:
    """Tests the keras layers layers.UpSampling3D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.UpSampling3D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.UpSampling3D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.UpSampling3D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.UpSampling3D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__zero_padding1_d() -> None:
    """Tests the keras layers layers.ZeroPadding1D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.ZeroPadding1D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.ZeroPadding1D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.ZeroPadding1D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.ZeroPadding1D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__zero_padding2_d() -> None:
    """Tests the keras layers layers.ZeroPadding2D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.ZeroPadding2D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.ZeroPadding2D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.ZeroPadding2D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.ZeroPadding2D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__zero_padding3_d() -> None:
    """Tests the keras layers layers.ZeroPadding3D functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.ZeroPadding3D", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.ZeroPadding3D"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.ZeroPadding3D", inputs=[])
    KERAS_LAYERS_MAPPING["layers.ZeroPadding3D"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__bidirectional() -> None:
    """Tests the keras layers layers.Bidirectional functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.Bidirectional", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.Bidirectional"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

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
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.SimpleRNN", inputs=[])
    KERAS_LAYERS_MAPPING["layers.SimpleRNN"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__simple_rnn_cell() -> None:
    """Tests the keras layers layers.SimpleRNNCell functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.SimpleRNNCell", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.SimpleRNNCell"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "layers.SimpleRNNCell", inputs=[])
    KERAS_LAYERS_MAPPING["layers.SimpleRNNCell"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_layers__stacked_rnn_cells() -> None:
    """Tests the keras layers layers.StackedRNNCells functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "layers.StackedRNNCells", inputs=["a"])

    KERAS_LAYERS_MAPPING["layers.StackedRNNCells"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

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


def test_keras_layers_missing_tree_map_to_none() -> None:
    """Tests the keras layers tree.MAP_TO_NONE functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "tree.MAP_TO_NONE", inputs=["a"])

    KERAS_LAYERS_MAPPING["tree.MAP_TO_NONE"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "tree.MAP_TO_NONE", inputs=[])
    KERAS_LAYERS_MAPPING["tree.MAP_TO_NONE"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_tree_assert_same_paths() -> None:
    """Tests the keras layers tree.assert_same_paths functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "tree.assert_same_paths", inputs=["a"])

    KERAS_LAYERS_MAPPING["tree.assert_same_paths"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "tree.assert_same_paths", inputs=[])
    KERAS_LAYERS_MAPPING["tree.assert_same_paths"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_tree_assert_same_structure() -> None:
    """Tests the keras layers tree.assert_same_structure functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "tree.assert_same_structure", inputs=["a"])

    KERAS_LAYERS_MAPPING["tree.assert_same_structure"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "tree.assert_same_structure", inputs=[])
    KERAS_LAYERS_MAPPING["tree.assert_same_structure"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_tree_flatten() -> None:
    """Tests the keras layers tree.flatten functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "tree.flatten", inputs=["a"])

    KERAS_LAYERS_MAPPING["tree.flatten"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "tree.flatten", inputs=[])
    KERAS_LAYERS_MAPPING["tree.flatten"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_tree_flatten_with_path() -> None:
    """Tests the keras layers tree.flatten_with_path functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "tree.flatten_with_path", inputs=["a"])

    KERAS_LAYERS_MAPPING["tree.flatten_with_path"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "tree.flatten_with_path", inputs=[])
    KERAS_LAYERS_MAPPING["tree.flatten_with_path"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_tree_is_nested() -> None:
    """Tests the keras layers tree.is_nested functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "tree.is_nested", inputs=["a"])

    KERAS_LAYERS_MAPPING["tree.is_nested"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "tree.is_nested", inputs=[])
    KERAS_LAYERS_MAPPING["tree.is_nested"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_tree_lists_to_tuples() -> None:
    """Tests the keras layers tree.lists_to_tuples functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "tree.lists_to_tuples", inputs=["a"])

    KERAS_LAYERS_MAPPING["tree.lists_to_tuples"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "tree.lists_to_tuples", inputs=[])
    KERAS_LAYERS_MAPPING["tree.lists_to_tuples"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_tree_map_shape_structure() -> None:
    """Tests the keras layers tree.map_shape_structure functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "tree.map_shape_structure", inputs=["a"])

    KERAS_LAYERS_MAPPING["tree.map_shape_structure"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "tree.map_shape_structure", inputs=[])
    KERAS_LAYERS_MAPPING["tree.map_shape_structure"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_tree_map_structure() -> None:
    """Tests the keras layers tree.map_structure functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "tree.map_structure", inputs=["a"])

    KERAS_LAYERS_MAPPING["tree.map_structure"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "tree.map_structure", inputs=[])
    KERAS_LAYERS_MAPPING["tree.map_structure"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_tree_map_structure_up_to() -> None:
    """Tests the keras layers tree.map_structure_up_to functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "tree.map_structure_up_to", inputs=["a"])

    KERAS_LAYERS_MAPPING["tree.map_structure_up_to"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "tree.map_structure_up_to", inputs=[])
    KERAS_LAYERS_MAPPING["tree.map_structure_up_to"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_tree_pack_sequence_as() -> None:
    """Tests the keras layers tree.pack_sequence_as functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "tree.pack_sequence_as", inputs=["a"])

    KERAS_LAYERS_MAPPING["tree.pack_sequence_as"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "tree.pack_sequence_as", inputs=[])
    KERAS_LAYERS_MAPPING["tree.pack_sequence_as"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"


def test_keras_layers_missing_tree_traverse() -> None:
    """Tests the keras layers tree.traverse functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "tree.traverse", inputs=["a"])

    KERAS_LAYERS_MAPPING["tree.traverse"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"

    node_empty = TFNode("n2", "tree.traverse", inputs=[])
    KERAS_LAYERS_MAPPING["tree.traverse"](builder, node_empty)
    assert builder.graph.nodes[-1].op_type == "Constant"
