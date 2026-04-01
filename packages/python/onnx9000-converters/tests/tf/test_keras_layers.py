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


def test_keras_semantic_generated_missing() -> None:
    """Tests all auto generated missing keras APIs semantically."""
    builder = TFToONNXGraphBuilder()
    names = [
        "config.backend",
        "backend.result_type",
        "backend.clear_session",
        "backend.is_keras_tensor",
        "backend.is_float_dtype",
        "backend.is_int_dtype",
        "backend.standardize_dtype",
        "backend.backend",
        "backend.epsilon",
        "backend.floatx",
        "backend.image_data_format",
        "backend.set_epsilon",
        "backend.set_floatx",
        "backend.set_image_data_format",
        "backend.get_uid",
        "ops.associative_scan",
        "ops.cast",
        "ops.cond",
        "ops.convert_to_numpy",
        "ops.convert_to_tensor",
        "ops.custom_gradient",
        "ops.dtype",
        "ops.fori_loop",
        "ops.is_tensor",
        "ops.map",
        "ops.saturate_cast",
        "ops.scan",
        "ops.scatter",
        "ops.scatter_update",
        "ops.shape",
        "ops.slice",
        "ops.slice_update",
        "ops.stop_gradient",
        "ops.switch",
        "ops.unstack",
        "ops.vectorized_map",
        "ops.while_loop",
        "ops.rearrange",
        "ops.cholesky",
        "ops.cholesky_inverse",
        "ops.det",
        "ops.eig",
        "ops.eigh",
        "ops.inv",
        "ops.jvp",
        "ops.lstsq",
        "ops.lu_factor",
        "ops.norm",
        "ops.qr",
        "ops.solve",
        "ops.solve_triangular",
        "ops.svd",
        "ops.erf",
        "ops.erfinv",
        "ops.extract_sequences",
        "ops.fft",
        "ops.fft2",
        "ops.ifft2",
        "ops.in_top_k",
        "ops.irfft",
        "ops.istft",
        "ops.logdet",
        "ops.logsumexp",
        "ops.rfft",
        "ops.rsqrt",
        "ops.segment_max",
        "ops.segment_sum",
        "ops.stft",
        "ops.top_k",
        "ops.view_as_complex",
        "ops.view_as_real",
        "ops.adaptive_average_pool",
        "ops.adaptive_max_pool",
        "ops.average_pool",
        "ops.batch_normalization",
        "ops.binary_crossentropy",
        "ops.categorical_crossentropy",
        "ops.celu",
        "ops.conv",
        "ops.conv_transpose",
        "ops.ctc_decode",
        "ops.ctc_loss",
        "ops.depthwise_conv",
        "ops.dot_product_attention",
        "ops.elu",
        "ops.gelu",
        "ops.glu",
        "ops.hard_shrink",
        "ops.hard_sigmoid",
        "ops.hard_silu",
        "ops.hard_swish",
        "ops.hard_tanh",
        "ops.layer_normalization",
        "ops.leaky_relu",
        "ops.log_sigmoid",
        "ops.log_softmax",
        "ops.max_pool",
        "ops.moments",
        "ops.multi_hot",
        "ops.normalize",
        "ops.one_hot",
        "ops.polar",
        "ops.psnr",
        "ops.relu",
        "ops.relu6",
        "ops.rms_normalization",
        "ops.selu",
        "ops.separable_conv",
        "ops.sigmoid",
        "ops.silu",
        "ops.swish",
        "ops.soft_shrink",
        "ops.softmax",
        "ops.softplus",
        "ops.softsign",
        "ops.sparse_categorical_crossentropy",
        "ops.sparse_plus",
        "ops.sparse_sigmoid",
        "ops.sparsemax",
        "ops.squareplus",
        "ops.tanh_shrink",
        "ops.threshold",
        "ops.unfold",
        "ops.abs",
        "ops.absolute",
        "ops.add",
        "ops.all",
        "ops.amax",
        "ops.amin",
        "ops.angle",
        "ops.any",
        "ops.append",
        "ops.arange",
        "ops.arccos",
        "ops.arccosh",
        "ops.arcsin",
        "ops.arcsinh",
        "ops.arctan",
        "ops.arctan2",
        "ops.arctanh",
        "ops.argmax",
        "ops.argmin",
        "ops.argpartition",
        "ops.argsort",
        "ops.array",
        "ops.array_split",
        "ops.average",
        "ops.bartlett",
        "ops.bincount",
        "ops.bitwise_and",
        "ops.bitwise_invert",
        "ops.bitwise_left_shift",
        "ops.bitwise_not",
        "ops.bitwise_or",
        "ops.bitwise_right_shift",
        "ops.bitwise_xor",
        "ops.blackman",
        "ops.broadcast_to",
        "ops.cbrt",
        "ops.ceil",
        "ops.clip",
        "ops.concatenate",
        "ops.conj",
        "ops.conjugate",
        "ops.copy",
        "ops.corrcoef",
        "ops.correlate",
        "ops.cos",
        "ops.cosh",
        "ops.count_nonzero",
        "ops.cross",
        "ops.cumprod",
        "ops.cumsum",
        "ops.deg2rad",
        "ops.diag",
        "ops.diagflat",
        "ops.diagonal",
        "ops.diff",
        "ops.digitize",
        "ops.divide",
        "ops.divide_no_nan",
        "ops.dot",
        "ops.einsum",
        "ops.empty",
        "ops.empty_like",
        "ops.equal",
        "ops.exp",
        "ops.exp2",
        "ops.expand_dims",
        "ops.expm1",
        "ops.eye",
        "ops.flip",
        "ops.floor",
        "ops.floor_divide",
        "ops.full",
        "ops.full_like",
        "ops.gcd",
        "ops.get_item",
        "ops.greater",
        "ops.greater_equal",
        "ops.hamming",
        "ops.hanning",
        "ops.heaviside",
        "ops.histogram",
        "ops.hstack",
        "ops.hypot",
        "ops.identity",
        "ops.imag",
        "ops.inner",
        "ops.isclose",
        "ops.isfinite",
        "ops.isin",
        "ops.isinf",
        "ops.isnan",
        "ops.isneginf",
        "ops.isposinf",
        "ops.isreal",
        "ops.kaiser",
        "ops.kron",
        "ops.lcm",
        "ops.ldexp",
        "ops.left_shift",
        "ops.less",
        "ops.less_equal",
        "ops.linspace",
        "ops.log",
        "ops.log1p",
        "ops.log2",
        "ops.log10",
        "ops.logaddexp",
        "ops.logaddexp2",
        "ops.logical_and",
        "ops.logical_not",
        "ops.logical_or",
        "ops.logical_xor",
        "ops.logspace",
        "ops.matmul",
        "ops.max",
        "ops.maximum",
        "ops.mean",
        "ops.median",
        "ops.meshgrid",
        "ops.min",
        "ops.minimum",
        "ops.mod",
        "ops.moveaxis",
        "ops.multiply",
        "ops.nan_to_num",
        "ops.ndim",
        "ops.negative",
        "ops.nonzero",
        "ops.not_equal",
        "ops.ones",
        "ops.ones_like",
        "ops.outer",
        "ops.pad",
        "ops.power",
        "ops.prod",
        "ops.quantile",
        "ops.ravel",
        "ops.real",
        "ops.reciprocal",
        "ops.repeat",
        "ops.reshape",
        "ops.right_shift",
        "ops.roll",
        "ops.rot90",
        "ops.round",
        "ops.searchsorted",
        "ops.select",
        "ops.sign",
        "ops.signbit",
        "ops.sin",
        "ops.sinh",
        "ops.size",
        "ops.slogdet",
        "ops.sort",
        "ops.split",
        "ops.sqrt",
        "ops.square",
        "ops.squeeze",
        "ops.stack",
        "ops.std",
        "ops.subtract",
        "ops.sum",
        "ops.swapaxes",
        "ops.take",
        "ops.take_along_axis",
        "ops.tan",
        "ops.tanh",
        "ops.tensordot",
        "ops.tile",
        "ops.trace",
        "ops.transpose",
        "ops.trapezoid",
        "ops.tri",
        "ops.tril",
        "ops.triu",
        "ops.true_divide",
        "ops.trunc",
        "ops.unravel_index",
        "ops.vander",
        "ops.var",
        "ops.vdot",
        "ops.vectorize",
        "ops.view",
        "ops.vstack",
        "ops.where",
        "ops.zeros",
        "ops.zeros_like",
        "ops.linalg.cholesky",
        "ops.linalg.cholesky_inverse",
        "ops.linalg.det",
        "ops.linalg.eig",
        "ops.linalg.eigh",
        "ops.linalg.inv",
        "ops.linalg.jvp",
        "ops.linalg.lstsq",
        "ops.linalg.lu_factor",
        "ops.linalg.norm",
        "ops.linalg.qr",
        "ops.linalg.solve",
        "ops.linalg.solve_triangular",
        "ops.linalg.svd",
        "ops.nn.adaptive_average_pool",
        "ops.nn.adaptive_max_pool",
        "ops.nn.average_pool",
        "ops.nn.batch_normalization",
        "ops.nn.binary_crossentropy",
        "ops.nn.categorical_crossentropy",
        "ops.nn.celu",
        "ops.nn.conv",
        "ops.nn.conv_transpose",
        "ops.nn.ctc_decode",
        "ops.nn.ctc_loss",
        "ops.nn.depthwise_conv",
        "ops.nn.dot_product_attention",
        "ops.nn.elu",
        "ops.nn.gelu",
        "ops.nn.glu",
        "ops.nn.hard_shrink",
        "ops.nn.hard_sigmoid",
        "ops.nn.hard_silu",
        "ops.nn.hard_swish",
        "ops.nn.hard_tanh",
        "ops.nn.layer_normalization",
        "ops.nn.leaky_relu",
        "ops.nn.log_sigmoid",
        "ops.nn.log_softmax",
        "ops.nn.max_pool",
        "ops.nn.moments",
        "ops.nn.multi_hot",
        "ops.nn.normalize",
        "ops.nn.one_hot",
        "ops.nn.polar",
        "ops.nn.psnr",
        "ops.nn.relu",
        "ops.nn.relu6",
        "ops.nn.rms_normalization",
        "ops.nn.selu",
        "ops.nn.separable_conv",
        "ops.nn.sigmoid",
        "ops.nn.silu",
        "ops.nn.swish",
        "ops.nn.soft_shrink",
        "ops.nn.softmax",
        "ops.nn.softplus",
        "ops.nn.softsign",
        "ops.nn.sparse_categorical_crossentropy",
        "ops.nn.sparse_plus",
        "ops.nn.sparse_sigmoid",
        "ops.nn.sparsemax",
        "ops.nn.squareplus",
        "ops.nn.tanh_shrink",
        "ops.nn.threshold",
        "ops.nn.unfold",
        "ops.numpy.abs",
        "ops.numpy.absolute",
        "ops.numpy.add",
        "ops.numpy.all",
        "ops.numpy.amax",
        "ops.numpy.amin",
        "ops.numpy.angle",
        "ops.numpy.any",
        "ops.numpy.append",
        "ops.numpy.arange",
        "ops.numpy.arccos",
        "ops.numpy.arccosh",
        "ops.numpy.arcsin",
        "ops.numpy.arcsinh",
        "ops.numpy.arctan",
        "ops.numpy.arctan2",
        "ops.numpy.arctanh",
        "ops.numpy.argmax",
        "ops.numpy.argmin",
        "ops.numpy.argpartition",
        "ops.numpy.argsort",
        "ops.numpy.array",
        "ops.numpy.array_split",
        "ops.numpy.average",
        "ops.numpy.bartlett",
        "ops.numpy.bincount",
        "ops.numpy.bitwise_and",
        "ops.numpy.bitwise_invert",
        "ops.numpy.bitwise_left_shift",
        "ops.numpy.bitwise_not",
        "ops.numpy.bitwise_or",
        "ops.numpy.bitwise_right_shift",
        "ops.numpy.bitwise_xor",
        "ops.numpy.blackman",
        "ops.numpy.broadcast_to",
        "ops.numpy.cbrt",
        "ops.numpy.ceil",
        "ops.numpy.clip",
        "ops.numpy.concatenate",
        "ops.numpy.conj",
        "ops.numpy.conjugate",
        "ops.numpy.copy",
        "ops.numpy.corrcoef",
        "ops.numpy.correlate",
        "ops.numpy.cos",
        "ops.numpy.cosh",
        "ops.numpy.count_nonzero",
        "ops.numpy.cross",
        "ops.numpy.cumprod",
        "ops.numpy.cumsum",
        "ops.numpy.deg2rad",
        "ops.numpy.diag",
        "ops.numpy.diagflat",
        "ops.numpy.diagonal",
        "ops.numpy.diff",
        "ops.numpy.digitize",
        "ops.numpy.divide",
        "ops.numpy.divide_no_nan",
        "ops.numpy.dot",
        "ops.numpy.einsum",
        "ops.numpy.empty",
        "ops.numpy.empty_like",
        "ops.numpy.equal",
        "ops.numpy.exp",
        "ops.numpy.exp2",
        "ops.numpy.expand_dims",
        "ops.numpy.expm1",
        "ops.numpy.eye",
        "ops.numpy.flip",
        "ops.numpy.floor",
        "ops.numpy.floor_divide",
        "ops.numpy.full",
        "ops.numpy.full_like",
        "ops.numpy.gcd",
        "ops.numpy.get_item",
        "ops.numpy.greater",
        "ops.numpy.greater_equal",
        "ops.numpy.hamming",
        "ops.numpy.hanning",
        "ops.numpy.heaviside",
        "ops.numpy.histogram",
        "ops.numpy.hstack",
        "ops.numpy.hypot",
        "ops.numpy.identity",
        "ops.numpy.imag",
        "ops.numpy.inner",
        "ops.numpy.isclose",
        "ops.numpy.isfinite",
        "ops.numpy.isin",
        "ops.numpy.isinf",
        "ops.numpy.isnan",
        "ops.numpy.isneginf",
        "ops.numpy.isposinf",
        "ops.numpy.isreal",
        "ops.numpy.kaiser",
        "ops.numpy.kron",
        "ops.numpy.lcm",
        "ops.numpy.ldexp",
        "ops.numpy.left_shift",
        "ops.numpy.less",
        "ops.numpy.less_equal",
        "ops.numpy.linspace",
        "ops.numpy.log",
        "ops.numpy.log1p",
        "ops.numpy.log2",
        "ops.numpy.log10",
        "ops.numpy.logaddexp",
        "ops.numpy.logaddexp2",
        "ops.numpy.logical_and",
        "ops.numpy.logical_not",
        "ops.numpy.logical_or",
        "ops.numpy.logical_xor",
        "ops.numpy.logspace",
        "ops.numpy.matmul",
        "ops.numpy.max",
        "ops.numpy.maximum",
        "ops.numpy.mean",
        "ops.numpy.median",
        "ops.numpy.meshgrid",
        "ops.numpy.min",
        "ops.numpy.minimum",
        "ops.numpy.mod",
        "ops.numpy.moveaxis",
        "ops.numpy.multiply",
        "ops.numpy.nan_to_num",
        "ops.numpy.ndim",
        "ops.numpy.negative",
        "ops.numpy.nonzero",
        "ops.numpy.not_equal",
        "ops.numpy.ones",
        "ops.numpy.ones_like",
        "ops.numpy.outer",
        "ops.numpy.pad",
        "ops.numpy.power",
        "ops.numpy.prod",
        "ops.numpy.quantile",
        "ops.numpy.ravel",
        "ops.numpy.real",
        "ops.numpy.reciprocal",
        "ops.numpy.repeat",
        "ops.numpy.reshape",
        "ops.numpy.right_shift",
        "ops.numpy.roll",
        "ops.numpy.rot90",
        "ops.numpy.round",
        "ops.numpy.searchsorted",
        "ops.numpy.select",
        "ops.numpy.sign",
        "ops.numpy.signbit",
        "ops.numpy.sin",
        "ops.numpy.sinh",
        "ops.numpy.size",
        "ops.numpy.slogdet",
        "ops.numpy.sort",
        "ops.numpy.split",
        "ops.numpy.sqrt",
        "ops.numpy.square",
        "ops.numpy.squeeze",
        "ops.numpy.stack",
        "ops.numpy.std",
        "ops.numpy.subtract",
        "ops.numpy.sum",
        "ops.numpy.swapaxes",
        "ops.numpy.take",
        "ops.numpy.take_along_axis",
        "ops.numpy.tan",
        "ops.numpy.tanh",
        "ops.numpy.tensordot",
        "ops.numpy.tile",
        "ops.numpy.trace",
        "ops.numpy.transpose",
        "ops.numpy.trapezoid",
        "ops.numpy.tri",
        "ops.numpy.tril",
        "ops.numpy.triu",
        "ops.numpy.true_divide",
        "ops.numpy.trunc",
        "ops.numpy.unravel_index",
        "ops.numpy.vander",
        "ops.numpy.var",
        "ops.numpy.vdot",
        "ops.numpy.vectorize",
        "ops.numpy.view",
        "ops.numpy.vstack",
        "ops.numpy.where",
        "ops.numpy.zeros",
        "ops.numpy.zeros_like",
        "ops.image.affine_transform",
        "ops.image.crop_images",
        "ops.image.elastic_transform",
        "ops.image.extract_patches",
        "ops.image.extract_patches_3d",
        "ops.image.gaussian_blur",
        "ops.image.hsv_to_rgb",
        "ops.image.map_coordinates",
        "ops.image.pad_images",
        "ops.image.perspective_transform",
        "ops.image.resize",
        "ops.image.rgb_to_grayscale",
        "ops.image.rgb_to_hsv",
        "ops.image.scale_and_translate",
        "src.KerasTensor",
        "src.Input",
        "src.Layer",
        "src.Functional",
        "src.Model",
        "src.Sequential",
        "src.version.keras_export",
        "src.version.version",
        "src.api_export.REGISTERED_NAMES_TO_OBJS",
        "src.api_export.REGISTERED_OBJS_TO_NAMES",
        "src.api_export.register_internal_serializable",
        "src.api_export.get_symbol_from_name",
        "src.api_export.get_name_from_symbol",
        "src.api_export.keras_export",
        "src.activations.celu",
        "src.activations.elu",
        "src.activations.exponential",
        "src.activations.gelu",
        "src.activations.glu",
        "src.activations.hard_shrink",
        "src.activations.hard_sigmoid",
        "src.activations.hard_silu",
        "src.activations.hard_tanh",
        "src.activations.leaky_relu",
        "src.activations.linear",
        "src.activations.log_sigmoid",
        "src.activations.log_softmax",
        "src.activations.mish",
        "src.activations.relu",
        "src.activations.relu6",
        "src.activations.selu",
        "src.activations.sigmoid",
        "src.activations.silu",
        "src.activations.soft_shrink",
        "src.activations.softmax",
        "src.activations.softplus",
        "src.activations.softsign",
        "src.activations.sparse_plus",
        "src.activations.sparse_sigmoid",
        "src.activations.sparsemax",
        "src.activations.squareplus",
        "src.activations.tanh",
        "src.activations.tanh_shrink",
        "src.activations.threshold",
        "src.activations.keras_export",
        "src.activations.object_registration",
        "src.activations.serialization_lib",
        "src.activations.ALL_OBJECTS",
        "src.activations.ALL_OBJECTS_DICT",
        "src.activations.serialize",
        "src.activations.deserialize",
        "src.activations.get",
        "src.activations.activations.backend",
        "src.activations.activations.ops",
        "src.activations.activations.keras_export",
        "src.activations.activations.relu",
        "src.activations.activations.ReLU",
        "src.activations.activations.leaky_relu",
        "src.activations.activations.relu6",
        "src.activations.activations.softmax",
        "src.activations.activations.elu",
        "src.activations.activations.selu",
        "src.activations.activations.softplus",
        "src.activations.activations.softsign",
        "src.activations.activations.soft_shrink",
        "src.activations.activations.sparse_plus",
        "src.activations.activations.silu",
        "src.activations.activations.squareplus",
        "src.activations.activations.gelu",
        "src.activations.activations.celu",
        "src.activations.activations.glu",
        "src.activations.activations.tanh",
        "src.activations.activations.tanh_shrink",
        "src.activations.activations.hard_tanh",
        "src.activations.activations.hard_shrink",
        "src.activations.activations.threshold",
        "src.activations.activations.sigmoid",
        "src.activations.activations.exponential",
        "src.activations.activations.hard_sigmoid",
        "src.activations.activations.log_sigmoid",
        "src.activations.activations.sparse_sigmoid",
        "src.activations.activations.hard_silu",
        "src.activations.activations.linear",
        "src.activations.activations.Mish",
        "src.activations.activations.mish",
        "src.activations.activations.log_softmax",
        "src.activations.activations.sparsemax",
        "src.visualization.plot_bounding_box_gallery.backend",
        "src.visualization.plot_bounding_box_gallery.ops",
        "src.visualization.plot_bounding_box_gallery.keras_export",
        "src.visualization.plot_bounding_box_gallery.draw_bounding_boxes",
        "src.visualization.plot_bounding_box_gallery.plot_image_gallery",
        "src.visualization.plot_bounding_box_gallery.plot_bounding_box_gallery",
        "src.visualization.draw_bounding_boxes.backend",
        "src.visualization.draw_bounding_boxes.ops",
        "src.visualization.draw_bounding_boxes.keras_export",
        "src.visualization.draw_bounding_boxes.convert_format",
        "src.visualization.draw_bounding_boxes.draw_bounding_boxes",
        "src.visualization.draw_segmentation_masks.backend",
        "src.visualization.draw_segmentation_masks.ops",
        "src.visualization.draw_segmentation_masks.keras_export",
        "src.visualization.draw_segmentation_masks.draw_segmentation_masks",
        "src.visualization.plot_segmentation_mask_gallery.backend",
        "src.visualization.plot_segmentation_mask_gallery.ops",
        "src.visualization.plot_segmentation_mask_gallery.keras_export",
        "src.visualization.plot_segmentation_mask_gallery.draw_segmentation_masks",
        "src.visualization.plot_segmentation_mask_gallery.plot_image_gallery",
        "src.visualization.plot_segmentation_mask_gallery.plot_segmentation_mask_gallery",
        "src.visualization.plot_image_gallery.backend",
        "src.visualization.plot_image_gallery.ops",
        "src.visualization.plot_image_gallery.keras_export",
        "src.visualization.plot_image_gallery.BaseImagePreprocessingLayer",
        "src.visualization.plot_image_gallery.plot_image_gallery",
        "src.tree.assert_same_paths",
        "src.tree.assert_same_structure",
        "src.tree.flatten",
        "src.tree.flatten_with_path",
        "src.tree.is_nested",
        "src.tree.lists_to_tuples",
        "src.tree.map_shape_structure",
        "src.tree.map_structure",
        "src.tree.map_structure_up_to",
        "src.tree.pack_sequence_as",
        "src.tree.register_tree_node_class",
        "src.tree.traverse",
        "src.tree.dmtree_impl.backend",
        "src.tree.dmtree_impl.dmtree",
        "src.tree.dmtree_impl.REGISTERED_CLASSES",
        "src.tree.dmtree_impl.ClassRegistration",
        "src.tree.dmtree_impl.TypeErrorRemapping",
        "src.tree.dmtree_impl.register_tree_node",
        "src.tree.dmtree_impl.register_tree_node_class",
        "src.tree.dmtree_impl.sorted_keys_and_values",
        "src.tree.dmtree_impl.is_nested",
        "src.tree.dmtree_impl.traverse",
        "src.tree.dmtree_impl.flatten",
        "src.tree.dmtree_impl.flatten_with_path",
        "src.tree.dmtree_impl.map_structure",
        "src.tree.dmtree_impl.map_structure_up_to",
        "src.tree.dmtree_impl.assert_same_structure",
        "src.tree.dmtree_impl.assert_same_paths",
        "src.tree.dmtree_impl.pack_sequence_as",
        "src.tree.dmtree_impl.lists_to_tuples",
        "src.tree.dmtree_impl.map_shape_structure",
        "src.tree.torchtree_impl.register_tree_node_class",
        "src.tree.torchtree_impl.is_nested",
        "src.tree.torchtree_impl.traverse",
        "src.tree.torchtree_impl.flatten",
        "src.tree.torchtree_impl.flatten_with_path",
        "src.tree.torchtree_impl.map_structure",
        "src.tree.torchtree_impl.map_structure_up_to",
        "src.tree.torchtree_impl.assert_same_structure",
        "src.tree.torchtree_impl.assert_same_paths",
        "src.tree.torchtree_impl.pack_sequence_as",
        "src.tree.torchtree_impl.lists_to_tuples",
        "src.tree.torchtree_impl.map_shape_structure",
        "src.tree.optree_impl.backend",
        "src.tree.optree_impl.register_tree_node_class",
        "src.tree.optree_impl.sorted_keys_and_values",
        "src.tree.optree_impl.is_nested",
        "src.tree.optree_impl.traverse",
        "src.tree.optree_impl.flatten",
        "src.tree.optree_impl.flatten_with_path",
        "src.tree.optree_impl.map_structure",
        "src.tree.optree_impl.map_structure_up_to",
        "src.tree.optree_impl.assert_same_structure",
        "src.tree.optree_impl.assert_same_paths",
        "src.tree.optree_impl.pack_sequence_as",
        "src.tree.optree_impl.lists_to_tuples",
        "src.tree.optree_impl.map_shape_structure",
        "src.tree.tree_api.keras_export",
        "src.tree.tree_api.backend",
        "src.tree.tree_api.dmtree",
        "src.tree.tree_api.optree",
        "src.tree.tree_api.tree_impl",
        "src.tree.tree_api.register_tree_node_class",
        "src.tree.tree_api.MAP_TO_NONE",
        "src.tree.tree_api.is_nested",
        "src.tree.tree_api.traverse",
        "src.tree.tree_api.flatten",
        "src.tree.tree_api.flatten_with_path",
        "src.tree.tree_api.map_structure",
        "src.tree.tree_api.map_structure_up_to",
        "src.tree.tree_api.assert_same_structure",
        "src.tree.tree_api.assert_same_paths",
        "src.tree.tree_api.pack_sequence_as",
        "src.tree.tree_api.lists_to_tuples",
        "src.tree.tree_api.map_shape_structure",
        "src.metrics.keras_export",
        "src.metrics.Accuracy",
        "src.metrics.BinaryAccuracy",
        "src.metrics.CategoricalAccuracy",
        "src.metrics.SparseCategoricalAccuracy",
        "src.metrics.SparseTopKCategoricalAccuracy",
        "src.metrics.TopKCategoricalAccuracy",
        "src.metrics.AUC",
        "src.metrics.FalseNegatives",
        "src.metrics.FalsePositives",
        "src.metrics.Precision",
        "src.metrics.PrecisionAtRecall",
        "src.metrics.Recall",
        "src.metrics.RecallAtPrecision",
        "src.metrics.SensitivityAtSpecificity",
        "src.metrics.SpecificityAtSensitivity",
        "src.metrics.TrueNegatives",
        "src.metrics.TruePositives",
        "src.metrics.ConcordanceCorrelation",
        "src.metrics.PearsonCorrelation",
        "src.metrics.F1Score",
        "src.metrics.FBetaScore",
        "src.metrics.CategoricalHinge",
        "src.metrics.Hinge",
        "src.metrics.SquaredHinge",
        "src.metrics.BinaryIoU",
        "src.metrics.IoU",
        "src.metrics.MeanIoU",
        "src.metrics.OneHotIoU",
        "src.metrics.OneHotMeanIoU",
        "src.metrics.Metric",
        "src.metrics.BinaryCrossentropy",
        "src.metrics.CategoricalCrossentropy",
        "src.metrics.KLDivergence",
        "src.metrics.Poisson",
        "src.metrics.SparseCategoricalCrossentropy",
        "src.metrics.Mean",
        "src.metrics.MeanMetricWrapper",
        "src.metrics.Sum",
        "src.metrics.CosineSimilarity",
        "src.metrics.LogCoshError",
        "src.metrics.MeanAbsoluteError",
        "src.metrics.MeanAbsolutePercentageError",
        "src.metrics.MeanSquaredError",
        "src.metrics.MeanSquaredLogarithmicError",
        "src.metrics.R2Score",
        "src.metrics.RootMeanSquaredError",
        "src.metrics.serialization_lib",
        "src.metrics.to_snake_case",
        "src.metrics.ALL_OBJECTS",
        "src.metrics.ALL_OBJECTS_DICT",
        "src.metrics.serialize",
        "src.metrics.deserialize",
        "src.metrics.get",
        "src.metrics.hinge_metrics.keras_export",
        "src.metrics.hinge_metrics.categorical_hinge",
        "src.metrics.hinge_metrics.hinge",
        "src.metrics.hinge_metrics.squared_hinge",
        "src.metrics.hinge_metrics.reduction_metrics",
        "src.metrics.hinge_metrics.Hinge",
        "src.metrics.hinge_metrics.SquaredHinge",
        "src.metrics.hinge_metrics.CategoricalHinge",
        "src.metrics.reduction_metrics.backend",
        "src.metrics.reduction_metrics.initializers",
        "src.metrics.reduction_metrics.losses",
        "src.metrics.reduction_metrics.ops",
        "src.metrics.reduction_metrics.keras_export",
        "src.metrics.reduction_metrics.Metric",
        "src.metrics.reduction_metrics.serialization_lib",
        "src.metrics.reduction_metrics.reduce_to_samplewise_values",
        "src.metrics.reduction_metrics.Sum",
        "src.metrics.reduction_metrics.Mean",
        "src.metrics.reduction_metrics.MeanMetricWrapper",
        "src.metrics.correlation_metrics.backend",
        "src.metrics.correlation_metrics.ops",
        "src.metrics.correlation_metrics.keras_export",
        "src.metrics.correlation_metrics.squeeze_or_expand_to_same_rank",
        "src.metrics.correlation_metrics.reduction_metrics",
        "src.metrics.correlation_metrics.pearson_correlation",
        "src.metrics.correlation_metrics.concordance_correlation",
        "src.metrics.correlation_metrics.PearsonCorrelation",
        "src.metrics.correlation_metrics.ConcordanceCorrelation",
        "src.metrics.f_score_metrics.backend",
        "src.metrics.f_score_metrics.initializers",
        "src.metrics.f_score_metrics.ops",
        "src.metrics.f_score_metrics.keras_export",
        "src.metrics.f_score_metrics.Metric",
        "src.metrics.f_score_metrics.FBetaScore",
        "src.metrics.f_score_metrics.F1Score",
        "src.metrics.probabilistic_metrics.keras_export",
        "src.metrics.probabilistic_metrics.binary_crossentropy",
        "src.metrics.probabilistic_metrics.categorical_crossentropy",
        "src.metrics.probabilistic_metrics.kl_divergence",
        "src.metrics.probabilistic_metrics.poisson",
        "src.metrics.probabilistic_metrics.sparse_categorical_crossentropy",
        "src.metrics.probabilistic_metrics.reduction_metrics",
        "src.metrics.probabilistic_metrics.KLDivergence",
        "src.metrics.probabilistic_metrics.Poisson",
        "src.metrics.probabilistic_metrics.BinaryCrossentropy",
        "src.metrics.probabilistic_metrics.CategoricalCrossentropy",
        "src.metrics.probabilistic_metrics.SparseCategoricalCrossentropy",
        "src.metrics.confusion_metrics.activations",
        "src.metrics.confusion_metrics.backend",
        "src.metrics.confusion_metrics.initializers",
        "src.metrics.confusion_metrics.ops",
        "src.metrics.confusion_metrics.keras_export",
        "src.metrics.confusion_metrics.metrics_utils",
        "src.metrics.confusion_metrics.Metric",
        "src.metrics.confusion_metrics.to_list",
        "src.metrics.confusion_metrics.FalsePositives",
        "src.metrics.confusion_metrics.FalseNegatives",
        "src.metrics.confusion_metrics.TrueNegatives",
        "src.metrics.confusion_metrics.TruePositives",
        "src.metrics.confusion_metrics.Precision",
        "src.metrics.confusion_metrics.Recall",
        "src.metrics.confusion_metrics.SensitivitySpecificityBase",
        "src.metrics.confusion_metrics.SensitivityAtSpecificity",
        "src.metrics.confusion_metrics.SpecificityAtSensitivity",
        "src.metrics.confusion_metrics.PrecisionAtRecall",
        "src.metrics.confusion_metrics.RecallAtPrecision",
        "src.metrics.confusion_metrics.AUC",
        "src.metrics.regression_metrics.initializers",
        "src.metrics.regression_metrics.ops",
        "src.metrics.regression_metrics.keras_export",
        "src.metrics.regression_metrics.squeeze_or_expand_to_same_rank",
        "src.metrics.regression_metrics.log_cosh",
        "src.metrics.regression_metrics.mean_absolute_error",
        "src.metrics.regression_metrics.mean_absolute_percentage_error",
        "src.metrics.regression_metrics.mean_squared_error",
        "src.metrics.regression_metrics.mean_squared_logarithmic_error",
        "src.metrics.regression_metrics.reduction_metrics",
        "src.metrics.regression_metrics.normalize",
        "src.metrics.regression_metrics.MeanSquaredError",
        "src.metrics.regression_metrics.MeanAbsoluteError",
        "src.metrics.regression_metrics.MeanAbsolutePercentageError",
        "src.metrics.regression_metrics.MeanSquaredLogarithmicError",
        "src.metrics.regression_metrics.RootMeanSquaredError",
        "src.metrics.regression_metrics.CosineSimilarity",
        "src.metrics.regression_metrics.LogCoshError",
        "src.metrics.regression_metrics.R2Score",
        "src.metrics.regression_metrics.cosine_similarity",
        "src.metrics.metric.backend",
        "src.metrics.metric.dtype_policies",
        "src.metrics.metric.initializers",
        "src.metrics.metric.ops",
        "src.metrics.metric.keras_export",
        "src.metrics.metric.KerasSaveable",
        "src.metrics.metric.auto_name",
        "src.metrics.metric.Tracker",
        "src.metrics.metric.Metric",
        "src.metrics.metrics_utils.backend",
        "src.metrics.metrics_utils.ops",
        "src.metrics.metrics_utils.squeeze_or_expand_to_same_rank",
        "src.metrics.metrics_utils.to_list",
        "src.metrics.metrics_utils.NEG_INF",
        "src.metrics.metrics_utils.assert_thresholds_range",
        "src.metrics.metrics_utils.parse_init_thresholds",
        "src.metrics.metrics_utils.ConfusionMatrix",
        "src.metrics.metrics_utils.AUCCurve",
        "src.metrics.metrics_utils.AUCSummationMethod",
        "src.metrics.metrics_utils.is_evenly_distributed_thresholds",
        "src.metrics.metrics_utils.update_confusion_matrix_variables",
        "src.metrics.metrics_utils.confusion_matrix",
        "src.metrics.iou_metrics.backend",
        "src.metrics.iou_metrics.initializers",
        "src.metrics.iou_metrics.ops",
        "src.metrics.iou_metrics.keras_export",
        "src.metrics.iou_metrics.Metric",
        "src.metrics.iou_metrics.confusion_matrix",
        "src.metrics.iou_metrics.IoU",
        "src.metrics.iou_metrics.BinaryIoU",
        "src.metrics.iou_metrics.MeanIoU",
        "src.metrics.iou_metrics.OneHotIoU",
        "src.metrics.iou_metrics.OneHotMeanIoU",
        "src.metrics.accuracy_metrics.backend",
        "src.metrics.accuracy_metrics.ops",
        "src.metrics.accuracy_metrics.keras_export",
        "src.metrics.accuracy_metrics.squeeze_or_expand_to_same_rank",
        "src.metrics.accuracy_metrics.reduction_metrics",
        "src.metrics.accuracy_metrics.accuracy",
        "src.metrics.accuracy_metrics.Accuracy",
        "src.metrics.accuracy_metrics.binary_accuracy",
        "src.metrics.accuracy_metrics.BinaryAccuracy",
        "src.metrics.accuracy_metrics.categorical_accuracy",
        "src.metrics.accuracy_metrics.CategoricalAccuracy",
        "src.metrics.accuracy_metrics.sparse_categorical_accuracy",
        "src.metrics.accuracy_metrics.SparseCategoricalAccuracy",
        "src.metrics.accuracy_metrics.top_k_categorical_accuracy",
        "src.metrics.accuracy_metrics.TopKCategoricalAccuracy",
        "src.metrics.accuracy_metrics.sparse_top_k_categorical_accuracy",
        "src.metrics.accuracy_metrics.SparseTopKCategoricalAccuracy",
        "src.losses.keras_export",
        "src.losses.Loss",
        "src.losses.CTC",
        "src.losses.BinaryCrossentropy",
        "src.losses.BinaryFocalCrossentropy",
        "src.losses.CategoricalCrossentropy",
        "src.losses.CategoricalFocalCrossentropy",
        "src.losses.CategoricalHinge",
        "src.losses.Circle",
        "src.losses.CosineSimilarity",
        "src.losses.Dice",
        "src.losses.Hinge",
        "src.losses.Huber",
        "src.losses.KLDivergence",
        "src.losses.LogCosh",
        "src.losses.LossFunctionWrapper",
        "src.losses.MeanAbsoluteError",
        "src.losses.MeanAbsolutePercentageError",
        "src.losses.MeanSquaredError",
        "src.losses.MeanSquaredLogarithmicError",
        "src.losses.Poisson",
        "src.losses.SparseCategoricalCrossentropy",
        "src.losses.SquaredHinge",
        "src.losses.Tversky",
        "src.losses.binary_crossentropy",
        "src.losses.binary_focal_crossentropy",
        "src.losses.categorical_crossentropy",
        "src.losses.categorical_focal_crossentropy",
        "src.losses.categorical_hinge",
        "src.losses.circle",
        "src.losses.cosine_similarity",
        "src.losses.ctc",
        "src.losses.dice",
        "src.losses.hinge",
        "src.losses.huber",
        "src.losses.kl_divergence",
        "src.losses.log_cosh",
        "src.losses.mean_absolute_error",
        "src.losses.mean_absolute_percentage_error",
        "src.losses.mean_squared_error",
        "src.losses.mean_squared_logarithmic_error",
        "src.losses.poisson",
        "src.losses.sparse_categorical_crossentropy",
        "src.losses.squared_hinge",
        "src.losses.tversky",
        "src.losses.serialization_lib",
        "src.losses.ALL_OBJECTS",
        "src.losses.ALL_OBJECTS_DICT",
        "src.losses.serialize",
        "src.losses.deserialize",
        "src.losses.get",
        "src.losses.loss.backend",
        "src.losses.loss.dtype_policies",
        "src.losses.loss.ops",
        "src.losses.loss.tree",
        "src.losses.loss.keras_export",
        "src.losses.loss.KerasSaveable",
        "src.losses.loss.auto_name",
        "src.losses.loss.Loss",
        "src.losses.loss.standardize_reduction",
        "src.losses.loss.squeeze_or_expand_to_same_rank",
        "src.losses.loss.reduce_values",
        "src.losses.loss.reduce_weighted_values",
        "src.losses.loss.apply_mask",
        "src.losses.loss.scale_loss_for_distribution",
        "src.losses.loss.unscale_loss_for_distribution",
        "src.losses.losses.backend",
        "src.losses.losses.ops",
        "src.losses.losses.tree",
        "src.losses.losses.keras_export",
        "src.losses.losses.Loss",
        "src.losses.losses.squeeze_or_expand_to_same_rank",
        "src.losses.losses.serialization_lib",
        "src.losses.losses.build_pos_neg_masks",
        "src.losses.losses.normalize",
        "src.losses.losses.LossFunctionWrapper",
        "src.losses.losses.MeanSquaredError",
        "src.losses.losses.MeanAbsoluteError",
        "src.losses.losses.MeanAbsolutePercentageError",
        "src.losses.losses.MeanSquaredLogarithmicError",
        "src.losses.losses.CosineSimilarity",
        "src.losses.losses.Huber",
        "src.losses.losses.LogCosh",
        "src.losses.losses.Hinge",
        "src.losses.losses.SquaredHinge",
        "src.losses.losses.CategoricalHinge",
        "src.losses.losses.KLDivergence",
        "src.losses.losses.Poisson",
        "src.losses.losses.BinaryCrossentropy",
        "src.losses.losses.BinaryFocalCrossentropy",
        "src.losses.losses.CategoricalCrossentropy",
        "src.losses.losses.CategoricalFocalCrossentropy",
        "src.losses.losses.SparseCategoricalCrossentropy",
        "src.losses.losses.CTC",
        "src.losses.losses.Dice",
        "src.losses.losses.Tversky",
        "src.losses.losses.Circle",
        "src.losses.losses.CategoricalGeneralizedCrossEntropy",
        "src.losses.losses.convert_binary_labels_to_hinge",
        "src.losses.losses.hinge",
        "src.losses.losses.squared_hinge",
        "src.losses.losses.categorical_hinge",
        "src.losses.losses.mean_squared_error",
        "src.losses.losses.mean_absolute_error",
        "src.losses.losses.mean_absolute_percentage_error",
        "src.losses.losses.mean_squared_logarithmic_error",
        "src.losses.losses.cosine_similarity",
        "src.losses.losses.huber",
        "src.losses.losses.log_cosh",
        "src.losses.losses.kl_divergence",
        "src.losses.losses.poisson",
        "src.losses.losses.categorical_crossentropy",
        "src.losses.losses.categorical_focal_crossentropy",
        "src.losses.losses.sparse_categorical_crossentropy",
        "src.losses.losses.binary_crossentropy",
        "src.losses.losses.binary_focal_crossentropy",
        "src.losses.losses.ctc",
        "src.losses.losses.dice",
        "src.losses.losses.tversky",
        "src.losses.losses.circle",
        "src.losses.losses.categorical_generalized_cross_entropy",
        "src.wrappers.SKLearnClassifier",
        "src.wrappers.SKLearnRegressor",
        "src.wrappers.SKLearnTransformer",
        "src.wrappers.fixes.type_of_target",
        "src.wrappers.sklearn_wrapper.keras_export",
        "src.wrappers.sklearn_wrapper.clone_model",
        "src.wrappers.sklearn_wrapper.Model",
        "src.wrappers.sklearn_wrapper.type_of_target",
        "src.wrappers.sklearn_wrapper.TargetReshaper",
        "src.wrappers.sklearn_wrapper.assert_sklearn_installed",
        "src.wrappers.sklearn_wrapper.BaseEstimator",
        "src.wrappers.sklearn_wrapper.ClassifierMixin",
        "src.wrappers.sklearn_wrapper.RegressorMixin",
        "src.wrappers.sklearn_wrapper.TransformerMixin",
        "src.wrappers.sklearn_wrapper.SKLBase",
        "src.wrappers.sklearn_wrapper.SKLearnClassifier",
        "src.wrappers.sklearn_wrapper.SKLearnRegressor",
        "src.wrappers.sklearn_wrapper.SKLearnTransformer",
        "src.layers.keras_export",
        "src.layers.Activation",
        "src.layers.ELU",
        "src.layers.LeakyReLU",
        "src.layers.PReLU",
        "src.layers.ReLU",
        "src.layers.Softmax",
        "src.layers.AdditiveAttention",
        "src.layers.Attention",
        "src.layers.GroupedQueryAttention",
        "src.layers.MultiHeadAttention",
        "src.layers.Conv1D",
        "src.layers.Conv1DTranspose",
        "src.layers.Conv2D",
        "src.layers.Conv2DTranspose",
        "src.layers.Conv3D",
        "src.layers.Conv3DTranspose",
        "src.layers.DepthwiseConv1D",
        "src.layers.DepthwiseConv2D",
        "src.layers.SeparableConv1D",
        "src.layers.SeparableConv2D",
        "src.layers.Dense",
        "src.layers.EinsumDense",
        "src.layers.Embedding",
        "src.layers.Identity",
        "src.layers.Input",
        "src.layers.InputLayer",
        "src.layers.Lambda",
        "src.layers.Masking",
        "src.layers.ReversibleEmbedding",
        "src.layers.Wrapper",
        "src.layers.InputSpec",
        "src.layers.Layer",
        "src.layers.Add",
        "src.layers.add",
        "src.layers.Average",
        "src.layers.average",
        "src.layers.Concatenate",
        "src.layers.concatenate",
        "src.layers.Dot",
        "src.layers.dot",
        "src.layers.Maximum",
        "src.layers.maximum",
        "src.layers.Minimum",
        "src.layers.minimum",
        "src.layers.Multiply",
        "src.layers.multiply",
        "src.layers.Subtract",
        "src.layers.subtract",
        "src.layers.BatchNormalization",
        "src.layers.GroupNormalization",
        "src.layers.LayerNormalization",
        "src.layers.RMSNormalization",
        "src.layers.SpectralNormalization",
        "src.layers.UnitNormalization",
        "src.layers.AdaptiveAveragePooling1D",
        "src.layers.AdaptiveAveragePooling2D",
        "src.layers.AdaptiveAveragePooling3D",
        "src.layers.AdaptiveMaxPooling1D",
        "src.layers.AdaptiveMaxPooling2D",
        "src.layers.AdaptiveMaxPooling3D",
        "src.layers.AveragePooling1D",
        "src.layers.AveragePooling2D",
        "src.layers.AveragePooling3D",
        "src.layers.GlobalAveragePooling1D",
        "src.layers.GlobalAveragePooling2D",
        "src.layers.GlobalAveragePooling3D",
        "src.layers.GlobalMaxPooling1D",
        "src.layers.GlobalMaxPooling2D",
        "src.layers.GlobalMaxPooling3D",
        "src.layers.MaxPooling1D",
        "src.layers.MaxPooling2D",
        "src.layers.MaxPooling3D",
        "src.layers.CategoryEncoding",
        "src.layers.Discretization",
        "src.layers.HashedCrossing",
        "src.layers.Hashing",
        "src.layers.AugMix",
        "src.layers.AutoContrast",
        "src.layers.CenterCrop",
        "src.layers.CutMix",
        "src.layers.Equalization",
        "src.layers.MaxNumBoundingBoxes",
        "src.layers.MixUp",
        "src.layers.RandAugment",
        "src.layers.RandomBrightness",
        "src.layers.RandomColorDegeneration",
        "src.layers.RandomColorJitter",
        "src.layers.RandomContrast",
        "src.layers.RandomCrop",
        "src.layers.RandomElasticTransform",
        "src.layers.RandomErasing",
        "src.layers.RandomFlip",
        "src.layers.RandomGaussianBlur",
        "src.layers.RandomGrayscale",
        "src.layers.RandomHue",
        "src.layers.RandomInvert",
        "src.layers.RandomPerspective",
        "src.layers.RandomPosterization",
        "src.layers.RandomRotation",
        "src.layers.RandomSaturation",
        "src.layers.RandomSharpness",
        "src.layers.RandomShear",
        "src.layers.RandomTranslation",
        "src.layers.RandomZoom",
        "src.layers.Resizing",
        "src.layers.Solarization",
        "src.layers.IndexLookup",
        "src.layers.IntegerLookup",
        "src.layers.MelSpectrogram",
        "src.layers.Normalization",
        "src.layers.Pipeline",
        "src.layers.Rescaling",
        "src.layers.STFTSpectrogram",
        "src.layers.StringLookup",
        "src.layers.TextVectorization",
        "src.layers.ActivityRegularization",
        "src.layers.AlphaDropout",
        "src.layers.Dropout",
        "src.layers.GaussianDropout",
        "src.layers.GaussianNoise",
        "src.layers.SpatialDropout1D",
        "src.layers.SpatialDropout2D",
        "src.layers.SpatialDropout3D",
        "src.layers.Cropping1D",
        "src.layers.Cropping2D",
        "src.layers.Cropping3D",
        "src.layers.Flatten",
        "src.layers.Permute",
        "src.layers.RepeatVector",
        "src.layers.Reshape",
        "src.layers.UpSampling1D",
        "src.layers.UpSampling2D",
        "src.layers.UpSampling3D",
        "src.layers.ZeroPadding1D",
        "src.layers.ZeroPadding2D",
        "src.layers.ZeroPadding3D",
        "src.layers.Bidirectional",
        "src.layers.ConvLSTM1D",
        "src.layers.ConvLSTM2D",
        "src.layers.ConvLSTM3D",
        "src.layers.GRU",
        "src.layers.GRUCell",
        "src.layers.LSTM",
        "src.layers.LSTMCell",
        "src.layers.RNN",
        "src.layers.SimpleRNN",
        "src.layers.SimpleRNNCell",
        "src.layers.StackedRNNCells",
        "src.layers.TimeDistributed",
        "src.layers.serialization_lib",
        "src.layers.serialize",
        "src.layers.deserialize",
        "src.layers.input_spec.backend",
        "src.layers.input_spec.tree",
        "src.layers.input_spec.keras_export",
        "src.layers.input_spec.InputSpec",
        "src.layers.input_spec.assert_input_compatibility",
        "src.layers.layer.backend",
        "src.layers.layer.constraints",
        "src.layers.layer.dtype_policies",
        "src.layers.layer.initializers",
        "src.layers.layer.regularizers",
        "src.layers.layer.tree",
        "src.layers.layer.keras_export",
        "src.layers.layer.KerasTensor",
        "src.layers.layer.global_state",
        "src.layers.layer.remat",
        "src.layers.layer.any_symbolic_tensors",
        "src.layers.layer.current_path",
        "src.layers.layer.get_current_remat_mode",
        "src.layers.layer.in_symbolic_scope",
        "src.layers.layer.is_nnx_enabled",
        "src.layers.layer.distribution_lib",
        "src.layers.layer.DTypePolicyMap",
        "src.layers.layer.input_spec",
        "src.layers.layer.Metric",
        "src.layers.layer.Node",
        "src.layers.layer.Operation",
        "src.layers.layer.validate_and_resolve_config",
        "src.layers.layer.python_utils",
        "src.layers.layer.summary_utils",
        "src.layers.layer.traceback_utils",
        "src.layers.layer.tracking",
        "src.layers.layer.BackendLayer",
        "src.layers.layer.Layer",
        "src.layers.layer.is_backend_tensor_or_symbolic",
        "src.layers.layer.CallSpec",
        "src.layers.layer.get_arguments_dict",
        "src.layers.layer.get_shapes_dict",
        "src.layers.layer.update_shapes_dict_for_target_fn",
        "src.layers.layer.CallContext",
        "src.layers.layer.is_shape_tuple",
        "src.layers.layer.might_have_unbuilt_state",
        "src.layers.activations.ELU",
        "src.layers.activations.LeakyReLU",
        "src.layers.activations.PReLU",
        "src.layers.activations.ReLU",
        "src.layers.activations.Softmax",
        "src.layers.activations.leaky_relu.activations",
        "src.layers.activations.leaky_relu.keras_export",
        "src.layers.activations.leaky_relu.Layer",
        "src.layers.activations.leaky_relu.LeakyReLU",
        "src.layers.activations.activation.activations",
        "src.layers.activations.activation.keras_export",
        "src.layers.activations.activation.Layer",
        "src.layers.activations.activation.Activation",
        "src.layers.activations.prelu.activations",
        "src.layers.activations.prelu.constraints",
        "src.layers.activations.prelu.initializers",
        "src.layers.activations.prelu.regularizers",
        "src.layers.activations.prelu.keras_export",
        "src.layers.activations.prelu.InputSpec",
        "src.layers.activations.prelu.Layer",
        "src.layers.activations.prelu.PReLU",
        "src.layers.activations.relu.activations",
        "src.layers.activations.relu.keras_export",
        "src.layers.activations.relu.Layer",
        "src.layers.activations.relu.ReLU",
        "src.layers.activations.elu.activations",
        "src.layers.activations.elu.keras_export",
        "src.layers.activations.elu.Layer",
        "src.layers.activations.elu.ELU",
        "src.layers.activations.softmax.activations",
        "src.layers.activations.softmax.backend",
        "src.layers.activations.softmax.keras_export",
        "src.layers.activations.softmax.Layer",
        "src.layers.activations.softmax.Softmax",
        "src.layers.attention.grouped_query_attention.constraints",
        "src.layers.attention.grouped_query_attention.initializers",
        "src.layers.attention.grouped_query_attention.ops",
        "src.layers.attention.grouped_query_attention.regularizers",
        "src.layers.attention.grouped_query_attention.keras_export",
        "src.layers.attention.grouped_query_attention.is_flash_attention_enabled",
        "src.layers.attention.grouped_query_attention.Softmax",
        "src.layers.attention.grouped_query_attention.EinsumDense",
        "src.layers.attention.grouped_query_attention.Layer",
        "src.layers.attention.grouped_query_attention.Dropout",
        "src.layers.attention.grouped_query_attention.GroupedQueryAttention",
        "src.layers.attention.attention.backend",
        "src.layers.attention.attention.ops",
        "src.layers.attention.attention.keras_export",
        "src.layers.attention.attention.KerasTensor",
        "src.layers.attention.attention.Layer",
        "src.layers.attention.attention.Attention",
        "src.layers.attention.multi_head_attention.backend",
        "src.layers.attention.multi_head_attention.constraints",
        "src.layers.attention.multi_head_attention.initializers",
        "src.layers.attention.multi_head_attention.ops",
        "src.layers.attention.multi_head_attention.regularizers",
        "src.layers.attention.multi_head_attention.keras_export",
        "src.layers.attention.multi_head_attention.is_flash_attention_enabled",
        "src.layers.attention.multi_head_attention.Softmax",
        "src.layers.attention.multi_head_attention.EinsumDense",
        "src.layers.attention.multi_head_attention.Layer",
        "src.layers.attention.multi_head_attention.Dropout",
        "src.layers.attention.multi_head_attention.MultiHeadAttention",
        "src.layers.attention.additive_attention.ops",
        "src.layers.attention.additive_attention.keras_export",
        "src.layers.attention.additive_attention.Attention",
        "src.layers.attention.additive_attention.AdditiveAttention",
        "src.layers.reshaping.up_sampling3d.backend",
        "src.layers.reshaping.up_sampling3d.ops",
        "src.layers.reshaping.up_sampling3d.keras_export",
        "src.layers.reshaping.up_sampling3d.InputSpec",
        "src.layers.reshaping.up_sampling3d.Layer",
        "src.layers.reshaping.up_sampling3d.argument_validation",
        "src.layers.reshaping.up_sampling3d.UpSampling3D",
        "src.layers.reshaping.up_sampling2d.backend",
        "src.layers.reshaping.up_sampling2d.ops",
        "src.layers.reshaping.up_sampling2d.keras_export",
        "src.layers.reshaping.up_sampling2d.InputSpec",
        "src.layers.reshaping.up_sampling2d.Layer",
        "src.layers.reshaping.up_sampling2d.argument_validation",
        "src.layers.reshaping.up_sampling2d.UpSampling2D",
        "src.layers.reshaping.permute.ops",
        "src.layers.reshaping.permute.keras_export",
        "src.layers.reshaping.permute.KerasTensor",
        "src.layers.reshaping.permute.InputSpec",
        "src.layers.reshaping.permute.Layer",
        "src.layers.reshaping.permute.Permute",
        "src.layers.reshaping.up_sampling1d.ops",
        "src.layers.reshaping.up_sampling1d.keras_export",
        "src.layers.reshaping.up_sampling1d.InputSpec",
        "src.layers.reshaping.up_sampling1d.Layer",
        "src.layers.reshaping.up_sampling1d.UpSampling1D",
        "src.layers.reshaping.flatten.backend",
        "src.layers.reshaping.flatten.ops",
        "src.layers.reshaping.flatten.keras_export",
        "src.layers.reshaping.flatten.KerasTensor",
        "src.layers.reshaping.flatten.InputSpec",
        "src.layers.reshaping.flatten.Layer",
        "src.layers.reshaping.flatten.Flatten",
        "src.layers.reshaping.repeat_vector.ops",
        "src.layers.reshaping.repeat_vector.keras_export",
        "src.layers.reshaping.repeat_vector.InputSpec",
        "src.layers.reshaping.repeat_vector.Layer",
        "src.layers.reshaping.repeat_vector.RepeatVector",
        "src.layers.reshaping.cropping1d.keras_export",
        "src.layers.reshaping.cropping1d.InputSpec",
        "src.layers.reshaping.cropping1d.Layer",
        "src.layers.reshaping.cropping1d.argument_validation",
        "src.layers.reshaping.cropping1d.Cropping1D",
        "src.layers.reshaping.reshape.ops",
        "src.layers.reshaping.reshape.keras_export",
        "src.layers.reshaping.reshape.KerasTensor",
        "src.layers.reshaping.reshape.Layer",
        "src.layers.reshaping.reshape.operation_utils",
        "src.layers.reshaping.reshape.Reshape",
        "src.layers.reshaping.cropping3d.backend",
        "src.layers.reshaping.cropping3d.keras_export",
        "src.layers.reshaping.cropping3d.InputSpec",
        "src.layers.reshaping.cropping3d.Layer",
        "src.layers.reshaping.cropping3d.argument_validation",
        "src.layers.reshaping.cropping3d.Cropping3D",
        "src.layers.reshaping.cropping2d.backend",
        "src.layers.reshaping.cropping2d.keras_export",
        "src.layers.reshaping.cropping2d.InputSpec",
        "src.layers.reshaping.cropping2d.Layer",
        "src.layers.reshaping.cropping2d.argument_validation",
        "src.layers.reshaping.cropping2d.Cropping2D",
        "src.layers.reshaping.zero_padding3d.backend",
        "src.layers.reshaping.zero_padding3d.ops",
        "src.layers.reshaping.zero_padding3d.keras_export",
        "src.layers.reshaping.zero_padding3d.InputSpec",
        "src.layers.reshaping.zero_padding3d.Layer",
        "src.layers.reshaping.zero_padding3d.argument_validation",
        "src.layers.reshaping.zero_padding3d.ZeroPadding3D",
        "src.layers.reshaping.zero_padding2d.backend",
        "src.layers.reshaping.zero_padding2d.ops",
        "src.layers.reshaping.zero_padding2d.keras_export",
        "src.layers.reshaping.zero_padding2d.InputSpec",
        "src.layers.reshaping.zero_padding2d.Layer",
        "src.layers.reshaping.zero_padding2d.argument_validation",
        "src.layers.reshaping.zero_padding2d.ZeroPadding2D",
        "src.layers.reshaping.zero_padding1d.backend",
        "src.layers.reshaping.zero_padding1d.ops",
        "src.layers.reshaping.zero_padding1d.keras_export",
        "src.layers.reshaping.zero_padding1d.InputSpec",
        "src.layers.reshaping.zero_padding1d.Layer",
        "src.layers.reshaping.zero_padding1d.argument_validation",
        "src.layers.reshaping.zero_padding1d.ZeroPadding1D",
        "src.layers.pooling.max_pooling1d.keras_export",
        "src.layers.pooling.max_pooling1d.BasePooling",
        "src.layers.pooling.max_pooling1d.MaxPooling1D",
        "src.layers.pooling.max_pooling3d.keras_export",
        "src.layers.pooling.max_pooling3d.BasePooling",
        "src.layers.pooling.max_pooling3d.MaxPooling3D",
        "src.layers.pooling.max_pooling2d.keras_export",
        "src.layers.pooling.max_pooling2d.BasePooling",
        "src.layers.pooling.max_pooling2d.MaxPooling2D",
        "src.layers.pooling.global_max_pooling1d.ops",
        "src.layers.pooling.global_max_pooling1d.keras_export",
        "src.layers.pooling.global_max_pooling1d.BaseGlobalPooling",
        "src.layers.pooling.global_max_pooling1d.GlobalMaxPooling1D",
        "src.layers.pooling.average_pooling1d.keras_export",
        "src.layers.pooling.average_pooling1d.BasePooling",
        "src.layers.pooling.average_pooling1d.AveragePooling1D",
        "src.layers.pooling.global_max_pooling2d.ops",
        "src.layers.pooling.global_max_pooling2d.keras_export",
        "src.layers.pooling.global_max_pooling2d.BaseGlobalPooling",
        "src.layers.pooling.global_max_pooling2d.GlobalMaxPooling2D",
        "src.layers.pooling.average_pooling3d.keras_export",
        "src.layers.pooling.average_pooling3d.BasePooling",
        "src.layers.pooling.average_pooling3d.AveragePooling3D",
        "src.layers.pooling.average_pooling2d.keras_export",
        "src.layers.pooling.average_pooling2d.BasePooling",
        "src.layers.pooling.average_pooling2d.AveragePooling2D",
        "src.layers.pooling.global_max_pooling3d.ops",
        "src.layers.pooling.global_max_pooling3d.keras_export",
        "src.layers.pooling.global_max_pooling3d.BaseGlobalPooling",
        "src.layers.pooling.global_max_pooling3d.GlobalMaxPooling3D",
        "src.layers.pooling.adaptive_average_pooling3d.keras_export",
        "src.layers.pooling.adaptive_average_pooling3d.BaseAdaptiveAveragePooling",
        "src.layers.pooling.adaptive_average_pooling3d.AdaptiveAveragePooling3D",
        "src.layers.pooling.adaptive_max_pooling3d.keras_export",
        "src.layers.pooling.adaptive_max_pooling3d.BaseAdaptiveMaxPooling",
        "src.layers.pooling.adaptive_max_pooling3d.AdaptiveMaxPooling3D",
        "src.layers.pooling.global_average_pooling2d.ops",
        "src.layers.pooling.global_average_pooling2d.keras_export",
        "src.layers.pooling.global_average_pooling2d.BaseGlobalPooling",
        "src.layers.pooling.global_average_pooling2d.GlobalAveragePooling2D",
        "src.layers.pooling.global_average_pooling3d.ops",
        "src.layers.pooling.global_average_pooling3d.keras_export",
        "src.layers.pooling.global_average_pooling3d.BaseGlobalPooling",
        "src.layers.pooling.global_average_pooling3d.GlobalAveragePooling3D",
        "src.layers.pooling.adaptive_max_pooling2d.keras_export",
        "src.layers.pooling.adaptive_max_pooling2d.BaseAdaptiveMaxPooling",
        "src.layers.pooling.adaptive_max_pooling2d.AdaptiveMaxPooling2D",
        "src.layers.pooling.base_global_pooling.backend",
        "src.layers.pooling.base_global_pooling.InputSpec",
        "src.layers.pooling.base_global_pooling.Layer",
        "src.layers.pooling.base_global_pooling.BaseGlobalPooling",
        "src.layers.pooling.adaptive_average_pooling2d.keras_export",
        "src.layers.pooling.adaptive_average_pooling2d.BaseAdaptiveAveragePooling",
        "src.layers.pooling.adaptive_average_pooling2d.AdaptiveAveragePooling2D",
        "src.layers.pooling.global_average_pooling1d.backend",
        "src.layers.pooling.global_average_pooling1d.ops",
        "src.layers.pooling.global_average_pooling1d.keras_export",
        "src.layers.pooling.global_average_pooling1d.BaseGlobalPooling",
        "src.layers.pooling.global_average_pooling1d.GlobalAveragePooling1D",
        "src.layers.pooling.adaptive_max_pooling1d.keras_export",
        "src.layers.pooling.adaptive_max_pooling1d.BaseAdaptiveMaxPooling",
        "src.layers.pooling.adaptive_max_pooling1d.AdaptiveMaxPooling1D",
        "src.layers.pooling.adaptive_average_pooling1d.keras_export",
        "src.layers.pooling.adaptive_average_pooling1d.BaseAdaptiveAveragePooling",
        "src.layers.pooling.adaptive_average_pooling1d.AdaptiveAveragePooling1D",
        "src.layers.pooling.base_adaptive_pooling.ops",
        "src.layers.pooling.base_adaptive_pooling.config",
        "src.layers.pooling.base_adaptive_pooling.Layer",
        "src.layers.pooling.base_adaptive_pooling.BaseAdaptivePooling",
        "src.layers.pooling.base_adaptive_pooling.BaseAdaptiveAveragePooling",
        "src.layers.pooling.base_adaptive_pooling.BaseAdaptiveMaxPooling",
        "src.layers.pooling.base_pooling.backend",
        "src.layers.pooling.base_pooling.ops",
        "src.layers.pooling.base_pooling.InputSpec",
        "src.layers.pooling.base_pooling.Layer",
        "src.layers.pooling.base_pooling.compute_pooling_output_shape",
        "src.layers.pooling.base_pooling.argument_validation",
        "src.layers.pooling.base_pooling.BasePooling",
        "src.layers.core.wrapper.keras_export",
        "src.layers.core.wrapper.Layer",
        "src.layers.core.wrapper.serialization_lib",
        "src.layers.core.wrapper.Wrapper",
        "src.layers.core.embedding.backend",
        "src.layers.core.embedding.constraints",
        "src.layers.core.embedding.dtype_policies",
        "src.layers.core.embedding.initializers",
        "src.layers.core.embedding.ops",
        "src.layers.core.embedding.quantizers",
        "src.layers.core.embedding.regularizers",
        "src.layers.core.embedding.keras_export",
        "src.layers.core.embedding.KerasTensor",
        "src.layers.core.embedding.Layer",
        "src.layers.core.embedding.QuantizationConfig",
        "src.layers.core.embedding.serialization_lib",
        "src.layers.core.embedding.Embedding",
        "src.layers.core.dense.activations",
        "src.layers.core.dense.constraints",
        "src.layers.core.dense.initializers",
        "src.layers.core.dense.ops",
        "src.layers.core.dense.quantizers",
        "src.layers.core.dense.regularizers",
        "src.layers.core.dense.keras_export",
        "src.layers.core.dense.InputSpec",
        "src.layers.core.dense.Layer",
        "src.layers.core.dense.QuantizationConfig",
        "src.layers.core.dense.dequantize_with_sz_map",
        "src.layers.core.dense.serialization_lib",
        "src.layers.core.dense.Dense",
        "src.layers.core.masking.backend",
        "src.layers.core.masking.ops",
        "src.layers.core.masking.keras_export",
        "src.layers.core.masking.Layer",
        "src.layers.core.masking.deserialize_keras_object",
        "src.layers.core.masking.Masking",
        "src.layers.core.lambda_layer.backend",
        "src.layers.core.lambda_layer.tree",
        "src.layers.core.lambda_layer.keras_export",
        "src.layers.core.lambda_layer.Layer",
        "src.layers.core.lambda_layer.serialization_lib",
        "src.layers.core.lambda_layer.python_utils",
        "src.layers.core.lambda_layer.Lambda",
        "src.layers.core.input_layer.backend",
        "src.layers.core.input_layer.keras_export",
        "src.layers.core.input_layer.Layer",
        "src.layers.core.input_layer.Node",
        "src.layers.core.input_layer.InputLayer",
        "src.layers.core.input_layer.Input",
        "src.layers.core.einsum_dense.activations",
        "src.layers.core.einsum_dense.backend",
        "src.layers.core.einsum_dense.constraints",
        "src.layers.core.einsum_dense.dtype_policies",
        "src.layers.core.einsum_dense.initializers",
        "src.layers.core.einsum_dense.ops",
        "src.layers.core.einsum_dense.quantizers",
        "src.layers.core.einsum_dense.regularizers",
        "src.layers.core.einsum_dense.keras_export",
        "src.layers.core.einsum_dense.InputSpec",
        "src.layers.core.einsum_dense.Layer",
        "src.layers.core.einsum_dense.QuantizationConfig",
        "src.layers.core.einsum_dense.dequantize_with_sz_map",
        "src.layers.core.einsum_dense.serialization_lib",
        "src.layers.core.einsum_dense.EinsumDense",
        "src.layers.core.reversible_embedding.dtype_policies",
        "src.layers.core.reversible_embedding.layers",
        "src.layers.core.reversible_embedding.ops",
        "src.layers.core.reversible_embedding.quantizers",
        "src.layers.core.reversible_embedding.keras_export",
        "src.layers.core.reversible_embedding.KerasTensor",
        "src.layers.core.reversible_embedding.QuantizationConfig",
        "src.layers.core.reversible_embedding.ReversibleEmbedding",
        "src.layers.core.identity.tree",
        "src.layers.core.identity.keras_export",
        "src.layers.core.identity.KerasTensor",
        "src.layers.core.identity.Layer",
        "src.layers.core.identity.Identity",
        "src.layers.normalization.batch_normalization.backend",
        "src.layers.normalization.batch_normalization.constraints",
        "src.layers.normalization.batch_normalization.initializers",
        "src.layers.normalization.batch_normalization.ops",
        "src.layers.normalization.batch_normalization.regularizers",
        "src.layers.normalization.batch_normalization.keras_export",
        "src.layers.normalization.batch_normalization.InputSpec",
        "src.layers.normalization.batch_normalization.Layer",
        "src.layers.normalization.batch_normalization.BatchNormalization",
        "src.layers.normalization.layer_normalization.constraints",
        "src.layers.normalization.layer_normalization.initializers",
        "src.layers.normalization.layer_normalization.ops",
        "src.layers.normalization.layer_normalization.regularizers",
        "src.layers.normalization.layer_normalization.keras_export",
        "src.layers.normalization.layer_normalization.Layer",
        "src.layers.normalization.layer_normalization.LayerNormalization",
        "src.layers.normalization.unit_normalization.ops",
        "src.layers.normalization.unit_normalization.keras_export",
        "src.layers.normalization.unit_normalization.Layer",
        "src.layers.normalization.unit_normalization.UnitNormalization",
        "src.layers.normalization.group_normalization.backend",
        "src.layers.normalization.group_normalization.constraints",
        "src.layers.normalization.group_normalization.initializers",
        "src.layers.normalization.group_normalization.ops",
        "src.layers.normalization.group_normalization.regularizers",
        "src.layers.normalization.group_normalization.keras_export",
        "src.layers.normalization.group_normalization.InputSpec",
        "src.layers.normalization.group_normalization.Layer",
        "src.layers.normalization.group_normalization.GroupNormalization",
        "src.layers.normalization.rms_normalization.ops",
        "src.layers.normalization.rms_normalization.keras_export",
        "src.layers.normalization.rms_normalization.Layer",
        "src.layers.normalization.rms_normalization.RMSNormalization",
        "src.layers.normalization.spectral_normalization.initializers",
        "src.layers.normalization.spectral_normalization.ops",
        "src.layers.normalization.spectral_normalization.keras_export",
        "src.layers.normalization.spectral_normalization.Wrapper",
        "src.layers.normalization.spectral_normalization.InputSpec",
        "src.layers.normalization.spectral_normalization.normalize",
        "src.layers.normalization.spectral_normalization.SpectralNormalization",
        "src.layers.convolutional.base_depthwise_conv.activations",
        "src.layers.convolutional.base_depthwise_conv.constraints",
        "src.layers.convolutional.base_depthwise_conv.initializers",
        "src.layers.convolutional.base_depthwise_conv.ops",
        "src.layers.convolutional.base_depthwise_conv.regularizers",
        "src.layers.convolutional.base_depthwise_conv.standardize_data_format",
        "src.layers.convolutional.base_depthwise_conv.InputSpec",
        "src.layers.convolutional.base_depthwise_conv.Layer",
        "src.layers.convolutional.base_depthwise_conv.compute_conv_output_shape",
        "src.layers.convolutional.base_depthwise_conv.standardize_padding",
        "src.layers.convolutional.base_depthwise_conv.standardize_tuple",
        "src.layers.convolutional.base_depthwise_conv.BaseDepthwiseConv",
        "src.layers.convolutional.depthwise_conv2d.keras_export",
        "src.layers.convolutional.depthwise_conv2d.BaseDepthwiseConv",
        "src.layers.convolutional.depthwise_conv2d.DepthwiseConv2D",
        "src.layers.convolutional.depthwise_conv1d.keras_export",
        "src.layers.convolutional.depthwise_conv1d.BaseDepthwiseConv",
        "src.layers.convolutional.depthwise_conv1d.DepthwiseConv1D",
        "src.layers.convolutional.base_conv_transpose.activations",
        "src.layers.convolutional.base_conv_transpose.constraints",
        "src.layers.convolutional.base_conv_transpose.initializers",
        "src.layers.convolutional.base_conv_transpose.ops",
        "src.layers.convolutional.base_conv_transpose.regularizers",
        "src.layers.convolutional.base_conv_transpose.standardize_data_format",
        "src.layers.convolutional.base_conv_transpose.compute_conv_transpose_output_shape",
        "src.layers.convolutional.base_conv_transpose.InputSpec",
        "src.layers.convolutional.base_conv_transpose.Layer",
        "src.layers.convolutional.base_conv_transpose.standardize_padding",
        "src.layers.convolutional.base_conv_transpose.standardize_tuple",
        "src.layers.convolutional.base_conv_transpose.BaseConvTranspose",
        "src.layers.convolutional.separable_conv2d.keras_export",
        "src.layers.convolutional.separable_conv2d.BaseSeparableConv",
        "src.layers.convolutional.separable_conv2d.SeparableConv2D",
        "src.layers.convolutional.separable_conv1d.keras_export",
        "src.layers.convolutional.separable_conv1d.BaseSeparableConv",
        "src.layers.convolutional.separable_conv1d.SeparableConv1D",
        "src.layers.convolutional.conv3d_transpose.keras_export",
        "src.layers.convolutional.conv3d_transpose.BaseConvTranspose",
        "src.layers.convolutional.conv3d_transpose.Conv3DTranspose",
        "src.layers.convolutional.conv1d.ops",
        "src.layers.convolutional.conv1d.keras_export",
        "src.layers.convolutional.conv1d.BaseConv",
        "src.layers.convolutional.conv1d.Conv1D",
        "src.layers.convolutional.conv2d_transpose.keras_export",
        "src.layers.convolutional.conv2d_transpose.BaseConvTranspose",
        "src.layers.convolutional.conv2d_transpose.Conv2DTranspose",
        "src.layers.convolutional.conv3d.keras_export",
        "src.layers.convolutional.conv3d.BaseConv",
        "src.layers.convolutional.conv3d.Conv3D",
        "src.layers.convolutional.conv2d.keras_export",
        "src.layers.convolutional.conv2d.BaseConv",
        "src.layers.convolutional.conv2d.Conv2D",
        "src.layers.convolutional.conv1d_transpose.keras_export",
        "src.layers.convolutional.conv1d_transpose.BaseConvTranspose",
        "src.layers.convolutional.conv1d_transpose.Conv1DTranspose",
        "src.layers.convolutional.base_separable_conv.activations",
        "src.layers.convolutional.base_separable_conv.constraints",
        "src.layers.convolutional.base_separable_conv.initializers",
        "src.layers.convolutional.base_separable_conv.ops",
        "src.layers.convolutional.base_separable_conv.regularizers",
        "src.layers.convolutional.base_separable_conv.standardize_data_format",
        "src.layers.convolutional.base_separable_conv.InputSpec",
        "src.layers.convolutional.base_separable_conv.Layer",
        "src.layers.convolutional.base_separable_conv.compute_conv_output_shape",
        "src.layers.convolutional.base_separable_conv.standardize_padding",
        "src.layers.convolutional.base_separable_conv.standardize_tuple",
        "src.layers.convolutional.base_separable_conv.BaseSeparableConv",
        "src.layers.convolutional.base_conv.activations",
        "src.layers.convolutional.base_conv.constraints",
        "src.layers.convolutional.base_conv.initializers",
        "src.layers.convolutional.base_conv.ops",
        "src.layers.convolutional.base_conv.regularizers",
        "src.layers.convolutional.base_conv.standardize_data_format",
        "src.layers.convolutional.base_conv.InputSpec",
        "src.layers.convolutional.base_conv.Layer",
        "src.layers.convolutional.base_conv.compute_conv_output_shape",
        "src.layers.convolutional.base_conv.standardize_padding",
        "src.layers.convolutional.base_conv.standardize_tuple",
        "src.layers.convolutional.base_conv.BaseConv",
        "src.layers.regularization.spatial_dropout.backend",
        "src.layers.regularization.spatial_dropout.ops",
        "src.layers.regularization.spatial_dropout.keras_export",
        "src.layers.regularization.spatial_dropout.InputSpec",
        "src.layers.regularization.spatial_dropout.Dropout",
        "src.layers.regularization.spatial_dropout.BaseSpatialDropout",
        "src.layers.regularization.spatial_dropout.SpatialDropout1D",
        "src.layers.regularization.spatial_dropout.SpatialDropout2D",
        "src.layers.regularization.spatial_dropout.SpatialDropout3D",
        "src.layers.regularization.gaussian_dropout.backend",
        "src.layers.regularization.gaussian_dropout.layers",
        "src.layers.regularization.gaussian_dropout.ops",
        "src.layers.regularization.gaussian_dropout.keras_export",
        "src.layers.regularization.gaussian_dropout.GaussianDropout",
        "src.layers.regularization.activity_regularization.regularizers",
        "src.layers.regularization.activity_regularization.keras_export",
        "src.layers.regularization.activity_regularization.Layer",
        "src.layers.regularization.activity_regularization.ActivityRegularization",
        "src.layers.regularization.dropout.backend",
        "src.layers.regularization.dropout.keras_export",
        "src.layers.regularization.dropout.Layer",
        "src.layers.regularization.dropout.Dropout",
        "src.layers.regularization.gaussian_noise.backend",
        "src.layers.regularization.gaussian_noise.layers",
        "src.layers.regularization.gaussian_noise.ops",
        "src.layers.regularization.gaussian_noise.keras_export",
        "src.layers.regularization.gaussian_noise.GaussianNoise",
        "src.layers.regularization.alpha_dropout.backend",
        "src.layers.regularization.alpha_dropout.ops",
        "src.layers.regularization.alpha_dropout.keras_export",
        "src.layers.regularization.alpha_dropout.Layer",
        "src.layers.regularization.alpha_dropout.AlphaDropout",
        "src.layers.preprocessing.mel_spectrogram.keras_export",
        "src.layers.preprocessing.mel_spectrogram.DataLayer",
        "src.layers.preprocessing.mel_spectrogram.MelSpectrogram",
        "src.layers.preprocessing.rescaling.backend",
        "src.layers.preprocessing.rescaling.keras_export",
        "src.layers.preprocessing.rescaling.DataLayer",
        "src.layers.preprocessing.rescaling.serialization_lib",
        "src.layers.preprocessing.rescaling.Rescaling",
        "src.layers.preprocessing.index_lookup.backend",
        "src.layers.preprocessing.index_lookup.Layer",
        "src.layers.preprocessing.index_lookup.serialization_lib",
        "src.layers.preprocessing.index_lookup.argument_validation",
        "src.layers.preprocessing.index_lookup.numerical_utils",
        "src.layers.preprocessing.index_lookup.tf_utils",
        "src.layers.preprocessing.index_lookup.tf",
        "src.layers.preprocessing.index_lookup.IndexLookup",
        "src.layers.preprocessing.index_lookup.get_null_initializer",
        "src.layers.preprocessing.index_lookup.listify_tensors",
        "src.layers.preprocessing.hashed_crossing.backend",
        "src.layers.preprocessing.hashed_crossing.keras_export",
        "src.layers.preprocessing.hashed_crossing.Layer",
        "src.layers.preprocessing.hashed_crossing.argument_validation",
        "src.layers.preprocessing.hashed_crossing.backend_utils",
        "src.layers.preprocessing.hashed_crossing.numerical_utils",
        "src.layers.preprocessing.hashed_crossing.tf_utils",
        "src.layers.preprocessing.hashed_crossing.tf",
        "src.layers.preprocessing.hashed_crossing.HashedCrossing",
        "src.layers.preprocessing.discretization.backend",
        "src.layers.preprocessing.discretization.keras_export",
        "src.layers.preprocessing.discretization.DataLayer",
        "src.layers.preprocessing.discretization.argument_validation",
        "src.layers.preprocessing.discretization.numerical_utils",
        "src.layers.preprocessing.discretization.tf",
        "src.layers.preprocessing.discretization.Discretization",
        "src.layers.preprocessing.discretization.summarize",
        "src.layers.preprocessing.discretization.merge_summaries",
        "src.layers.preprocessing.discretization.get_bin_boundaries",
        "src.layers.preprocessing.discretization.compress_summary",
        "src.layers.preprocessing.category_encoding.keras_export",
        "src.layers.preprocessing.category_encoding.KerasTensor",
        "src.layers.preprocessing.category_encoding.DataLayer",
        "src.layers.preprocessing.category_encoding.backend_utils",
        "src.layers.preprocessing.category_encoding.numerical_utils",
        "src.layers.preprocessing.category_encoding.CategoryEncoding",
        "src.layers.preprocessing.integer_lookup.backend",
        "src.layers.preprocessing.integer_lookup.keras_export",
        "src.layers.preprocessing.integer_lookup.IndexLookup",
        "src.layers.preprocessing.integer_lookup.backend_utils",
        "src.layers.preprocessing.integer_lookup.tf",
        "src.layers.preprocessing.integer_lookup.IntegerLookup",
        "src.layers.preprocessing.pipeline.tree",
        "src.layers.preprocessing.pipeline.keras_export",
        "src.layers.preprocessing.pipeline.Layer",
        "src.layers.preprocessing.pipeline.serialization_lib",
        "src.layers.preprocessing.pipeline.Pipeline",
        "src.layers.preprocessing.text_vectorization.backend",
        "src.layers.preprocessing.text_vectorization.keras_export",
        "src.layers.preprocessing.text_vectorization.Layer",
        "src.layers.preprocessing.text_vectorization.listify_tensors",
        "src.layers.preprocessing.text_vectorization.StringLookup",
        "src.layers.preprocessing.text_vectorization.serialization_lib",
        "src.layers.preprocessing.text_vectorization.argument_validation",
        "src.layers.preprocessing.text_vectorization.backend_utils",
        "src.layers.preprocessing.text_vectorization.tf_utils",
        "src.layers.preprocessing.text_vectorization.tf",
        "src.layers.preprocessing.text_vectorization.TextVectorization",
        "src.layers.preprocessing.stft_spectrogram.backend",
        "src.layers.preprocessing.stft_spectrogram.initializers",
        "src.layers.preprocessing.stft_spectrogram.layers",
        "src.layers.preprocessing.stft_spectrogram.ops",
        "src.layers.preprocessing.stft_spectrogram.keras_export",
        "src.layers.preprocessing.stft_spectrogram.scipy",
        "src.layers.preprocessing.stft_spectrogram.STFTSpectrogram",
        "src.layers.preprocessing.data_layer.tree",
        "src.layers.preprocessing.data_layer.Layer",
        "src.layers.preprocessing.data_layer.SeedGenerator",
        "src.layers.preprocessing.data_layer.backend_utils",
        "src.layers.preprocessing.data_layer.jax_utils",
        "src.layers.preprocessing.data_layer.tracking",
        "src.layers.preprocessing.data_layer.DataLayer",
        "src.layers.preprocessing.string_lookup.backend",
        "src.layers.preprocessing.string_lookup.keras_export",
        "src.layers.preprocessing.string_lookup.IndexLookup",
        "src.layers.preprocessing.string_lookup.backend_utils",
        "src.layers.preprocessing.string_lookup.tf",
        "src.layers.preprocessing.string_lookup.StringLookup",
        "src.layers.preprocessing.normalization.backend",
        "src.layers.preprocessing.normalization.ops",
        "src.layers.preprocessing.normalization.keras_export",
        "src.layers.preprocessing.normalization.DataLayer",
        "src.layers.preprocessing.normalization.PyDataset",
        "src.layers.preprocessing.normalization.tf",
        "src.layers.preprocessing.normalization.Normalization",
        "src.layers.preprocessing.hashing.backend",
        "src.layers.preprocessing.hashing.keras_export",
        "src.layers.preprocessing.hashing.Layer",
        "src.layers.preprocessing.hashing.backend_utils",
        "src.layers.preprocessing.hashing.numerical_utils",
        "src.layers.preprocessing.hashing.tf_utils",
        "src.layers.preprocessing.hashing.tf",
        "src.layers.preprocessing.hashing.Hashing",
        "src.layers.preprocessing.feature_space.backend",
        "src.layers.preprocessing.feature_space.layers",
        "src.layers.preprocessing.feature_space.tree",
        "src.layers.preprocessing.feature_space.keras_export",
        "src.layers.preprocessing.feature_space.Layer",
        "src.layers.preprocessing.feature_space.DataLayer",
        "src.layers.preprocessing.feature_space.saving_lib",
        "src.layers.preprocessing.feature_space.serialization_lib",
        "src.layers.preprocessing.feature_space.KerasSaveable",
        "src.layers.preprocessing.feature_space.backend_utils",
        "src.layers.preprocessing.feature_space.tf",
        "src.layers.preprocessing.feature_space.auto_name",
        "src.layers.preprocessing.feature_space.Cross",
        "src.layers.preprocessing.feature_space.Feature",
        "src.layers.preprocessing.feature_space.FeatureSpace",
        "src.layers.preprocessing.feature_space.TFDConcat",
        "src.layers.preprocessing.feature_space.TFDIdentity",
        "src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer.backend_config",
        "src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer.DataLayer",
        "src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer.densify_bounding_boxes",
        "src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer.BaseImagePreprocessingLayer",
        "src.layers.preprocessing.image_preprocessing.max_num_bounding_box.keras_export",
        "src.layers.preprocessing.image_preprocessing.max_num_bounding_box.BaseImagePreprocessingLayer",
        "src.layers.preprocessing.image_preprocessing.max_num_bounding_box.MaxNumBoundingBoxes",
        "src.layers.preprocessing.image_preprocessing.auto_contrast.backend",
        "src.layers.preprocessing.image_preprocessing.auto_contrast.keras_export",
        "src.layers.preprocessing.image_preprocessing.auto_contrast.BaseImagePreprocessingLayer",
        "src.layers.preprocessing.image_preprocessing.auto_contrast.AutoContrast",
        "src.layers.preprocessing.image_preprocessing.mix_up.ops",
        "src.layers.preprocessing.image_preprocessing.mix_up.keras_export",
        "src.layers.preprocessing.image_preprocessing.mix_up.BaseImagePreprocessingLayer",
        "src.layers.preprocessing.image_preprocessing.mix_up.SeedGenerator",
        "src.layers.preprocessing.image_preprocessing.mix_up.backend_utils",
        "src.layers.preprocessing.image_preprocessing.mix_up.MixUp",
        "src.layers.preprocessing.image_preprocessing.random_posterization.keras_export",
        "src.layers.preprocessing.image_preprocessing.random_posterization.BaseImagePreprocessingLayer",
        "src.layers.preprocessing.image_preprocessing.random_posterization.RandomPosterization",
        "src.layers.preprocessing.image_preprocessing.random_perspective.keras_export",
        "src.layers.preprocessing.image_preprocessing.random_perspective.BaseImagePreprocessingLayer",
        "src.layers.preprocessing.image_preprocessing.random_perspective.clip_to_image_size",
        "src.layers.preprocessing.image_preprocessing.random_perspective.convert_format",
        "src.layers.preprocessing.image_preprocessing.random_perspective.SeedGenerator",
        "src.layers.preprocessing.image_preprocessing.random_perspective.backend_utils",
        "src.layers.preprocessing.image_preprocessing.random_perspective.RandomPerspective",
        "src.layers.preprocessing.image_preprocessing.random_zoom.backend",
        "src.layers.preprocessing.image_preprocessing.random_zoom.keras_export",
        "src.layers.preprocessing.image_preprocessing.random_zoom.BaseImagePreprocessingLayer",
        "src.layers.preprocessing.image_preprocessing.random_zoom.clip_to_image_size",
        "src.layers.preprocessing.image_preprocessing.random_zoom.convert_format",
        "src.layers.preprocessing.image_preprocessing.random_zoom.SeedGenerator",
        "src.layers.preprocessing.image_preprocessing.random_zoom.backend_utils",
        "src.layers.preprocessing.image_preprocessing.random_zoom.RandomZoom",
        "src.layers.preprocessing.image_preprocessing.random_grayscale.backend",
        "src.layers.preprocessing.image_preprocessing.random_grayscale.keras_export",
        "src.layers.preprocessing.image_preprocessing.random_grayscale.BaseImagePreprocessingLayer",
        "src.layers.preprocessing.image_preprocessing.random_grayscale.RandomGrayscale",
        "src.layers.preprocessing.image_preprocessing.random_color_jitter.random_brightness",
        "src.layers.preprocessing.image_preprocessing.random_color_jitter.random_contrast",
        "src.layers.preprocessing.image_preprocessing.random_color_jitter.random_hue",
        "src.layers.preprocessing.image_preprocessing.random_color_jitter.random_saturation",
        "src.layers.preprocessing.image_preprocessing.random_color_jitter.keras_export",
        "src.layers.preprocessing.image_preprocessing.random_color_jitter.BaseImagePreprocessingLayer",
        "src.layers.preprocessing.image_preprocessing.random_color_jitter.SeedGenerator",
        "src.layers.preprocessing.image_preprocessing.random_color_jitter.backend_utils",
        "src.layers.preprocessing.image_preprocessing.random_color_jitter.RandomColorJitter",
        "src.layers.preprocessing.image_preprocessing.random_contrast.keras_export",
        "src.layers.preprocessing.image_preprocessing.random_contrast.BaseImagePreprocessingLayer",
        "src.layers.preprocessing.image_preprocessing.random_contrast.SeedGenerator",
        "src.layers.preprocessing.image_preprocessing.random_contrast.RandomContrast",
        "src.layers.preprocessing.image_preprocessing.center_crop.keras_export",
        "src.layers.preprocessing.image_preprocessing.center_crop.BaseImagePreprocessingLayer",
        "src.layers.preprocessing.image_preprocessing.center_crop.clip_to_image_size",
        "src.layers.preprocessing.image_preprocessing.center_crop.convert_format",
        "src.layers.preprocessing.image_preprocessing.center_crop.image_utils",
        "src.layers.preprocessing.image_preprocessing.center_crop.CenterCrop",
        "src.layers.preprocessing.image_preprocessing.random_brightness.keras_export",
        "src.layers.preprocessing.image_preprocessing.random_brightness.BaseImagePreprocessingLayer",
        "src.layers.preprocessing.image_preprocessing.random_brightness.SeedGenerator",
        "src.layers.preprocessing.image_preprocessing.random_brightness.RandomBrightness",
        "src.layers.preprocessing.image_preprocessing.random_color_degeneration.keras_export",
        "src.layers.preprocessing.image_preprocessing.random_color_degeneration.BaseImagePreprocessingLayer",
        "src.layers.preprocessing.image_preprocessing.random_color_degeneration.SeedGenerator",
        "src.layers.preprocessing.image_preprocessing.random_color_degeneration.RandomColorDegeneration",
        "src.layers.preprocessing.image_preprocessing.rand_augment.layers",
        "src.layers.preprocessing.image_preprocessing.rand_augment.keras_export",
        "src.layers.preprocessing.image_preprocessing.rand_augment.BaseImagePreprocessingLayer",
        "src.layers.preprocessing.image_preprocessing.rand_augment.SeedGenerator",
        "src.layers.preprocessing.image_preprocessing.rand_augment.backend_utils",
        "src.layers.preprocessing.image_preprocessing.rand_augment.RandAugment",
        "src.layers.preprocessing.image_preprocessing.resizing.backend",
        "src.layers.preprocessing.image_preprocessing.resizing.keras_export",
        "src.layers.preprocessing.image_preprocessing.resizing.BaseImagePreprocessingLayer",
        "src.layers.preprocessing.image_preprocessing.resizing.clip_to_image_size",
        "src.layers.preprocessing.image_preprocessing.resizing.convert_format",
        "src.layers.preprocessing.image_preprocessing.resizing.Resizing",
        "src.layers.preprocessing.image_preprocessing.random_rotation.keras_export",
        "src.layers.preprocessing.image_preprocessing.random_rotation.BaseImagePreprocessingLayer",
        "src.layers.preprocessing.image_preprocessing.random_rotation.converters",
        "src.layers.preprocessing.image_preprocessing.random_rotation.SeedGenerator",
        "src.layers.preprocessing.image_preprocessing.random_rotation.RandomRotation",
        "src.layers.preprocessing.image_preprocessing.solarization.backend",
        "src.layers.preprocessing.image_preprocessing.solarization.keras_export",
        "src.layers.preprocessing.image_preprocessing.solarization.BaseImagePreprocessingLayer",
        "src.layers.preprocessing.image_preprocessing.solarization.SeedGenerator",
        "src.layers.preprocessing.image_preprocessing.solarization.Solarization",
        "src.layers.preprocessing.image_preprocessing.random_gaussian_blur.keras_export",
        "src.layers.preprocessing.image_preprocessing.random_gaussian_blur.BaseImagePreprocessingLayer",
        "src.layers.preprocessing.image_preprocessing.random_gaussian_blur.SeedGenerator",
        "src.layers.preprocessing.image_preprocessing.random_gaussian_blur.RandomGaussianBlur",
        "src.layers.preprocessing.image_preprocessing.cut_mix.keras_export",
        "src.layers.preprocessing.image_preprocessing.cut_mix.BaseImagePreprocessingLayer",
        "src.layers.preprocessing.image_preprocessing.cut_mix.SeedGenerator",
        "src.layers.preprocessing.image_preprocessing.cut_mix.CutMix",
        "src.layers.preprocessing.image_preprocessing.random_saturation.keras_export",
        "src.layers.preprocessing.image_preprocessing.random_saturation.epsilon",
        "src.layers.preprocessing.image_preprocessing.random_saturation.BaseImagePreprocessingLayer",
        "src.layers.preprocessing.image_preprocessing.random_saturation.SeedGenerator",
        "src.layers.preprocessing.image_preprocessing.random_saturation.RandomSaturation",
        "src.layers.preprocessing.image_preprocessing.random_translation.keras_export",
        "src.layers.preprocessing.image_preprocessing.random_translation.BaseImagePreprocessingLayer",
        "src.layers.preprocessing.image_preprocessing.random_translation.clip_to_image_size",
        "src.layers.preprocessing.image_preprocessing.random_translation.convert_format",
        "src.layers.preprocessing.image_preprocessing.random_translation.SeedGenerator",
        "src.layers.preprocessing.image_preprocessing.random_translation.backend_utils",
        "src.layers.preprocessing.image_preprocessing.random_translation.RandomTranslation",
        "src.layers.preprocessing.image_preprocessing.random_elastic_transform.keras_export",
        "src.layers.preprocessing.image_preprocessing.random_elastic_transform.BaseImagePreprocessingLayer",
        "src.layers.preprocessing.image_preprocessing.random_elastic_transform.SeedGenerator",
        "src.layers.preprocessing.image_preprocessing.random_elastic_transform.RandomElasticTransform",
        "src.layers.preprocessing.image_preprocessing.random_shear.keras_export",
        "src.layers.preprocessing.image_preprocessing.random_shear.BaseImagePreprocessingLayer",
        "src.layers.preprocessing.image_preprocessing.random_shear.clip_to_image_size",
        "src.layers.preprocessing.image_preprocessing.random_shear.convert_format",
        "src.layers.preprocessing.image_preprocessing.random_shear.SeedGenerator",
        "src.layers.preprocessing.image_preprocessing.random_shear.backend_utils",
        "src.layers.preprocessing.image_preprocessing.random_shear.RandomShear",
        "src.layers.preprocessing.image_preprocessing.random_sharpness.keras_export",
        "src.layers.preprocessing.image_preprocessing.random_sharpness.BaseImagePreprocessingLayer",
        "src.layers.preprocessing.image_preprocessing.random_sharpness.SeedGenerator",
        "src.layers.preprocessing.image_preprocessing.random_sharpness.RandomSharpness",
        "src.layers.preprocessing.image_preprocessing.random_flip.keras_export",
        "src.layers.preprocessing.image_preprocessing.random_flip.BaseImagePreprocessingLayer",
        "src.layers.preprocessing.image_preprocessing.random_flip.clip_to_image_size",
        "src.layers.preprocessing.image_preprocessing.random_flip.convert_format",
        "src.layers.preprocessing.image_preprocessing.random_flip.SeedGenerator",
        "src.layers.preprocessing.image_preprocessing.random_flip.backend_utils",
        "src.layers.preprocessing.image_preprocessing.random_flip.HORIZONTAL",
        "src.layers.preprocessing.image_preprocessing.random_flip.VERTICAL",
        "src.layers.preprocessing.image_preprocessing.random_flip.HORIZONTAL_AND_VERTICAL",
        "src.layers.preprocessing.image_preprocessing.random_flip.RandomFlip",
        "src.layers.preprocessing.image_preprocessing.aug_mix.layers",
        "src.layers.preprocessing.image_preprocessing.aug_mix.keras_export",
        "src.layers.preprocessing.image_preprocessing.aug_mix.BaseImagePreprocessingLayer",
        "src.layers.preprocessing.image_preprocessing.aug_mix.SeedGenerator",
        "src.layers.preprocessing.image_preprocessing.aug_mix.backend_utils",
        "src.layers.preprocessing.image_preprocessing.aug_mix.AUGMENT_LAYERS_ALL",
        "src.layers.preprocessing.image_preprocessing.aug_mix.AUGMENT_LAYERS",
        "src.layers.preprocessing.image_preprocessing.aug_mix.AugMix",
        "src.layers.preprocessing.image_preprocessing.random_erasing.keras_export",
        "src.layers.preprocessing.image_preprocessing.random_erasing.BaseImagePreprocessingLayer",
        "src.layers.preprocessing.image_preprocessing.random_erasing.SeedGenerator",
        "src.layers.preprocessing.image_preprocessing.random_erasing.RandomErasing",
        "src.layers.preprocessing.image_preprocessing.random_hue.keras_export",
        "src.layers.preprocessing.image_preprocessing.random_hue.BaseImagePreprocessingLayer",
        "src.layers.preprocessing.image_preprocessing.random_hue.RandomHue",
        "src.layers.preprocessing.image_preprocessing.equalization.backend",
        "src.layers.preprocessing.image_preprocessing.equalization.keras_export",
        "src.layers.preprocessing.image_preprocessing.equalization.BaseImagePreprocessingLayer",
        "src.layers.preprocessing.image_preprocessing.equalization.Equalization",
        "src.layers.preprocessing.image_preprocessing.random_crop.backend",
        "src.layers.preprocessing.image_preprocessing.random_crop.keras_export",
        "src.layers.preprocessing.image_preprocessing.random_crop.BaseImagePreprocessingLayer",
        "src.layers.preprocessing.image_preprocessing.random_crop.convert_format",
        "src.layers.preprocessing.image_preprocessing.random_crop.densify_bounding_boxes",
        "src.layers.preprocessing.image_preprocessing.random_crop.SeedGenerator",
        "src.layers.preprocessing.image_preprocessing.random_crop.RandomCrop",
        "src.layers.preprocessing.image_preprocessing.random_invert.keras_export",
        "src.layers.preprocessing.image_preprocessing.random_invert.BaseImagePreprocessingLayer",
        "src.layers.preprocessing.image_preprocessing.random_invert.RandomInvert",
        "src.layers.preprocessing.image_preprocessing.bounding_boxes.bounding_box.backend_utils",
        "src.layers.preprocessing.image_preprocessing.bounding_boxes.bounding_box.SUPPORTED_FORMATS",
        "src.layers.preprocessing.image_preprocessing.bounding_boxes.bounding_box.BoundingBox",
        "src.layers.preprocessing.image_preprocessing.bounding_boxes.formats.XYXY",
        "src.layers.preprocessing.image_preprocessing.bounding_boxes.formats.REL_XYXY",
        "src.layers.preprocessing.image_preprocessing.bounding_boxes.formats.CENTER_XYWH",
        "src.layers.preprocessing.image_preprocessing.bounding_boxes.formats.XYWH",
        "src.layers.preprocessing.image_preprocessing.bounding_boxes.formats.REL_XYWH",
        "src.layers.preprocessing.image_preprocessing.bounding_boxes.formats.YXYX",
        "src.layers.preprocessing.image_preprocessing.bounding_boxes.formats.REL_YXYX",
        "src.layers.preprocessing.image_preprocessing.bounding_boxes.converters.backend",
        "src.layers.preprocessing.image_preprocessing.bounding_boxes.converters.ops",
        "src.layers.preprocessing.image_preprocessing.bounding_boxes.converters.keras_export",
        "src.layers.preprocessing.image_preprocessing.bounding_boxes.converters.BoundingBox",
        "src.layers.preprocessing.image_preprocessing.bounding_boxes.converters.backend_utils",
        "src.layers.preprocessing.image_preprocessing.bounding_boxes.converters.convert_format",
        "src.layers.preprocessing.image_preprocessing.bounding_boxes.converters.clip_to_image_size",
        "src.layers.preprocessing.image_preprocessing.bounding_boxes.converters.affine_transform",
        "src.layers.preprocessing.image_preprocessing.bounding_boxes.converters.crop",
        "src.layers.preprocessing.image_preprocessing.bounding_boxes.converters.pad",
        "src.layers.preprocessing.image_preprocessing.bounding_boxes.converters.encode_box_to_deltas",
        "src.layers.preprocessing.image_preprocessing.bounding_boxes.converters.decode_deltas_to_boxes",
        "src.layers.preprocessing.image_preprocessing.bounding_boxes.iou.backend",
        "src.layers.preprocessing.image_preprocessing.bounding_boxes.iou.ops",
        "src.layers.preprocessing.image_preprocessing.bounding_boxes.iou.keras_export",
        "src.layers.preprocessing.image_preprocessing.bounding_boxes.iou.converters",
        "src.layers.preprocessing.image_preprocessing.bounding_boxes.iou.compute_iou",
        "src.layers.preprocessing.image_preprocessing.bounding_boxes.iou.compute_ciou",
        "src.layers.preprocessing.image_preprocessing.bounding_boxes.validation.current_backend",
        "src.layers.preprocessing.image_preprocessing.bounding_boxes.validation.tf_utils",
        "src.layers.preprocessing.image_preprocessing.bounding_boxes.validation.densify_bounding_boxes",
        "src.layers.preprocessing.image_preprocessing.bounding_boxes.validation.validate_bounding_boxes",
        "src.layers.merging.concatenate.ops",
        "src.layers.merging.concatenate.keras_export",
        "src.layers.merging.concatenate.Merge",
        "src.layers.merging.concatenate.Concatenate",
        "src.layers.merging.concatenate.concatenate",
        "src.layers.merging.add.ops",
        "src.layers.merging.add.keras_export",
        "src.layers.merging.add.Merge",
        "src.layers.merging.add.Add",
        "src.layers.merging.add.add",
        "src.layers.merging.subtract.ops",
        "src.layers.merging.subtract.keras_export",
        "src.layers.merging.subtract.Merge",
        "src.layers.merging.subtract.Subtract",
        "src.layers.merging.subtract.subtract",
        "src.layers.merging.multiply.backend",
        "src.layers.merging.multiply.ops",
        "src.layers.merging.multiply.keras_export",
        "src.layers.merging.multiply.Merge",
        "src.layers.merging.multiply.Multiply",
        "src.layers.merging.multiply.multiply",
        "src.layers.merging.maximum.ops",
        "src.layers.merging.maximum.keras_export",
        "src.layers.merging.maximum.Merge",
        "src.layers.merging.maximum.Maximum",
        "src.layers.merging.maximum.maximum",
        "src.layers.merging.minimum.ops",
        "src.layers.merging.minimum.keras_export",
        "src.layers.merging.minimum.Merge",
        "src.layers.merging.minimum.Minimum",
        "src.layers.merging.minimum.minimum",
        "src.layers.merging.dot.ops",
        "src.layers.merging.dot.keras_export",
        "src.layers.merging.dot.Merge",
        "src.layers.merging.dot.normalize",
        "src.layers.merging.dot.batch_dot",
        "src.layers.merging.dot.Dot",
        "src.layers.merging.dot.dot",
        "src.layers.merging.average.ops",
        "src.layers.merging.average.keras_export",
        "src.layers.merging.average.Merge",
        "src.layers.merging.average.Average",
        "src.layers.merging.average.average",
        "src.layers.merging.base_merge.backend",
        "src.layers.merging.base_merge.ops",
        "src.layers.merging.base_merge.KerasTensor",
        "src.layers.merging.base_merge.Layer",
        "src.layers.merging.base_merge.Merge",
        "src.layers.rnn.time_distributed.backend",
        "src.layers.rnn.time_distributed.ops",
        "src.layers.rnn.time_distributed.keras_export",
        "src.layers.rnn.time_distributed.Wrapper",
        "src.layers.rnn.time_distributed.Layer",
        "src.layers.rnn.time_distributed.TimeDistributed",
        "src.layers.rnn.bidirectional.ops",
        "src.layers.rnn.bidirectional.keras_export",
        "src.layers.rnn.bidirectional.Layer",
        "src.layers.rnn.bidirectional.serialization_lib",
        "src.layers.rnn.bidirectional.Bidirectional",
        "src.layers.rnn.conv_lstm3d.keras_export",
        "src.layers.rnn.conv_lstm3d.ConvLSTM",
        "src.layers.rnn.conv_lstm3d.ConvLSTM3D",
        "src.layers.rnn.stacked_rnn_cells.ops",
        "src.layers.rnn.stacked_rnn_cells.tree",
        "src.layers.rnn.stacked_rnn_cells.keras_export",
        "src.layers.rnn.stacked_rnn_cells.Layer",
        "src.layers.rnn.stacked_rnn_cells.serialization_lib",
        "src.layers.rnn.stacked_rnn_cells.StackedRNNCells",
        "src.layers.rnn.conv_lstm2d.keras_export",
        "src.layers.rnn.conv_lstm2d.ConvLSTM",
        "src.layers.rnn.conv_lstm2d.ConvLSTM2D",
        "src.layers.rnn.lstm.activations",
        "src.layers.rnn.lstm.backend",
        "src.layers.rnn.lstm.constraints",
        "src.layers.rnn.lstm.initializers",
        "src.layers.rnn.lstm.ops",
        "src.layers.rnn.lstm.regularizers",
        "src.layers.rnn.lstm.tree",
        "src.layers.rnn.lstm.keras_export",
        "src.layers.rnn.lstm.InputSpec",
        "src.layers.rnn.lstm.Layer",
        "src.layers.rnn.lstm.DropoutRNNCell",
        "src.layers.rnn.lstm.RNN",
        "src.layers.rnn.lstm.LSTMCell",
        "src.layers.rnn.lstm.LSTM",
        "src.layers.rnn.gru.activations",
        "src.layers.rnn.gru.backend",
        "src.layers.rnn.gru.constraints",
        "src.layers.rnn.gru.initializers",
        "src.layers.rnn.gru.ops",
        "src.layers.rnn.gru.regularizers",
        "src.layers.rnn.gru.tree",
        "src.layers.rnn.gru.keras_export",
        "src.layers.rnn.gru.InputSpec",
        "src.layers.rnn.gru.Layer",
        "src.layers.rnn.gru.DropoutRNNCell",
        "src.layers.rnn.gru.RNN",
        "src.layers.rnn.gru.GRUCell",
        "src.layers.rnn.gru.GRU",
        "src.layers.rnn.conv_lstm1d.keras_export",
        "src.layers.rnn.conv_lstm1d.ConvLSTM",
        "src.layers.rnn.conv_lstm1d.ConvLSTM1D",
        "src.layers.rnn.rnn.backend",
        "src.layers.rnn.rnn.ops",
        "src.layers.rnn.rnn.tree",
        "src.layers.rnn.rnn.keras_export",
        "src.layers.rnn.rnn.Layer",
        "src.layers.rnn.rnn.DropoutRNNCell",
        "src.layers.rnn.rnn.StackedRNNCells",
        "src.layers.rnn.rnn.serialization_lib",
        "src.layers.rnn.rnn.tracking",
        "src.layers.rnn.rnn.RNN",
        "src.layers.rnn.dropout_rnn_cell.backend",
        "src.layers.rnn.dropout_rnn_cell.ops",
        "src.layers.rnn.dropout_rnn_cell.DropoutRNNCell",
        "src.layers.rnn.simple_rnn.activations",
        "src.layers.rnn.simple_rnn.backend",
        "src.layers.rnn.simple_rnn.constraints",
        "src.layers.rnn.simple_rnn.initializers",
        "src.layers.rnn.simple_rnn.ops",
        "src.layers.rnn.simple_rnn.regularizers",
        "src.layers.rnn.simple_rnn.keras_export",
        "src.layers.rnn.simple_rnn.InputSpec",
        "src.layers.rnn.simple_rnn.Layer",
        "src.layers.rnn.simple_rnn.DropoutRNNCell",
        "src.layers.rnn.simple_rnn.RNN",
        "src.layers.rnn.simple_rnn.SimpleRNNCell",
        "src.layers.rnn.simple_rnn.SimpleRNN",
        "src.layers.rnn.conv_lstm.activations",
        "src.layers.rnn.conv_lstm.backend",
        "src.layers.rnn.conv_lstm.constraints",
        "src.layers.rnn.conv_lstm.initializers",
        "src.layers.rnn.conv_lstm.ops",
        "src.layers.rnn.conv_lstm.regularizers",
        "src.layers.rnn.conv_lstm.tree",
        "src.layers.rnn.conv_lstm.InputSpec",
        "src.layers.rnn.conv_lstm.Layer",
        "src.layers.rnn.conv_lstm.DropoutRNNCell",
        "src.layers.rnn.conv_lstm.RNN",
        "src.layers.rnn.conv_lstm.operation_utils",
        "src.layers.rnn.conv_lstm.argument_validation",
        "src.layers.rnn.conv_lstm.ConvLSTMCell",
        "src.layers.rnn.conv_lstm.ConvLSTM",
        "src.constraints.keras_export",
        "src.constraints.Constraint",
        "src.constraints.MaxNorm",
        "src.constraints.MinMaxNorm",
        "src.constraints.NonNeg",
        "src.constraints.UnitNorm",
        "src.constraints.serialization_lib",
        "src.constraints.to_snake_case",
        "src.constraints.ALL_OBJECTS",
        "src.constraints.ALL_OBJECTS_DICT",
        "src.constraints.serialize",
        "src.constraints.deserialize",
        "src.constraints.get",
        "src.constraints.constraints.backend",
        "src.constraints.constraints.ops",
        "src.constraints.constraints.keras_export",
        "src.constraints.constraints.Constraint",
        "src.constraints.constraints.MaxNorm",
        "src.constraints.constraints.NonNeg",
        "src.constraints.constraints.UnitNorm",
        "src.constraints.constraints.MinMaxNorm",
        "src.callbacks.BackupAndRestore",
        "src.callbacks.Callback",
        "src.callbacks.CallbackList",
        "src.callbacks.CSVLogger",
        "src.callbacks.EarlyStopping",
        "src.callbacks.History",
        "src.callbacks.LambdaCallback",
        "src.callbacks.LearningRateScheduler",
        "src.callbacks.ModelCheckpoint",
        "src.callbacks.MonitorCallback",
        "src.callbacks.OrbaxCheckpoint",
        "src.callbacks.ProgbarLogger",
        "src.callbacks.ReduceLROnPlateau",
        "src.callbacks.RemoteMonitor",
        "src.callbacks.SwapEMAWeights",
        "src.callbacks.TensorBoard",
        "src.callbacks.TerminateOnNaN",
        "src.callbacks.callback.backend",
        "src.callbacks.callback.keras_export",
        "src.callbacks.callback.Callback",
        "src.callbacks.callback_list.backend",
        "src.callbacks.callback_list.tree",
        "src.callbacks.callback_list.keras_export",
        "src.callbacks.callback_list.Callback",
        "src.callbacks.callback_list.History",
        "src.callbacks.callback_list.ProgbarLogger",
        "src.callbacks.callback_list.python_utils",
        "src.callbacks.callback_list.CallbackList",
        "src.callbacks.remote_monitor.keras_export",
        "src.callbacks.remote_monitor.Callback",
        "src.callbacks.remote_monitor.RemoteMonitor",
        "src.callbacks.backup_and_restore.keras_export",
        "src.callbacks.backup_and_restore.Callback",
        "src.callbacks.backup_and_restore.file_utils",
        "src.callbacks.backup_and_restore.BackupAndRestore",
        "src.callbacks.lambda_callback.keras_export",
        "src.callbacks.lambda_callback.Callback",
        "src.callbacks.lambda_callback.LambdaCallback",
        "src.callbacks.swap_ema_weights.backend",
        "src.callbacks.swap_ema_weights.ops",
        "src.callbacks.swap_ema_weights.keras_export",
        "src.callbacks.swap_ema_weights.Callback",
        "src.callbacks.swap_ema_weights.SwapEMAWeights",
        "src.callbacks.terminate_on_nan.keras_export",
        "src.callbacks.terminate_on_nan.Callback",
        "src.callbacks.terminate_on_nan.io_utils",
        "src.callbacks.terminate_on_nan.TerminateOnNaN",
        "src.callbacks.reduce_lr_on_plateau.backend",
        "src.callbacks.reduce_lr_on_plateau.keras_export",
        "src.callbacks.reduce_lr_on_plateau.MonitorCallback",
        "src.callbacks.reduce_lr_on_plateau.io_utils",
        "src.callbacks.reduce_lr_on_plateau.ReduceLROnPlateau",
        "src.callbacks.orbax_checkpoint.backend",
        "src.callbacks.orbax_checkpoint.tree",
        "src.callbacks.orbax_checkpoint.MonitorCallback",
        "src.callbacks.orbax_checkpoint.print_msg",
        "src.callbacks.orbax_checkpoint.ocp",
        "src.callbacks.orbax_checkpoint.OrbaxCheckpoint",
        "src.callbacks.model_checkpoint.backend",
        "src.callbacks.model_checkpoint.keras_export",
        "src.callbacks.model_checkpoint.MonitorCallback",
        "src.callbacks.model_checkpoint.file_utils",
        "src.callbacks.model_checkpoint.io_utils",
        "src.callbacks.model_checkpoint.ModelCheckpoint",
        "src.callbacks.csv_logger.keras_export",
        "src.callbacks.csv_logger.Callback",
        "src.callbacks.csv_logger.file_utils",
        "src.callbacks.csv_logger.CSVLogger",
        "src.callbacks.early_stopping.keras_export",
        "src.callbacks.early_stopping.MonitorCallback",
        "src.callbacks.early_stopping.io_utils",
        "src.callbacks.early_stopping.EarlyStopping",
        "src.callbacks.learning_rate_scheduler.backend",
        "src.callbacks.learning_rate_scheduler.keras_export",
        "src.callbacks.learning_rate_scheduler.Callback",
        "src.callbacks.learning_rate_scheduler.io_utils",
        "src.callbacks.learning_rate_scheduler.LearningRateScheduler",
        "src.callbacks.monitor_callback.ops",
        "src.callbacks.monitor_callback.Callback",
        "src.callbacks.monitor_callback.compile_utils",
        "src.callbacks.monitor_callback.MonitorCallback",
        "src.callbacks.progbar_logger.keras_export",
        "src.callbacks.progbar_logger.Callback",
        "src.callbacks.progbar_logger.io_utils",
        "src.callbacks.progbar_logger.Progbar",
        "src.callbacks.progbar_logger.ProgbarLogger",
        "src.callbacks.tensorboard.backend",
        "src.callbacks.tensorboard.ops",
        "src.callbacks.tensorboard.tree",
        "src.callbacks.tensorboard.keras_export",
        "src.callbacks.tensorboard.Callback",
        "src.callbacks.tensorboard.Embedding",
        "src.callbacks.tensorboard.Optimizer",
        "src.callbacks.tensorboard.file_utils",
        "src.callbacks.tensorboard.TensorBoard",
        "src.callbacks.tensorboard.keras_model_summary",
        "src.callbacks.history.keras_export",
        "src.callbacks.history.Callback",
        "src.callbacks.history.History",
        "src.quantizers.keras_export",
        "src.quantizers.Float8QuantizationConfig",
        "src.quantizers.Int4QuantizationConfig",
        "src.quantizers.Int8QuantizationConfig",
        "src.quantizers.QuantizationConfig",
        "src.quantizers.AbsMaxQuantizer",
        "src.quantizers.Quantizer",
        "src.quantizers.abs_max_quantize",
        "src.quantizers.compute_float8_amax_history",
        "src.quantizers.compute_float8_scale",
        "src.quantizers.fake_quant_with_min_max_vars",
        "src.quantizers.pack_int4",
        "src.quantizers.quantize_and_dequantize",
        "src.quantizers.unpack_int4",
        "src.quantizers.serialization_lib",
        "src.quantizers.to_snake_case",
        "src.quantizers.ALL_OBJECTS",
        "src.quantizers.ALL_OBJECTS_DICT",
        "src.quantizers.serialize",
        "src.quantizers.deserialize",
        "src.quantizers.get",
        "src.quantizers.gptq_config.keras_export",
        "src.quantizers.gptq_config.QuantizationConfig",
        "src.quantizers.gptq_config.GPTQConfig",
        "src.quantizers.quantizers.backend",
        "src.quantizers.quantizers.ops",
        "src.quantizers.quantizers.keras_export",
        "src.quantizers.quantizers.KerasTensor",
        "src.quantizers.quantizers.any_symbolic_tensors",
        "src.quantizers.quantizers.canonicalize_axis",
        "src.quantizers.quantizers.standardize_axis_for_numpy",
        "src.quantizers.quantizers.Operation",
        "src.quantizers.quantizers.GPTQConfig",
        "src.quantizers.quantizers.Quantizer",
        "src.quantizers.quantizers.abs_max_quantize",
        "src.quantizers.quantizers.AbsMaxQuantizer",
        "src.quantizers.quantizers.adjust_and_nudge",
        "src.quantizers.quantizers.FakeQuantWithMinMaxVars",
        "src.quantizers.quantizers.fake_quant_with_min_max_vars",
        "src.quantizers.quantizers.compute_float8_scale",
        "src.quantizers.quantizers.compute_float8_amax_history",
        "src.quantizers.quantizers.quantize_and_dequantize",
        "src.quantizers.quantizers.pack_int4",
        "src.quantizers.quantizers.unpack_int4",
        "src.quantizers.quantizers.GPTQQuantizer",
        "src.quantizers.quantizers.compute_quantization_parameters",
        "src.quantizers.quantizers.quantize_with_zero_point",
        "src.quantizers.quantizers.dequantize_with_zero_point",
        "src.quantizers.quantizers.quantize_with_sz_map",
        "src.quantizers.quantizers.dequantize_with_sz_map",
        "src.quantizers.gptq.ops",
        "src.quantizers.gptq.quantizers",
        "src.quantizers.gptq.Dense",
        "src.quantizers.gptq.EinsumDense",
        "src.quantizers.gptq.linalg",
        "src.quantizers.gptq.GPTQConfig",
        "src.quantizers.gptq.GPTQQuantizer",
        "src.quantizers.gptq.compute_quantization_parameters",
        "src.quantizers.gptq.dequantize_with_zero_point",
        "src.quantizers.gptq.quantize_with_zero_point",
        "src.quantizers.gptq.gptq_quantize_matrix",
        "src.quantizers.gptq.GPTQ",
        "src.quantizers.gptq_core.ops",
        "src.quantizers.gptq_core.keras_utils",
        "src.quantizers.gptq_core.GPTQDTypePolicy",
        "src.quantizers.gptq_core.DTypePolicyMap",
        "src.quantizers.gptq_core.Dense",
        "src.quantizers.gptq_core.EinsumDense",
        "src.quantizers.gptq_core.GPTQ",
        "src.quantizers.gptq_core.GPTQConfig",
        "src.quantizers.gptq_core.should_quantize_layer",
        "src.quantizers.gptq_core.stream_hessians",
        "src.quantizers.gptq_core.get_dataloader",
        "src.quantizers.gptq_core.find_layers_in_block",
        "src.quantizers.gptq_core.apply_gptq_layerwise",
        "src.quantizers.gptq_core.gptq_quantize",
        "src.quantizers.gptq_core.get_group_size_for_layer",
        "src.quantizers.gptq_core.get_weight_bits_for_layer",
        "src.quantizers.quantization_config.keras_export",
        "src.quantizers.quantization_config.QUANTIZATION_MODES",
        "src.quantizers.quantization_config.serialization_lib",
        "src.quantizers.quantization_config.QuantizationConfig",
        "src.quantizers.quantization_config.Int8QuantizationConfig",
        "src.quantizers.quantization_config.Int4QuantizationConfig",
        "src.quantizers.quantization_config.Float8QuantizationConfig",
        "src.quantizers.quantization_config.validate_and_resolve_config",
        "src.datasets.cifar100.backend",
        "src.datasets.cifar100.keras_export",
        "src.datasets.cifar100.load_batch",
        "src.datasets.cifar100.get_file",
        "src.datasets.cifar100.load_data",
        "src.datasets.fashion_mnist.keras_export",
        "src.datasets.fashion_mnist.get_file",
        "src.datasets.fashion_mnist.load_data",
        "src.datasets.imdb.keras_export",
        "src.datasets.imdb.get_file",
        "src.datasets.imdb.remove_long_seq",
        "src.datasets.imdb.load_data",
        "src.datasets.imdb.get_word_index",
        "src.datasets.boston_housing.keras_export",
        "src.datasets.boston_housing.get_file",
        "src.datasets.boston_housing.load_data",
        "src.datasets.reuters.keras_export",
        "src.datasets.reuters.get_file",
        "src.datasets.reuters.remove_long_seq",
        "src.datasets.reuters.load_data",
        "src.datasets.reuters.get_word_index",
        "src.datasets.reuters.get_label_names",
        "src.datasets.cifar.load_batch",
        "src.datasets.cifar10.backend",
        "src.datasets.cifar10.keras_export",
        "src.datasets.cifar10.load_batch",
        "src.datasets.cifar10.get_file",
        "src.datasets.cifar10.load_data",
        "src.datasets.california_housing.keras_export",
        "src.datasets.california_housing.get_file",
        "src.datasets.california_housing.load_data",
        "src.datasets.mnist.keras_export",
        "src.datasets.mnist.get_file",
        "src.datasets.mnist.load_data",
        "src.distribution.DataParallel",
        "src.distribution.DeviceMesh",
        "src.distribution.Distribution",
        "src.distribution.LayoutMap",
        "src.distribution.ModelParallel",
        "src.distribution.TensorLayout",
        "src.distribution.distribute_tensor",
        "src.distribution.distribution",
        "src.distribution.initialize",
        "src.distribution.list_devices",
        "src.distribution.set_distribution",
        "src.distribution.distribution_lib.keras_export",
        "src.distribution.distribution_lib.KerasTensor",
        "src.distribution.distribution_lib.distribution_lib",
        "src.distribution.distribution_lib.global_state",
        "src.distribution.distribution_lib.DEFAULT_BATCH_DIM_NAME",
        "src.distribution.distribution_lib.GLOBAL_ATTRIBUTE_NAME",
        "src.distribution.distribution_lib.list_devices",
        "src.distribution.distribution_lib.get_device_count",
        "src.distribution.distribution_lib.initialize",
        "src.distribution.distribution_lib.DeviceMesh",
        "src.distribution.distribution_lib.TensorLayout",
        "src.distribution.distribution_lib.Distribution",
        "src.distribution.distribution_lib.DataParallel",
        "src.distribution.distribution_lib.ModelParallel",
        "src.distribution.distribution_lib.LayoutMap",
        "src.distribution.distribution_lib.distribute_tensor",
        "src.distribution.distribution_lib.distribution",
        "src.distribution.distribution_lib.set_distribution",
        "src.distillation.distillation_loss.tree",
        "src.distillation.distillation_loss.keras_export",
        "src.distillation.distillation_loss.serialization_lib",
        "src.distillation.distillation_loss.tracking",
        "src.distillation.distillation_loss.DistillationLoss",
        "src.distillation.distillation_loss.FeatureDistillation",
        "src.distillation.distillation_loss.LogitsDistillation",
        "src.distillation.distiller.tree",
        "src.distillation.distiller.keras_export",
        "src.distillation.distiller.Model",
        "src.distillation.distiller.serialization_lib",
        "src.distillation.distiller.Distiller",
        "src.backend.backend",
        "src.backend.torch.name_scope",
        "src.backend.torch.IS_THREAD_SAFE",
        "src.backend.torch.SUPPORTS_RAGGED_TENSORS",
        "src.backend.torch.SUPPORTS_SPARSE_TENSORS",
        "src.backend.torch.Variable",
        "src.backend.torch.cast",
        "src.backend.torch.compute_output_spec",
        "src.backend.torch.cond",
        "src.backend.torch.convert_to_numpy",
        "src.backend.torch.convert_to_tensor",
        "src.backend.torch.device_scope",
        "src.backend.torch.is_tensor",
        "src.backend.torch.random_seed_dtype",
        "src.backend.torch.scatter",
        "src.backend.torch.shape",
        "src.backend.torch.stop_gradient",
        "src.backend.torch.to_torch_dtype",
        "src.backend.torch.vectorized_map",
        "src.backend.torch.cudnn_ok",
        "src.backend.torch.gru",
        "src.backend.torch.lstm",
        "src.backend.torch.rnn.tree",
        "src.backend.torch.rnn.convert_to_tensor",
        "src.backend.torch.rnn.get_device",
        "src.backend.torch.rnn.rnn",
        "src.backend.torch.rnn.prepare_lstm_weights",
        "src.backend.torch.rnn.cudnn_ok",
        "src.backend.torch.rnn.lstm",
        "src.backend.torch.rnn.gru",
        "src.backend.torch.nn.backend",
        "src.backend.torch.nn.compute_conv_transpose_padding_args_for_torch",
        "src.backend.torch.nn.cast",
        "src.backend.torch.nn.convert_to_tensor",
        "src.backend.torch.nn.get_device",
        "src.backend.torch.nn.expand_dims",
        "src.backend.torch.nn.where",
        "src.backend.torch.nn.standardize_tuple",
        "src.backend.torch.nn.relu",
        "src.backend.torch.nn.relu6",
        "src.backend.torch.nn.sigmoid",
        "src.backend.torch.nn.sparse_sigmoid",
        "src.backend.torch.nn.tanh",
        "src.backend.torch.nn.tanh_shrink",
        "src.backend.torch.nn.softplus",
        "src.backend.torch.nn.softsign",
        "src.backend.torch.nn.soft_shrink",
        "src.backend.torch.nn.sparse_plus",
        "src.backend.torch.nn.silu",
        "src.backend.torch.nn.squareplus",
        "src.backend.torch.nn.log_sigmoid",
        "src.backend.torch.nn.leaky_relu",
        "src.backend.torch.nn.hard_sigmoid",
        "src.backend.torch.nn.hard_silu",
        "src.backend.torch.nn.elu",
        "src.backend.torch.nn.selu",
        "src.backend.torch.nn.gelu",
        "src.backend.torch.nn.celu",
        "src.backend.torch.nn.glu",
        "src.backend.torch.nn.hard_tanh",
        "src.backend.torch.nn.hard_shrink",
        "src.backend.torch.nn.threshold",
        "src.backend.torch.nn.softmax",
        "src.backend.torch.nn.log_softmax",
        "src.backend.torch.nn.sparsemax",
        "src.backend.torch.nn.max_pool",
        "src.backend.torch.nn.average_pool",
        "src.backend.torch.nn.adaptive_average_pool",
        "src.backend.torch.nn.adaptive_max_pool",
        "src.backend.torch.nn.conv",
        "src.backend.torch.nn.depthwise_conv",
        "src.backend.torch.nn.separable_conv",
        "src.backend.torch.nn.conv_transpose",
        "src.backend.torch.nn.one_hot",
        "src.backend.torch.nn.multi_hot",
        "src.backend.torch.nn.categorical_crossentropy",
        "src.backend.torch.nn.sparse_categorical_crossentropy",
        "src.backend.torch.nn.binary_crossentropy",
        "src.backend.torch.nn.moments",
        "src.backend.torch.nn.batch_normalization",
        "src.backend.torch.nn.ctc_loss",
        "src.backend.torch.nn.ctc_decode",
        "src.backend.torch.nn.psnr",
        "src.backend.torch.nn.dot_product_attention",
        "src.backend.torch.nn.unfold",
        "src.backend.torch.core.tree",
        "src.backend.torch.core.KerasVariable",
        "src.backend.torch.core.global_state",
        "src.backend.torch.core.standardize_dtype",
        "src.backend.torch.core.slice_along_axis",
        "src.backend.torch.core.result_type",
        "src.backend.torch.core.KerasTensor",
        "src.backend.torch.core.StatelessScope",
        "src.backend.torch.core.get_stateless_scope",
        "src.backend.torch.core.in_stateless_scope",
        "src.backend.torch.core.SymbolicScope",
        "src.backend.torch.core.floatx",
        "src.backend.torch.core.SUPPORTS_SPARSE_TENSORS",
        "src.backend.torch.core.SUPPORTS_RAGGED_TENSORS",
        "src.backend.torch.core.IS_THREAD_SAFE",
        "src.backend.torch.core.DEFAULT_DEVICE",
        "src.backend.torch.core.TORCH_DTYPES",
        "src.backend.torch.core.device_scope",
        "src.backend.torch.core.get_device",
        "src.backend.torch.core.to_torch_dtype",
        "src.backend.torch.core.Variable",
        "src.backend.torch.core.convert_to_tensor",
        "src.backend.torch.core.convert_to_numpy",
        "src.backend.torch.core.is_tensor",
        "src.backend.torch.core.shape",
        "src.backend.torch.core.cast",
        "src.backend.torch.core.compute_output_spec",
        "src.backend.torch.core.cond",
        "src.backend.torch.core.vectorized_map",
        "src.backend.torch.core.map",
        "src.backend.torch.core.scan",
        "src.backend.torch.core.associative_scan",
        "src.backend.torch.core.scatter",
        "src.backend.torch.core.scatter_update",
        "src.backend.torch.core.slice",
        "src.backend.torch.core.slice_update",
        "src.backend.torch.core.switch",
        "src.backend.torch.core.while_loop",
        "src.backend.torch.core.fori_loop",
        "src.backend.torch.core.stop_gradient",
        "src.backend.torch.core.unstack",
        "src.backend.torch.core.random_seed_dtype",
        "src.backend.torch.core.remat",
        "src.backend.torch.core.custom_gradient",
        "src.backend.torch.core.CustomGradientFunction",
        "src.backend.torch.numpy.KerasTensor",
        "src.backend.torch.numpy.config",
        "src.backend.torch.numpy.dtypes",
        "src.backend.torch.numpy.canonicalize_axis",
        "src.backend.torch.numpy.to_tuple_or_list",
        "src.backend.torch.numpy.vectorize_impl",
        "src.backend.torch.numpy.standardize_dtype",
        "src.backend.torch.numpy.cast",
        "src.backend.torch.numpy.convert_to_tensor",
        "src.backend.torch.numpy.get_device",
        "src.backend.torch.numpy.is_tensor",
        "src.backend.torch.numpy.to_torch_dtype",
        "src.backend.torch.numpy.TORCH_INT_TYPES",
        "src.backend.torch.numpy.rot90",
        "src.backend.torch.numpy.add",
        "src.backend.torch.numpy.einsum",
        "src.backend.torch.numpy.subtract",
        "src.backend.torch.numpy.matmul",
        "src.backend.torch.numpy.multiply",
        "src.backend.torch.numpy.mean",
        "src.backend.torch.numpy.max",
        "src.backend.torch.numpy.ones",
        "src.backend.torch.numpy.zeros",
        "src.backend.torch.numpy.zeros_like",
        "src.backend.torch.numpy.absolute",
        "src.backend.torch.numpy.abs",
        "src.backend.torch.numpy.all",
        "src.backend.torch.numpy.angle",
        "src.backend.torch.numpy.any",
        "src.backend.torch.numpy.amax",
        "src.backend.torch.numpy.amin",
        "src.backend.torch.numpy.append",
        "src.backend.torch.numpy.arange",
        "src.backend.torch.numpy.arccos",
        "src.backend.torch.numpy.arccosh",
        "src.backend.torch.numpy.arcsin",
        "src.backend.torch.numpy.arcsinh",
        "src.backend.torch.numpy.arctan",
        "src.backend.torch.numpy.arctan2",
        "src.backend.torch.numpy.arctanh",
        "src.backend.torch.numpy.argmax",
        "src.backend.torch.numpy.argmin",
        "src.backend.torch.numpy.argsort",
        "src.backend.torch.numpy.array",
        "src.backend.torch.numpy.view",
        "src.backend.torch.numpy.average",
        "src.backend.torch.numpy.bartlett",
        "src.backend.torch.numpy.hamming",
        "src.backend.torch.numpy.hanning",
        "src.backend.torch.numpy.heaviside",
        "src.backend.torch.numpy.kaiser",
        "src.backend.torch.numpy.bincount",
        "src.backend.torch.numpy.bitwise_and",
        "src.backend.torch.numpy.bitwise_invert",
        "src.backend.torch.numpy.bitwise_not",
        "src.backend.torch.numpy.bitwise_or",
        "src.backend.torch.numpy.bitwise_xor",
        "src.backend.torch.numpy.bitwise_left_shift",
        "src.backend.torch.numpy.left_shift",
        "src.backend.torch.numpy.bitwise_right_shift",
        "src.backend.torch.numpy.right_shift",
        "src.backend.torch.numpy.blackman",
        "src.backend.torch.numpy.broadcast_to",
        "src.backend.torch.numpy.cbrt",
        "src.backend.torch.numpy.ceil",
        "src.backend.torch.numpy.clip",
        "src.backend.torch.numpy.concatenate",
        "src.backend.torch.numpy.conjugate",
        "src.backend.torch.numpy.conj",
        "src.backend.torch.numpy.copy",
        "src.backend.torch.numpy.cos",
        "src.backend.torch.numpy.cosh",
        "src.backend.torch.numpy.count_nonzero",
        "src.backend.torch.numpy.cross",
        "src.backend.torch.numpy.cumprod",
        "src.backend.torch.numpy.cumsum",
        "src.backend.torch.numpy.deg2rad",
        "src.backend.torch.numpy.diag",
        "src.backend.torch.numpy.diagflat",
        "src.backend.torch.numpy.diagonal",
        "src.backend.torch.numpy.diff",
        "src.backend.torch.numpy.digitize",
        "src.backend.torch.numpy.dot",
        "src.backend.torch.numpy.empty",
        "src.backend.torch.numpy.empty_like",
        "src.backend.torch.numpy.equal",
        "src.backend.torch.numpy.exp",
        "src.backend.torch.numpy.exp2",
        "src.backend.torch.numpy.expand_dims",
        "src.backend.torch.numpy.expm1",
        "src.backend.torch.numpy.flip",
        "src.backend.torch.numpy.floor",
        "src.backend.torch.numpy.full",
        "src.backend.torch.numpy.full_like",
        "src.backend.torch.numpy.gcd",
        "src.backend.torch.numpy.greater",
        "src.backend.torch.numpy.greater_equal",
        "src.backend.torch.numpy.hstack",
        "src.backend.torch.numpy.hypot",
        "src.backend.torch.numpy.identity",
        "src.backend.torch.numpy.imag",
        "src.backend.torch.numpy.isclose",
        "src.backend.torch.numpy.isfinite",
        "src.backend.torch.numpy.isin",
        "src.backend.torch.numpy.isinf",
        "src.backend.torch.numpy.isnan",
        "src.backend.torch.numpy.isneginf",
        "src.backend.torch.numpy.isposinf",
        "src.backend.torch.numpy.isreal",
        "src.backend.torch.numpy.kron",
        "src.backend.torch.numpy.lcm",
        "src.backend.torch.numpy.ldexp",
        "src.backend.torch.numpy.less",
        "src.backend.torch.numpy.less_equal",
        "src.backend.torch.numpy.linspace",
        "src.backend.torch.numpy.log",
        "src.backend.torch.numpy.log10",
        "src.backend.torch.numpy.log1p",
        "src.backend.torch.numpy.log2",
        "src.backend.torch.numpy.logaddexp",
        "src.backend.torch.numpy.logaddexp2",
        "src.backend.torch.numpy.logical_and",
        "src.backend.torch.numpy.logical_not",
        "src.backend.torch.numpy.logical_or",
        "src.backend.torch.numpy.logspace",
        "src.backend.torch.numpy.maximum",
        "src.backend.torch.numpy.median",
        "src.backend.torch.numpy.meshgrid",
        "src.backend.torch.numpy.min",
        "src.backend.torch.numpy.minimum",
        "src.backend.torch.numpy.mod",
        "src.backend.torch.numpy.moveaxis",
        "src.backend.torch.numpy.nan_to_num",
        "src.backend.torch.numpy.ndim",
        "src.backend.torch.numpy.nonzero",
        "src.backend.torch.numpy.not_equal",
        "src.backend.torch.numpy.ones_like",
        "src.backend.torch.numpy.outer",
        "src.backend.torch.numpy.pad",
        "src.backend.torch.numpy.prod",
        "src.backend.torch.numpy.quantile",
        "src.backend.torch.numpy.ravel",
        "src.backend.torch.numpy.unravel_index",
        "src.backend.torch.numpy.real",
        "src.backend.torch.numpy.reciprocal",
        "src.backend.torch.numpy.repeat",
        "src.backend.torch.numpy.reshape",
        "src.backend.torch.numpy.roll",
        "src.backend.torch.numpy.searchsorted",
        "src.backend.torch.numpy.sign",
        "src.backend.torch.numpy.signbit",
        "src.backend.torch.numpy.sin",
        "src.backend.torch.numpy.sinh",
        "src.backend.torch.numpy.size",
        "src.backend.torch.numpy.sort",
        "src.backend.torch.numpy.split",
        "src.backend.torch.numpy.array_split",
        "src.backend.torch.numpy.stack",
        "src.backend.torch.numpy.std",
        "src.backend.torch.numpy.swapaxes",
        "src.backend.torch.numpy.take",
        "src.backend.torch.numpy.take_along_axis",
        "src.backend.torch.numpy.tan",
        "src.backend.torch.numpy.tanh",
        "src.backend.torch.numpy.tensordot",
        "src.backend.torch.numpy.round",
        "src.backend.torch.numpy.tile",
        "src.backend.torch.numpy.trace",
        "src.backend.torch.numpy.tri",
        "src.backend.torch.numpy.tril",
        "src.backend.torch.numpy.triu",
        "src.backend.torch.numpy.trunc",
        "src.backend.torch.numpy.vdot",
        "src.backend.torch.numpy.inner",
        "src.backend.torch.numpy.vstack",
        "src.backend.torch.numpy.vectorize",
        "src.backend.torch.numpy.where",
        "src.backend.torch.numpy.divide",
        "src.backend.torch.numpy.divide_no_nan",
        "src.backend.torch.numpy.true_divide",
        "src.backend.torch.numpy.power",
        "src.backend.torch.numpy.negative",
        "src.backend.torch.numpy.square",
        "src.backend.torch.numpy.sqrt",
        "src.backend.torch.numpy.squeeze",
        "src.backend.torch.numpy.transpose",
        "src.backend.torch.numpy.trapezoid",
        "src.backend.torch.numpy.vander",
        "src.backend.torch.numpy.var",
        "src.backend.torch.numpy.sum",
        "src.backend.torch.numpy.eye",
        "src.backend.torch.numpy.floor_divide",
        "src.backend.torch.numpy.logical_xor",
        "src.backend.torch.numpy.corrcoef",
        "src.backend.torch.numpy.correlate",
        "src.backend.torch.numpy.select",
        "src.backend.torch.numpy.slogdet",
        "src.backend.torch.numpy.argpartition",
        "src.backend.torch.numpy.histogram",
        "src.backend.torch.export.tree",
        "src.backend.torch.export.convert_spec_to_tensor",
        "src.backend.torch.export.tf",
        "src.backend.torch.export.torch_xla",
        "src.backend.torch.export.TorchExportArchive",
        "src.backend.torch.random.floatx",
        "src.backend.torch.random.convert_to_tensor",
        "src.backend.torch.random.get_device",
        "src.backend.torch.random.to_torch_dtype",
        "src.backend.torch.random.SeedGenerator",
        "src.backend.torch.random.draw_seed",
        "src.backend.torch.random.make_default_seed",
        "src.backend.torch.random.torch_seed_generator",
        "src.backend.torch.random.normal",
        "src.backend.torch.random.categorical",
        "src.backend.torch.random.uniform",
        "src.backend.torch.random.randint",
        "src.backend.torch.random.truncated_normal",
        "src.backend.torch.random.dropout",
        "src.backend.torch.random.shuffle",
        "src.backend.torch.random.gamma",
        "src.backend.torch.random.binomial",
        "src.backend.torch.random.beta",
        "src.backend.torch.linalg.config",
        "src.backend.torch.linalg.standardize_dtype",
        "src.backend.torch.linalg.dtypes",
        "src.backend.torch.linalg.cast",
        "src.backend.torch.linalg.convert_to_tensor",
        "src.backend.torch.linalg.cholesky",
        "src.backend.torch.linalg.cholesky_inverse",
        "src.backend.torch.linalg.det",
        "src.backend.torch.linalg.eig",
        "src.backend.torch.linalg.eigh",
        "src.backend.torch.linalg.inv",
        "src.backend.torch.linalg.lu_factor",
        "src.backend.torch.linalg.norm",
        "src.backend.torch.linalg.qr",
        "src.backend.torch.linalg.solve",
        "src.backend.torch.linalg.solve_triangular",
        "src.backend.torch.linalg.svd",
        "src.backend.torch.linalg.lstsq",
        "src.backend.torch.linalg.jvp",
        "src.backend.torch.layer.in_stateless_scope",
        "src.backend.torch.layer.Operation",
        "src.backend.torch.layer.TorchLayer",
        "src.backend.torch.math.config",
        "src.backend.torch.math.standardize_dtype",
        "src.backend.torch.math.dtypes",
        "src.backend.torch.math.cast",
        "src.backend.torch.math.convert_to_tensor",
        "src.backend.torch.math.get_device",
        "src.backend.torch.math.pad",
        "src.backend.torch.math.segment_sum",
        "src.backend.torch.math.segment_max",
        "src.backend.torch.math.top_k",
        "src.backend.torch.math.in_top_k",
        "src.backend.torch.math.logsumexp",
        "src.backend.torch.math.qr",
        "src.backend.torch.math.extract_sequences",
        "src.backend.torch.math.fft",
        "src.backend.torch.math.fft2",
        "src.backend.torch.math.ifft2",
        "src.backend.torch.math.rfft",
        "src.backend.torch.math.irfft",
        "src.backend.torch.math.stft",
        "src.backend.torch.math.istft",
        "src.backend.torch.math.rsqrt",
        "src.backend.torch.math.erf",
        "src.backend.torch.math.erfinv",
        "src.backend.torch.math.solve",
        "src.backend.torch.math.norm",
        "src.backend.torch.math.logdet",
        "src.backend.torch.trainer.backend",
        "src.backend.torch.trainer.callbacks_module",
        "src.backend.torch.trainer.optimizers_module",
        "src.backend.torch.trainer.tree",
        "src.backend.torch.trainer.config",
        "src.backend.torch.trainer.base_trainer",
        "src.backend.torch.trainer.array_slicing",
        "src.backend.torch.trainer.data_adapter_utils",
        "src.backend.torch.trainer.EpochIterator",
        "src.backend.torch.trainer.traceback_utils",
        "src.backend.torch.trainer.TorchTrainer",
        "src.backend.torch.trainer.TorchEpochIterator",
        "src.backend.torch.image.backend",
        "src.backend.torch.image.cast",
        "src.backend.torch.image.convert_to_tensor",
        "src.backend.torch.image.get_device",
        "src.backend.torch.image.to_torch_dtype",
        "src.backend.torch.image.draw_seed",
        "src.backend.torch.image.RESIZE_INTERPOLATIONS",
        "src.backend.torch.image.UNSUPPORTED_INTERPOLATIONS",
        "src.backend.torch.image.AFFINE_TRANSFORM_INTERPOLATIONS",
        "src.backend.torch.image.AFFINE_TRANSFORM_FILL_MODES",
        "src.backend.torch.image.SCALE_AND_TRANSLATE_METHODS",
        "src.backend.torch.image.rgb_to_grayscale",
        "src.backend.torch.image.rgb_to_hsv",
        "src.backend.torch.image.hsv_to_rgb",
        "src.backend.torch.image.resize",
        "src.backend.torch.image.affine_transform",
        "src.backend.torch.image.perspective_transform",
        "src.backend.torch.image.compute_homography_matrix",
        "src.backend.torch.image.map_coordinates",
        "src.backend.torch.image.gaussian_blur",
        "src.backend.torch.image.elastic_transform",
        "src.backend.torch.image.scale_and_translate",
        "src.backend.torch.optimizers.TorchOptimizer",
        "src.backend.torch.optimizers.torch_optimizer.optimizers",
        "src.backend.torch.optimizers.torch_optimizer.BaseOptimizer",
        "src.backend.torch.optimizers.torch_optimizer.torch_utils",
        "src.backend.torch.optimizers.torch_optimizer.TorchOptimizer",
        "src.backend.torch.optimizers.torch_adadelta.ops",
        "src.backend.torch.optimizers.torch_adadelta.optimizers",
        "src.backend.torch.optimizers.torch_adadelta.torch_parallel_optimizer",
        "src.backend.torch.optimizers.torch_adadelta.Adadelta",
        "src.backend.torch.optimizers.torch_sgd.optimizers",
        "src.backend.torch.optimizers.torch_sgd.torch_parallel_optimizer",
        "src.backend.torch.optimizers.torch_sgd.SGD",
        "src.backend.torch.optimizers.torch_rmsprop.ops",
        "src.backend.torch.optimizers.torch_rmsprop.optimizers",
        "src.backend.torch.optimizers.torch_rmsprop.torch_parallel_optimizer",
        "src.backend.torch.optimizers.torch_rmsprop.RMSprop",
        "src.backend.torch.optimizers.torch_lion.ops",
        "src.backend.torch.optimizers.torch_lion.optimizers",
        "src.backend.torch.optimizers.torch_lion.torch_parallel_optimizer",
        "src.backend.torch.optimizers.torch_lion.Lion",
        "src.backend.torch.optimizers.torch_nadam.ops",
        "src.backend.torch.optimizers.torch_nadam.optimizers",
        "src.backend.torch.optimizers.torch_nadam.core",
        "src.backend.torch.optimizers.torch_nadam.torch_parallel_optimizer",
        "src.backend.torch.optimizers.torch_nadam.Nadam",
        "src.backend.torch.optimizers.torch_adamax.ops",
        "src.backend.torch.optimizers.torch_adamax.optimizers",
        "src.backend.torch.optimizers.torch_adamax.torch_parallel_optimizer",
        "src.backend.torch.optimizers.torch_adamax.Adamax",
        "src.backend.torch.optimizers.torch_parallel_optimizer.BaseOptimizer",
        "src.backend.torch.optimizers.torch_parallel_optimizer.torch_utils",
        "src.backend.torch.optimizers.torch_parallel_optimizer.TorchParallelOptimizer",
        "src.backend.torch.optimizers.torch_adam.ops",
        "src.backend.torch.optimizers.torch_adam.optimizers",
        "src.backend.torch.optimizers.torch_adam.torch_parallel_optimizer",
        "src.backend.torch.optimizers.torch_adam.Adam",
        "src.backend.torch.optimizers.torch_adamw.optimizers",
        "src.backend.torch.optimizers.torch_adamw.torch_adam",
        "src.backend.torch.optimizers.torch_adamw.AdamW",
        "src.backend.torch.optimizers.torch_adagrad.ops",
        "src.backend.torch.optimizers.torch_adagrad.optimizers",
        "src.backend.torch.optimizers.torch_adagrad.torch_parallel_optimizer",
        "src.backend.torch.optimizers.torch_adagrad.Adagrad",
        "src.backend.keras_export",
        "src.backend.result_type",
        "src.backend.KerasTensor",
        "src.backend.any_symbolic_tensors",
        "src.backend.is_keras_tensor",
        "src.backend.get_keras_mask",
        "src.backend.set_keras_mask",
        "src.backend.StatelessScope",
        "src.backend.get_stateless_scope",
        "src.backend.in_stateless_scope",
        "src.backend.SymbolicScope",
        "src.backend.in_symbolic_scope",
        "src.backend.AutocastScope",
        "src.backend.Variable",
        "src.backend.get_autocast_scope",
        "src.backend.is_float_dtype",
        "src.backend.is_int_dtype",
        "src.backend.standardize_dtype",
        "src.backend.standardize_shape",
        "src.backend.epsilon",
        "src.backend.floatx",
        "src.backend.image_data_format",
        "src.backend.set_epsilon",
        "src.backend.set_floatx",
        "src.backend.set_image_data_format",
        "src.backend.standardize_data_format",
        "src.backend.BackendVariable",
        "src.backend.distribution_lib",
        "src.backend.backend_name_scope",
        "src.backend.name_scope",
        "src.backend.device",
        "src.backend.config.keras_export",
        "src.backend.config.floatx",
        "src.backend.config.set_floatx",
        "src.backend.config.epsilon",
        "src.backend.config.set_epsilon",
        "src.backend.config.image_data_format",
        "src.backend.config.set_image_data_format",
        "src.backend.config.enable_flash_attention",
        "src.backend.config.disable_flash_attention",
        "src.backend.config.is_flash_attention_enabled",
        "src.backend.config.is_nnx_enabled",
        "src.backend.config.set_nnx_enabled",
        "src.backend.config.standardize_data_format",
        "src.backend.config.keras_home",
        "src.backend.config.backend",
        "src.backend.config.set_max_epochs",
        "src.backend.config.set_max_steps_per_epoch",
        "src.backend.config.max_epochs",
        "src.backend.config.max_steps_per_epoch",
        "src.backend.config.env_val",
        "src.backend.openvino.name_scope",
        "src.backend.openvino.IS_THREAD_SAFE",
        "src.backend.openvino.SUPPORTS_RAGGED_TENSORS",
        "src.backend.openvino.SUPPORTS_SPARSE_TENSORS",
        "src.backend.openvino.Variable",
        "src.backend.openvino.cast",
        "src.backend.openvino.compute_output_spec",
        "src.backend.openvino.cond",
        "src.backend.openvino.convert_to_numpy",
        "src.backend.openvino.convert_to_tensor",
        "src.backend.openvino.device_scope",
        "src.backend.openvino.is_tensor",
        "src.backend.openvino.random_seed_dtype",
        "src.backend.openvino.shape",
        "src.backend.openvino.vectorized_map",
        "src.backend.openvino.cudnn_ok",
        "src.backend.openvino.gru",
        "src.backend.openvino.lstm",
        "src.backend.openvino.rnn.rnn",
        "src.backend.openvino.rnn.lstm",
        "src.backend.openvino.rnn.gru",
        "src.backend.openvino.rnn.unstack",
        "src.backend.openvino.rnn.numpy_scan",
        "src.backend.openvino.rnn.cudnn_ok",
        "src.backend.openvino.nn.backend",
        "src.backend.openvino.nn.OPENVINO_DTYPES",
        "src.backend.openvino.nn.OpenVINOKerasTensor",
        "src.backend.openvino.nn.get_ov_output",
        "src.backend.openvino.nn.relu",
        "src.backend.openvino.nn.relu6",
        "src.backend.openvino.nn.celu",
        "src.backend.openvino.nn.sigmoid",
        "src.backend.openvino.nn.tanh",
        "src.backend.openvino.nn.tanh_shrink",
        "src.backend.openvino.nn.hard_tanh",
        "src.backend.openvino.nn.soft_shrink",
        "src.backend.openvino.nn.hard_shrink",
        "src.backend.openvino.nn.softplus",
        "src.backend.openvino.nn.softsign",
        "src.backend.openvino.nn.silu",
        "src.backend.openvino.nn.log_sigmoid",
        "src.backend.openvino.nn.leaky_relu",
        "src.backend.openvino.nn.sparse_sigmoid",
        "src.backend.openvino.nn.hard_sigmoid",
        "src.backend.openvino.nn.hard_silu",
        "src.backend.openvino.nn.elu",
        "src.backend.openvino.nn.selu",
        "src.backend.openvino.nn.gelu",
        "src.backend.openvino.nn.softmax",
        "src.backend.openvino.nn.log_softmax",
        "src.backend.openvino.nn.squareplus",
        "src.backend.openvino.nn.sparse_plus",
        "src.backend.openvino.nn.threshold",
        "src.backend.openvino.nn.max_pool",
        "src.backend.openvino.nn.average_pool",
        "src.backend.openvino.nn.adaptive_average_pool",
        "src.backend.openvino.nn.adaptive_max_pool",
        "src.backend.openvino.nn.conv",
        "src.backend.openvino.nn.depthwise_conv",
        "src.backend.openvino.nn.separable_conv",
        "src.backend.openvino.nn.conv_transpose",
        "src.backend.openvino.nn.one_hot",
        "src.backend.openvino.nn.multi_hot",
        "src.backend.openvino.nn.categorical_crossentropy",
        "src.backend.openvino.nn.sparse_categorical_crossentropy",
        "src.backend.openvino.nn.binary_crossentropy",
        "src.backend.openvino.nn.moments",
        "src.backend.openvino.nn.batch_normalization",
        "src.backend.openvino.nn.ctc_loss",
        "src.backend.openvino.nn.ctc_decode",
        "src.backend.openvino.nn.psnr",
        "src.backend.openvino.nn.dot_product_attention",
        "src.backend.openvino.nn.unfold",
        "src.backend.openvino.core.tree",
        "src.backend.openvino.core.KerasVariable",
        "src.backend.openvino.core.dtypes",
        "src.backend.openvino.core.standardize_dtype",
        "src.backend.openvino.core.result_type",
        "src.backend.openvino.core.KerasTensor",
        "src.backend.openvino.core.StatelessScope",
        "src.backend.openvino.core.SUPPORTS_SPARSE_TENSORS",
        "src.backend.openvino.core.SUPPORTS_RAGGED_TENSORS",
        "src.backend.openvino.core.IS_THREAD_SAFE",
        "src.backend.openvino.core.OPENVINO_DTYPES",
        "src.backend.openvino.core.DTYPES_MAX",
        "src.backend.openvino.core.DTYPES_MIN",
        "src.backend.openvino.core.align_operand_types",
        "src.backend.openvino.core.get_ov_output",
        "src.backend.openvino.core.OpenVINOKerasTensor",
        "src.backend.openvino.core.ov_to_keras_type",
        "src.backend.openvino.core.device_scope",
        "src.backend.openvino.core.get_device",
        "src.backend.openvino.core.Variable",
        "src.backend.openvino.core.convert_to_tensor",
        "src.backend.openvino.core.convert_to_numpy",
        "src.backend.openvino.core.is_tensor",
        "src.backend.openvino.core.shape",
        "src.backend.openvino.core.cast",
        "src.backend.openvino.core.cond",
        "src.backend.openvino.core.vectorized_map",
        "src.backend.openvino.core.compute_output_spec",
        "src.backend.openvino.core.scan",
        "src.backend.openvino.core.scatter",
        "src.backend.openvino.core.scatter_update",
        "src.backend.openvino.core.slice",
        "src.backend.openvino.core.slice_update",
        "src.backend.openvino.core.while_loop",
        "src.backend.openvino.core.fori_loop",
        "src.backend.openvino.core.stop_gradient",
        "src.backend.openvino.core.unstack",
        "src.backend.openvino.core.random_seed_dtype",
        "src.backend.openvino.core.custom_gradient",
        "src.backend.openvino.core.remat",
        "src.backend.openvino.numpy.config",
        "src.backend.openvino.numpy.dtypes",
        "src.backend.openvino.numpy.standardize_dtype",
        "src.backend.openvino.numpy.DTYPES_MAX",
        "src.backend.openvino.numpy.DTYPES_MIN",
        "src.backend.openvino.numpy.OPENVINO_DTYPES",
        "src.backend.openvino.numpy.OpenVINOKerasTensor",
        "src.backend.openvino.numpy.convert_to_tensor",
        "src.backend.openvino.numpy.get_ov_output",
        "src.backend.openvino.numpy.ov_to_keras_type",
        "src.backend.openvino.numpy.add",
        "src.backend.openvino.numpy.einsum",
        "src.backend.openvino.numpy.subtract",
        "src.backend.openvino.numpy.matmul",
        "src.backend.openvino.numpy.multiply",
        "src.backend.openvino.numpy.mean",
        "src.backend.openvino.numpy.max",
        "src.backend.openvino.numpy.ones",
        "src.backend.openvino.numpy.zeros",
        "src.backend.openvino.numpy.absolute",
        "src.backend.openvino.numpy.abs",
        "src.backend.openvino.numpy.all",
        "src.backend.openvino.numpy.angle",
        "src.backend.openvino.numpy.any",
        "src.backend.openvino.numpy.amax",
        "src.backend.openvino.numpy.amin",
        "src.backend.openvino.numpy.append",
        "src.backend.openvino.numpy.arange",
        "src.backend.openvino.numpy.arccos",
        "src.backend.openvino.numpy.arccosh",
        "src.backend.openvino.numpy.arcsin",
        "src.backend.openvino.numpy.arcsinh",
        "src.backend.openvino.numpy.arctan",
        "src.backend.openvino.numpy.arctan2",
        "src.backend.openvino.numpy.arctanh",
        "src.backend.openvino.numpy.argmax",
        "src.backend.openvino.numpy.argmin",
        "src.backend.openvino.numpy.argsort",
        "src.backend.openvino.numpy.array",
        "src.backend.openvino.numpy.view",
        "src.backend.openvino.numpy.average",
        "src.backend.openvino.numpy.bartlett",
        "src.backend.openvino.numpy.hamming",
        "src.backend.openvino.numpy.heaviside",
        "src.backend.openvino.numpy.kaiser",
        "src.backend.openvino.numpy.bincount",
        "src.backend.openvino.numpy.blackman",
        "src.backend.openvino.numpy.broadcast_to",
        "src.backend.openvino.numpy.cbrt",
        "src.backend.openvino.numpy.ceil",
        "src.backend.openvino.numpy.clip",
        "src.backend.openvino.numpy.concatenate",
        "src.backend.openvino.numpy.conjugate",
        "src.backend.openvino.numpy.conj",
        "src.backend.openvino.numpy.copy",
        "src.backend.openvino.numpy.cos",
        "src.backend.openvino.numpy.cosh",
        "src.backend.openvino.numpy.count_nonzero",
        "src.backend.openvino.numpy.cross",
        "src.backend.openvino.numpy.cumprod",
        "src.backend.openvino.numpy.cumsum",
        "src.backend.openvino.numpy.deg2rad",
        "src.backend.openvino.numpy.diag",
        "src.backend.openvino.numpy.diagonal",
        "src.backend.openvino.numpy.diff",
        "src.backend.openvino.numpy.digitize",
        "src.backend.openvino.numpy.dot",
        "src.backend.openvino.numpy.empty",
        "src.backend.openvino.numpy.empty_like",
        "src.backend.openvino.numpy.equal",
        "src.backend.openvino.numpy.exp",
        "src.backend.openvino.numpy.expand_dims",
        "src.backend.openvino.numpy.expm1",
        "src.backend.openvino.numpy.flip",
        "src.backend.openvino.numpy.floor",
        "src.backend.openvino.numpy.full",
        "src.backend.openvino.numpy.full_like",
        "src.backend.openvino.numpy.gcd",
        "src.backend.openvino.numpy.greater",
        "src.backend.openvino.numpy.greater_equal",
        "src.backend.openvino.numpy.hstack",
        "src.backend.openvino.numpy.hypot",
        "src.backend.openvino.numpy.identity",
        "src.backend.openvino.numpy.imag",
        "src.backend.openvino.numpy.isclose",
        "src.backend.openvino.numpy.isfinite",
        "src.backend.openvino.numpy.isin",
        "src.backend.openvino.numpy.isinf",
        "src.backend.openvino.numpy.isnan",
        "src.backend.openvino.numpy.isneginf",
        "src.backend.openvino.numpy.isposinf",
        "src.backend.openvino.numpy.isreal",
        "src.backend.openvino.numpy.kron",
        "src.backend.openvino.numpy.lcm",
        "src.backend.openvino.numpy.ldexp",
        "src.backend.openvino.numpy.less",
        "src.backend.openvino.numpy.less_equal",
        "src.backend.openvino.numpy.linspace",
        "src.backend.openvino.numpy.log",
        "src.backend.openvino.numpy.log10",
        "src.backend.openvino.numpy.log1p",
        "src.backend.openvino.numpy.log2",
        "src.backend.openvino.numpy.logaddexp",
        "src.backend.openvino.numpy.logaddexp2",
        "src.backend.openvino.numpy.logical_and",
        "src.backend.openvino.numpy.logical_not",
        "src.backend.openvino.numpy.logical_or",
        "src.backend.openvino.numpy.logspace",
        "src.backend.openvino.numpy.maximum",
        "src.backend.openvino.numpy.median",
        "src.backend.openvino.numpy.meshgrid",
        "src.backend.openvino.numpy.min",
        "src.backend.openvino.numpy.minimum",
        "src.backend.openvino.numpy.mod",
        "src.backend.openvino.numpy.moveaxis",
        "src.backend.openvino.numpy.nan_to_num",
        "src.backend.openvino.numpy.ndim",
        "src.backend.openvino.numpy.nonzero",
        "src.backend.openvino.numpy.not_equal",
        "src.backend.openvino.numpy.zeros_like",
        "src.backend.openvino.numpy.ones_like",
        "src.backend.openvino.numpy.outer",
        "src.backend.openvino.numpy.pad",
        "src.backend.openvino.numpy.prod",
        "src.backend.openvino.numpy.quantile",
        "src.backend.openvino.numpy.ravel",
        "src.backend.openvino.numpy.real",
        "src.backend.openvino.numpy.reciprocal",
        "src.backend.openvino.numpy.repeat",
        "src.backend.openvino.numpy.reshape",
        "src.backend.openvino.numpy.roll",
        "src.backend.openvino.numpy.sign",
        "src.backend.openvino.numpy.signbit",
        "src.backend.openvino.numpy.sin",
        "src.backend.openvino.numpy.sinh",
        "src.backend.openvino.numpy.size",
        "src.backend.openvino.numpy.sort",
        "src.backend.openvino.numpy.split",
        "src.backend.openvino.numpy.array_split",
        "src.backend.openvino.numpy.stack",
        "src.backend.openvino.numpy.std",
        "src.backend.openvino.numpy.swapaxes",
        "src.backend.openvino.numpy.take",
        "src.backend.openvino.numpy.take_along_axis",
        "src.backend.openvino.numpy.tan",
        "src.backend.openvino.numpy.tanh",
        "src.backend.openvino.numpy.tensordot",
        "src.backend.openvino.numpy.round",
        "src.backend.openvino.numpy.tile",
        "src.backend.openvino.numpy.trace",
        "src.backend.openvino.numpy.tri",
        "src.backend.openvino.numpy.tril",
        "src.backend.openvino.numpy.triu",
        "src.backend.openvino.numpy.vdot",
        "src.backend.openvino.numpy.vstack",
        "src.backend.openvino.numpy.vectorize",
        "src.backend.openvino.numpy.where",
        "src.backend.openvino.numpy.divide",
        "src.backend.openvino.numpy.divide_no_nan",
        "src.backend.openvino.numpy.true_divide",
        "src.backend.openvino.numpy.power",
        "src.backend.openvino.numpy.negative",
        "src.backend.openvino.numpy.square",
        "src.backend.openvino.numpy.sqrt",
        "src.backend.openvino.numpy.squeeze",
        "src.backend.openvino.numpy.transpose",
        "src.backend.openvino.numpy.trapezoid",
        "src.backend.openvino.numpy.vander",
        "src.backend.openvino.numpy.var",
        "src.backend.openvino.numpy.sum",
        "src.backend.openvino.numpy.eye",
        "src.backend.openvino.numpy.floor_divide",
        "src.backend.openvino.numpy.logical_xor",
        "src.backend.openvino.numpy.corrcoef",
        "src.backend.openvino.numpy.correlate",
        "src.backend.openvino.numpy.select",
        "src.backend.openvino.numpy.slogdet",
        "src.backend.openvino.numpy.argpartition",
        "src.backend.openvino.export.OpenvinoExportArchive",
        "src.backend.openvino.random.floatx",
        "src.backend.openvino.random.OPENVINO_DTYPES",
        "src.backend.openvino.random.OpenVINOKerasTensor",
        "src.backend.openvino.random.convert_to_numpy",
        "src.backend.openvino.random.get_ov_output",
        "src.backend.openvino.random.SeedGenerator",
        "src.backend.openvino.random.draw_seed",
        "src.backend.openvino.random.make_default_seed",
        "src.backend.openvino.random.normal",
        "src.backend.openvino.random.uniform",
        "src.backend.openvino.random.categorical",
        "src.backend.openvino.random.randint",
        "src.backend.openvino.random.truncated_normal",
        "src.backend.openvino.random.dropout",
        "src.backend.openvino.random.shuffle",
        "src.backend.openvino.random.gamma",
        "src.backend.openvino.random.binomial",
        "src.backend.openvino.random.beta",
        "src.backend.openvino.linalg.cholesky",
        "src.backend.openvino.linalg.cholesky_inverse",
        "src.backend.openvino.linalg.det",
        "src.backend.openvino.linalg.eig",
        "src.backend.openvino.linalg.eigh",
        "src.backend.openvino.linalg.inv",
        "src.backend.openvino.linalg.lu_factor",
        "src.backend.openvino.linalg.norm",
        "src.backend.openvino.linalg.qr",
        "src.backend.openvino.linalg.solve",
        "src.backend.openvino.linalg.solve_triangular",
        "src.backend.openvino.linalg.svd",
        "src.backend.openvino.linalg.lstsq",
        "src.backend.openvino.linalg.jvp",
        "src.backend.openvino.layer.OpenvinoLayer",
        "src.backend.openvino.math.OpenVINOKerasTensor",
        "src.backend.openvino.math.get_ov_output",
        "src.backend.openvino.math.segment_sum",
        "src.backend.openvino.math.segment_max",
        "src.backend.openvino.math.top_k",
        "src.backend.openvino.math.in_top_k",
        "src.backend.openvino.math.logsumexp",
        "src.backend.openvino.math.qr",
        "src.backend.openvino.math.extract_sequences",
        "src.backend.openvino.math.fft",
        "src.backend.openvino.math.fft2",
        "src.backend.openvino.math.rfft",
        "src.backend.openvino.math.irfft",
        "src.backend.openvino.math.stft",
        "src.backend.openvino.math.istft",
        "src.backend.openvino.math.rsqrt",
        "src.backend.openvino.math.erf",
        "src.backend.openvino.math.erfinv",
        "src.backend.openvino.math.solve",
        "src.backend.openvino.math.norm",
        "src.backend.openvino.trainer.backend",
        "src.backend.openvino.trainer.callbacks_module",
        "src.backend.openvino.trainer.tree",
        "src.backend.openvino.trainer.OPENVINO_DTYPES",
        "src.backend.openvino.trainer.OpenVINOKerasTensor",
        "src.backend.openvino.trainer.get_device",
        "src.backend.openvino.trainer.base_trainer",
        "src.backend.openvino.trainer.data_adapter_utils",
        "src.backend.openvino.trainer.EpochIterator",
        "src.backend.openvino.trainer.traceback_utils",
        "src.backend.openvino.trainer.OpenVINOTrainer",
        "src.backend.openvino.image.rgb_to_grayscale",
        "src.backend.openvino.image.resize",
        "src.backend.openvino.image.affine_transform",
        "src.backend.openvino.image.perspective_transform",
        "src.backend.openvino.image.map_coordinates",
        "src.backend.openvino.image.gaussian_blur",
        "src.backend.openvino.image.elastic_transform",
        "src.backend.openvino.image.scale_and_translate",
        "src.backend.numpy.name_scope",
        "src.backend.numpy.IS_THREAD_SAFE",
        "src.backend.numpy.SUPPORTS_RAGGED_TENSORS",
        "src.backend.numpy.SUPPORTS_SPARSE_TENSORS",
        "src.backend.numpy.Variable",
        "src.backend.numpy.cast",
        "src.backend.numpy.compute_output_spec",
        "src.backend.numpy.cond",
        "src.backend.numpy.convert_to_numpy",
        "src.backend.numpy.convert_to_tensor",
        "src.backend.numpy.device_scope",
        "src.backend.numpy.is_tensor",
        "src.backend.numpy.random_seed_dtype",
        "src.backend.numpy.shape",
        "src.backend.numpy.vectorized_map",
        "src.backend.numpy.cudnn_ok",
        "src.backend.numpy.gru",
        "src.backend.numpy.lstm",
        "src.backend.numpy.rnn.tree",
        "src.backend.numpy.rnn.rnn",
        "src.backend.numpy.rnn.lstm",
        "src.backend.numpy.rnn.gru",
        "src.backend.numpy.rnn.unstack",
        "src.backend.numpy.rnn.numpy_scan",
        "src.backend.numpy.rnn.cudnn_ok",
        "src.backend.numpy.nn.backend",
        "src.backend.numpy.nn.compute_adaptive_pooling_window_sizes",
        "src.backend.numpy.nn.compute_conv_transpose_padding_args_for_jax",
        "src.backend.numpy.nn.cast",
        "src.backend.numpy.nn.convert_to_tensor",
        "src.backend.numpy.nn.is_tensor",
        "src.backend.numpy.nn.scipy",
        "src.backend.numpy.nn.relu",
        "src.backend.numpy.nn.relu6",
        "src.backend.numpy.nn.sigmoid",
        "src.backend.numpy.nn.sparse_sigmoid",
        "src.backend.numpy.nn.tanh",
        "src.backend.numpy.nn.tanh_shrink",
        "src.backend.numpy.nn.softplus",
        "src.backend.numpy.nn.softsign",
        "src.backend.numpy.nn.soft_shrink",
        "src.backend.numpy.nn.sparse_plus",
        "src.backend.numpy.nn.silu",
        "src.backend.numpy.nn.squareplus",
        "src.backend.numpy.nn.log_sigmoid",
        "src.backend.numpy.nn.leaky_relu",
        "src.backend.numpy.nn.hard_sigmoid",
        "src.backend.numpy.nn.hard_silu",
        "src.backend.numpy.nn.elu",
        "src.backend.numpy.nn.selu",
        "src.backend.numpy.nn.gelu",
        "src.backend.numpy.nn.celu",
        "src.backend.numpy.nn.glu",
        "src.backend.numpy.nn.hard_tanh",
        "src.backend.numpy.nn.hard_shrink",
        "src.backend.numpy.nn.threshold",
        "src.backend.numpy.nn.softmax",
        "src.backend.numpy.nn.log_softmax",
        "src.backend.numpy.nn.sparsemax",
        "src.backend.numpy.nn.max_pool",
        "src.backend.numpy.nn.average_pool",
        "src.backend.numpy.nn.adaptive_average_pool",
        "src.backend.numpy.nn.adaptive_max_pool",
        "src.backend.numpy.nn.conv",
        "src.backend.numpy.nn.depthwise_conv",
        "src.backend.numpy.nn.separable_conv",
        "src.backend.numpy.nn.conv_transpose",
        "src.backend.numpy.nn.one_hot",
        "src.backend.numpy.nn.multi_hot",
        "src.backend.numpy.nn.categorical_crossentropy",
        "src.backend.numpy.nn.sparse_categorical_crossentropy",
        "src.backend.numpy.nn.binary_crossentropy",
        "src.backend.numpy.nn.moments",
        "src.backend.numpy.nn.batch_normalization",
        "src.backend.numpy.nn.ctc_loss",
        "src.backend.numpy.nn.ctc_decode",
        "src.backend.numpy.nn.psnr",
        "src.backend.numpy.nn.dot_product_attention",
        "src.backend.numpy.nn.unfold",
        "src.backend.numpy.core.tree",
        "src.backend.numpy.core.KerasVariable",
        "src.backend.numpy.core.standardize_dtype",
        "src.backend.numpy.core.slice_along_axis",
        "src.backend.numpy.core.result_type",
        "src.backend.numpy.core.KerasTensor",
        "src.backend.numpy.core.StatelessScope",
        "src.backend.numpy.core.SymbolicScope",
        "src.backend.numpy.core.SUPPORTS_SPARSE_TENSORS",
        "src.backend.numpy.core.SUPPORTS_RAGGED_TENSORS",
        "src.backend.numpy.core.IS_THREAD_SAFE",
        "src.backend.numpy.core.Variable",
        "src.backend.numpy.core.convert_to_tensor",
        "src.backend.numpy.core.convert_to_numpy",
        "src.backend.numpy.core.is_tensor",
        "src.backend.numpy.core.shape",
        "src.backend.numpy.core.cast",
        "src.backend.numpy.core.cond",
        "src.backend.numpy.core.vectorized_map",
        "src.backend.numpy.core.compute_output_spec",
        "src.backend.numpy.core.map",
        "src.backend.numpy.core.scan",
        "src.backend.numpy.core.associative_scan",
        "src.backend.numpy.core.scatter",
        "src.backend.numpy.core.scatter_update",
        "src.backend.numpy.core.slice",
        "src.backend.numpy.core.slice_update",
        "src.backend.numpy.core.switch",
        "src.backend.numpy.core.while_loop",
        "src.backend.numpy.core.fori_loop",
        "src.backend.numpy.core.stop_gradient",
        "src.backend.numpy.core.unstack",
        "src.backend.numpy.core.random_seed_dtype",
        "src.backend.numpy.core.custom_gradient",
        "src.backend.numpy.core.device_scope",
        "src.backend.numpy.core.remat",
        "src.backend.numpy.numpy.tree",
        "src.backend.numpy.numpy.config",
        "src.backend.numpy.numpy.standardize_dtype",
        "src.backend.numpy.numpy.dtypes",
        "src.backend.numpy.numpy.standardize_axis_for_numpy",
        "src.backend.numpy.numpy.convert_to_tensor",
        "src.backend.numpy.numpy.rot90",
        "src.backend.numpy.numpy.add",
        "src.backend.numpy.numpy.einsum",
        "src.backend.numpy.numpy.subtract",
        "src.backend.numpy.numpy.matmul",
        "src.backend.numpy.numpy.multiply",
        "src.backend.numpy.numpy.mean",
        "src.backend.numpy.numpy.max",
        "src.backend.numpy.numpy.ones",
        "src.backend.numpy.numpy.zeros",
        "src.backend.numpy.numpy.absolute",
        "src.backend.numpy.numpy.abs",
        "src.backend.numpy.numpy.all",
        "src.backend.numpy.numpy.angle",
        "src.backend.numpy.numpy.any",
        "src.backend.numpy.numpy.amax",
        "src.backend.numpy.numpy.amin",
        "src.backend.numpy.numpy.append",
        "src.backend.numpy.numpy.arange",
        "src.backend.numpy.numpy.arccos",
        "src.backend.numpy.numpy.arccosh",
        "src.backend.numpy.numpy.arcsin",
        "src.backend.numpy.numpy.arcsinh",
        "src.backend.numpy.numpy.arctan",
        "src.backend.numpy.numpy.arctan2",
        "src.backend.numpy.numpy.arctanh",
        "src.backend.numpy.numpy.argmax",
        "src.backend.numpy.numpy.argmin",
        "src.backend.numpy.numpy.argsort",
        "src.backend.numpy.numpy.array",
        "src.backend.numpy.numpy.view",
        "src.backend.numpy.numpy.average",
        "src.backend.numpy.numpy.bartlett",
        "src.backend.numpy.numpy.hamming",
        "src.backend.numpy.numpy.hanning",
        "src.backend.numpy.numpy.heaviside",
        "src.backend.numpy.numpy.kaiser",
        "src.backend.numpy.numpy.bincount",
        "src.backend.numpy.numpy.bitwise_and",
        "src.backend.numpy.numpy.bitwise_invert",
        "src.backend.numpy.numpy.bitwise_not",
        "src.backend.numpy.numpy.bitwise_or",
        "src.backend.numpy.numpy.bitwise_xor",
        "src.backend.numpy.numpy.bitwise_left_shift",
        "src.backend.numpy.numpy.left_shift",
        "src.backend.numpy.numpy.bitwise_right_shift",
        "src.backend.numpy.numpy.right_shift",
        "src.backend.numpy.numpy.blackman",
        "src.backend.numpy.numpy.broadcast_to",
        "src.backend.numpy.numpy.cbrt",
        "src.backend.numpy.numpy.ceil",
        "src.backend.numpy.numpy.clip",
        "src.backend.numpy.numpy.concatenate",
        "src.backend.numpy.numpy.conjugate",
        "src.backend.numpy.numpy.conj",
        "src.backend.numpy.numpy.copy",
        "src.backend.numpy.numpy.cos",
        "src.backend.numpy.numpy.cosh",
        "src.backend.numpy.numpy.count_nonzero",
        "src.backend.numpy.numpy.cross",
        "src.backend.numpy.numpy.cumprod",
        "src.backend.numpy.numpy.cumsum",
        "src.backend.numpy.numpy.deg2rad",
        "src.backend.numpy.numpy.diag",
        "src.backend.numpy.numpy.diagflat",
        "src.backend.numpy.numpy.diagonal",
        "src.backend.numpy.numpy.diff",
        "src.backend.numpy.numpy.digitize",
        "src.backend.numpy.numpy.dot",
        "src.backend.numpy.numpy.empty",
        "src.backend.numpy.numpy.empty_like",
        "src.backend.numpy.numpy.equal",
        "src.backend.numpy.numpy.exp",
        "src.backend.numpy.numpy.exp2",
        "src.backend.numpy.numpy.expand_dims",
        "src.backend.numpy.numpy.expm1",
        "src.backend.numpy.numpy.flip",
        "src.backend.numpy.numpy.floor",
        "src.backend.numpy.numpy.full",
        "src.backend.numpy.numpy.full_like",
        "src.backend.numpy.numpy.gcd",
        "src.backend.numpy.numpy.greater",
        "src.backend.numpy.numpy.greater_equal",
        "src.backend.numpy.numpy.hstack",
        "src.backend.numpy.numpy.hypot",
        "src.backend.numpy.numpy.identity",
        "src.backend.numpy.numpy.imag",
        "src.backend.numpy.numpy.isclose",
        "src.backend.numpy.numpy.isfinite",
        "src.backend.numpy.numpy.isin",
        "src.backend.numpy.numpy.isinf",
        "src.backend.numpy.numpy.isnan",
        "src.backend.numpy.numpy.isneginf",
        "src.backend.numpy.numpy.isposinf",
        "src.backend.numpy.numpy.isreal",
        "src.backend.numpy.numpy.kron",
        "src.backend.numpy.numpy.lcm",
        "src.backend.numpy.numpy.ldexp",
        "src.backend.numpy.numpy.less",
        "src.backend.numpy.numpy.less_equal",
        "src.backend.numpy.numpy.linspace",
        "src.backend.numpy.numpy.log",
        "src.backend.numpy.numpy.log10",
        "src.backend.numpy.numpy.log1p",
        "src.backend.numpy.numpy.log2",
        "src.backend.numpy.numpy.logaddexp",
        "src.backend.numpy.numpy.logaddexp2",
        "src.backend.numpy.numpy.logical_and",
        "src.backend.numpy.numpy.logical_not",
        "src.backend.numpy.numpy.logical_or",
        "src.backend.numpy.numpy.logspace",
        "src.backend.numpy.numpy.maximum",
        "src.backend.numpy.numpy.median",
        "src.backend.numpy.numpy.meshgrid",
        "src.backend.numpy.numpy.min",
        "src.backend.numpy.numpy.minimum",
        "src.backend.numpy.numpy.mod",
        "src.backend.numpy.numpy.moveaxis",
        "src.backend.numpy.numpy.nan_to_num",
        "src.backend.numpy.numpy.ndim",
        "src.backend.numpy.numpy.nonzero",
        "src.backend.numpy.numpy.not_equal",
        "src.backend.numpy.numpy.zeros_like",
        "src.backend.numpy.numpy.ones_like",
        "src.backend.numpy.numpy.outer",
        "src.backend.numpy.numpy.pad",
        "src.backend.numpy.numpy.prod",
        "src.backend.numpy.numpy.quantile",
        "src.backend.numpy.numpy.ravel",
        "src.backend.numpy.numpy.unravel_index",
        "src.backend.numpy.numpy.real",
        "src.backend.numpy.numpy.reciprocal",
        "src.backend.numpy.numpy.repeat",
        "src.backend.numpy.numpy.reshape",
        "src.backend.numpy.numpy.roll",
        "src.backend.numpy.numpy.searchsorted",
        "src.backend.numpy.numpy.sign",
        "src.backend.numpy.numpy.signbit",
        "src.backend.numpy.numpy.sin",
        "src.backend.numpy.numpy.sinh",
        "src.backend.numpy.numpy.size",
        "src.backend.numpy.numpy.sort",
        "src.backend.numpy.numpy.split",
        "src.backend.numpy.numpy.array_split",
        "src.backend.numpy.numpy.stack",
        "src.backend.numpy.numpy.std",
        "src.backend.numpy.numpy.swapaxes",
        "src.backend.numpy.numpy.take",
        "src.backend.numpy.numpy.take_along_axis",
        "src.backend.numpy.numpy.tan",
        "src.backend.numpy.numpy.tanh",
        "src.backend.numpy.numpy.tensordot",
        "src.backend.numpy.numpy.round",
        "src.backend.numpy.numpy.tile",
        "src.backend.numpy.numpy.trace",
        "src.backend.numpy.numpy.tri",
        "src.backend.numpy.numpy.tril",
        "src.backend.numpy.numpy.triu",
        "src.backend.numpy.numpy.trunc",
        "src.backend.numpy.numpy.vdot",
        "src.backend.numpy.numpy.inner",
        "src.backend.numpy.numpy.vstack",
        "src.backend.numpy.numpy.vectorize",
        "src.backend.numpy.numpy.where",
        "src.backend.numpy.numpy.divide",
        "src.backend.numpy.numpy.divide_no_nan",
        "src.backend.numpy.numpy.true_divide",
        "src.backend.numpy.numpy.power",
        "src.backend.numpy.numpy.negative",
        "src.backend.numpy.numpy.square",
        "src.backend.numpy.numpy.sqrt",
        "src.backend.numpy.numpy.squeeze",
        "src.backend.numpy.numpy.transpose",
        "src.backend.numpy.numpy.trapezoid",
        "src.backend.numpy.numpy.vander",
        "src.backend.numpy.numpy.var",
        "src.backend.numpy.numpy.sum",
        "src.backend.numpy.numpy.eye",
        "src.backend.numpy.numpy.floor_divide",
        "src.backend.numpy.numpy.logical_xor",
        "src.backend.numpy.numpy.corrcoef",
        "src.backend.numpy.numpy.correlate",
        "src.backend.numpy.numpy.select",
        "src.backend.numpy.numpy.slogdet",
        "src.backend.numpy.numpy.argpartition",
        "src.backend.numpy.numpy.histogram",
        "src.backend.numpy.export.NumpyExportArchive",
        "src.backend.numpy.random.floatx",
        "src.backend.numpy.random.softmax",
        "src.backend.numpy.random.SeedGenerator",
        "src.backend.numpy.random.draw_seed",
        "src.backend.numpy.random.make_default_seed",
        "src.backend.numpy.random.normal",
        "src.backend.numpy.random.uniform",
        "src.backend.numpy.random.categorical",
        "src.backend.numpy.random.randint",
        "src.backend.numpy.random.truncated_normal",
        "src.backend.numpy.random.dropout",
        "src.backend.numpy.random.shuffle",
        "src.backend.numpy.random.gamma",
        "src.backend.numpy.random.binomial",
        "src.backend.numpy.random.beta",
        "src.backend.numpy.linalg.standardize_dtype",
        "src.backend.numpy.linalg.dtypes",
        "src.backend.numpy.linalg.convert_to_tensor",
        "src.backend.numpy.linalg.cholesky",
        "src.backend.numpy.linalg.cholesky_inverse",
        "src.backend.numpy.linalg.det",
        "src.backend.numpy.linalg.eig",
        "src.backend.numpy.linalg.eigh",
        "src.backend.numpy.linalg.inv",
        "src.backend.numpy.linalg.lu_factor",
        "src.backend.numpy.linalg.norm",
        "src.backend.numpy.linalg.qr",
        "src.backend.numpy.linalg.solve",
        "src.backend.numpy.linalg.solve_triangular",
        "src.backend.numpy.linalg.svd",
        "src.backend.numpy.linalg.lstsq",
        "src.backend.numpy.linalg.jvp",
        "src.backend.numpy.layer.NumpyLayer",
        "src.backend.numpy.math.standardize_dtype",
        "src.backend.numpy.math.dtypes",
        "src.backend.numpy.math.jax_fft",
        "src.backend.numpy.math.jax_fft2",
        "src.backend.numpy.math.convert_to_tensor",
        "src.backend.numpy.math.scipy",
        "src.backend.numpy.math.segment_sum",
        "src.backend.numpy.math.segment_max",
        "src.backend.numpy.math.top_k",
        "src.backend.numpy.math.in_top_k",
        "src.backend.numpy.math.logsumexp",
        "src.backend.numpy.math.qr",
        "src.backend.numpy.math.extract_sequences",
        "src.backend.numpy.math.fft",
        "src.backend.numpy.math.fft2",
        "src.backend.numpy.math.ifft2",
        "src.backend.numpy.math.rfft",
        "src.backend.numpy.math.irfft",
        "src.backend.numpy.math.stft",
        "src.backend.numpy.math.istft",
        "src.backend.numpy.math.rsqrt",
        "src.backend.numpy.math.erf",
        "src.backend.numpy.math.erfinv",
        "src.backend.numpy.math.solve",
        "src.backend.numpy.math.norm",
        "src.backend.numpy.math.logdet",
        "src.backend.numpy.trainer.backend",
        "src.backend.numpy.trainer.callbacks_module",
        "src.backend.numpy.trainer.tree",
        "src.backend.numpy.trainer.standardize_dtype",
        "src.backend.numpy.trainer.KerasTensor",
        "src.backend.numpy.trainer.is_tensor",
        "src.backend.numpy.trainer.base_trainer",
        "src.backend.numpy.trainer.data_adapter_utils",
        "src.backend.numpy.trainer.EpochIterator",
        "src.backend.numpy.trainer.traceback_utils",
        "src.backend.numpy.trainer.NumpyTrainer",
        "src.backend.numpy.image.backend",
        "src.backend.numpy.image.convert_to_tensor",
        "src.backend.numpy.image.draw_seed",
        "src.backend.numpy.image.scipy",
        "src.backend.numpy.image.RESIZE_INTERPOLATIONS",
        "src.backend.numpy.image.AFFINE_TRANSFORM_INTERPOLATIONS",
        "src.backend.numpy.image.AFFINE_TRANSFORM_FILL_MODES",
        "src.backend.numpy.image.MAP_COORDINATES_FILL_MODES",
        "src.backend.numpy.image.SCALE_AND_TRANSLATE_METHODS",
        "src.backend.numpy.image.rgb_to_grayscale",
        "src.backend.numpy.image.rgb_to_hsv",
        "src.backend.numpy.image.hsv_to_rgb",
        "src.backend.numpy.image.resize",
        "src.backend.numpy.image.affine_transform",
        "src.backend.numpy.image.perspective_transform",
        "src.backend.numpy.image.compute_homography_matrix",
        "src.backend.numpy.image.map_coordinates",
        "src.backend.numpy.image.gaussian_blur",
        "src.backend.numpy.image.elastic_transform",
        "src.backend.numpy.image.scale_and_translate",
        "src.backend.common.result_type",
        "src.backend.common.AutocastScope",
        "src.backend.common.KerasVariable",
        "src.backend.common.get_autocast_scope",
        "src.backend.common.is_float_dtype",
        "src.backend.common.is_int_dtype",
        "src.backend.common.standardize_dtype",
        "src.backend.common.standardize_shape",
        "src.backend.common.random",
        "src.backend.common.name_scope.global_state",
        "src.backend.common.name_scope.name_scope",
        "src.backend.common.name_scope.current_path",
        "src.backend.common.remat.backend",
        "src.backend.common.remat.keras_export",
        "src.backend.common.remat.global_state",
        "src.backend.common.remat.RematScope",
        "src.backend.common.remat.RematMode",
        "src.backend.common.remat.get_current_remat_mode",
        "src.backend.common.remat.remat",
        "src.backend.common.variables.backend",
        "src.backend.common.variables.keras_export",
        "src.backend.common.variables.config",
        "src.backend.common.variables.dtypes",
        "src.backend.common.variables.global_state",
        "src.backend.common.variables.current_path",
        "src.backend.common.variables.get_stateless_scope",
        "src.backend.common.variables.in_stateless_scope",
        "src.backend.common.variables.tf",
        "src.backend.common.variables.auto_name",
        "src.backend.common.variables.Variable",
        "src.backend.common.variables.register_uninitialized_variable",
        "src.backend.common.variables.initialize_all_variables",
        "src.backend.common.variables.standardize_dtype",
        "src.backend.common.variables.standardize_shape",
        "src.backend.common.variables.shape_equal",
        "src.backend.common.variables.is_float_dtype",
        "src.backend.common.variables.is_int_dtype",
        "src.backend.common.variables.get_autocast_scope",
        "src.backend.common.variables.AutocastScope",
        "src.backend.common.backend_utils.compute_conv_transpose_padding_args_for_jax",
        "src.backend.common.backend_utils.compute_conv_transpose_padding_args_for_torch",
        "src.backend.common.backend_utils.compute_conv_transpose_output_shape",
        "src.backend.common.backend_utils.canonicalize_axis",
        "src.backend.common.backend_utils.standardize_axis_for_numpy",
        "src.backend.common.backend_utils.to_tuple_or_list",
        "src.backend.common.backend_utils.vectorize_impl",
        "src.backend.common.backend_utils.slice_along_axis",
        "src.backend.common.backend_utils.compute_adaptive_pooling_window_sizes",
        "src.backend.common.tensor_attributes.global_state",
        "src.backend.common.tensor_attributes.set_tensor_attr",
        "src.backend.common.tensor_attributes.get_tensor_attr",
        "src.backend.common.symbolic_scope.keras_export",
        "src.backend.common.symbolic_scope.global_state",
        "src.backend.common.symbolic_scope.SymbolicScope",
        "src.backend.common.symbolic_scope.in_symbolic_scope",
        "src.backend.common.symbolic_scope.get_symbolic_scope",
        "src.backend.common.dtypes.keras_export",
        "src.backend.common.dtypes.config",
        "src.backend.common.dtypes.standardize_dtype",
        "src.backend.common.dtypes.BOOL_TYPES",
        "src.backend.common.dtypes.INT_TYPES",
        "src.backend.common.dtypes.FLOAT_TYPES",
        "src.backend.common.dtypes.WEAK_TYPES",
        "src.backend.common.dtypes.COMPLEX_TYPES",
        "src.backend.common.dtypes.FLOAT8_TYPES",
        "src.backend.common.dtypes.ALLOWED_DTYPES",
        "src.backend.common.dtypes.PYTHON_DTYPES_MAP",
        "src.backend.common.dtypes.LATTICE_UPPER_BOUNDS",
        "src.backend.common.dtypes.BIT64_TO_BIT32_DTYPE",
        "src.backend.common.dtypes.result_type",
        "src.backend.common.masking.get_tensor_attr",
        "src.backend.common.masking.set_tensor_attr",
        "src.backend.common.masking.set_keras_mask",
        "src.backend.common.masking.get_keras_mask",
        "src.backend.common.global_state.backend",
        "src.backend.common.global_state.keras_export",
        "src.backend.common.global_state.GLOBAL_STATE_TRACKER",
        "src.backend.common.global_state.GLOBAL_SETTINGS_TRACKER",
        "src.backend.common.global_state.set_global_attribute",
        "src.backend.common.global_state.get_global_attribute",
        "src.backend.common.global_state.clear_session",
        "src.backend.common.keras_tensor.tree",
        "src.backend.common.keras_tensor.keras_export",
        "src.backend.common.keras_tensor.auto_name",
        "src.backend.common.keras_tensor.KerasTensor",
        "src.backend.common.keras_tensor.any_symbolic_tensors",
        "src.backend.common.keras_tensor.is_keras_tensor",
        "src.backend.common.stateless_scope.keras_export",
        "src.backend.common.stateless_scope.global_state",
        "src.backend.common.stateless_scope.StatelessScope",
        "src.backend.common.stateless_scope.in_stateless_scope",
        "src.backend.common.stateless_scope.get_stateless_scope",
        "src.backend.tensorflow.IS_THREAD_SAFE",
        "src.backend.tensorflow.SUPPORTS_RAGGED_TENSORS",
        "src.backend.tensorflow.SUPPORTS_SPARSE_TENSORS",
        "src.backend.tensorflow.Variable",
        "src.backend.tensorflow.cast",
        "src.backend.tensorflow.compute_output_spec",
        "src.backend.tensorflow.cond",
        "src.backend.tensorflow.convert_to_numpy",
        "src.backend.tensorflow.convert_to_tensor",
        "src.backend.tensorflow.device_scope",
        "src.backend.tensorflow.is_tensor",
        "src.backend.tensorflow.name_scope",
        "src.backend.tensorflow.random_seed_dtype",
        "src.backend.tensorflow.scatter",
        "src.backend.tensorflow.shape",
        "src.backend.tensorflow.stop_gradient",
        "src.backend.tensorflow.vectorized_map",
        "src.backend.tensorflow.cudnn_ok",
        "src.backend.tensorflow.gru",
        "src.backend.tensorflow.lstm",
        "src.backend.tensorflow.rnn.tree",
        "src.backend.tensorflow.rnn.rnn",
        "src.backend.tensorflow.rnn.gru",
        "src.backend.tensorflow.rnn.cudnn_ok",
        "src.backend.tensorflow.rnn.lstm",
        "src.backend.tensorflow.nn.backend",
        "src.backend.tensorflow.nn.compute_adaptive_pooling_window_sizes",
        "src.backend.tensorflow.nn.compute_conv_transpose_output_shape",
        "src.backend.tensorflow.nn.cast",
        "src.backend.tensorflow.nn.convert_to_tensor",
        "src.backend.tensorflow.nn.relu",
        "src.backend.tensorflow.nn.relu6",
        "src.backend.tensorflow.nn.sigmoid",
        "src.backend.tensorflow.nn.sparse_sigmoid",
        "src.backend.tensorflow.nn.tanh",
        "src.backend.tensorflow.nn.tanh_shrink",
        "src.backend.tensorflow.nn.softplus",
        "src.backend.tensorflow.nn.softsign",
        "src.backend.tensorflow.nn.soft_shrink",
        "src.backend.tensorflow.nn.sparse_plus",
        "src.backend.tensorflow.nn.silu",
        "src.backend.tensorflow.nn.squareplus",
        "src.backend.tensorflow.nn.log_sigmoid",
        "src.backend.tensorflow.nn.leaky_relu",
        "src.backend.tensorflow.nn.hard_sigmoid",
        "src.backend.tensorflow.nn.hard_silu",
        "src.backend.tensorflow.nn.elu",
        "src.backend.tensorflow.nn.selu",
        "src.backend.tensorflow.nn.gelu",
        "src.backend.tensorflow.nn.celu",
        "src.backend.tensorflow.nn.glu",
        "src.backend.tensorflow.nn.hard_tanh",
        "src.backend.tensorflow.nn.hard_shrink",
        "src.backend.tensorflow.nn.threshold",
        "src.backend.tensorflow.nn.softmax",
        "src.backend.tensorflow.nn.log_softmax",
        "src.backend.tensorflow.nn.sparsemax",
        "src.backend.tensorflow.nn.max_pool",
        "src.backend.tensorflow.nn.average_pool",
        "src.backend.tensorflow.nn.adaptive_average_pool",
        "src.backend.tensorflow.nn.adaptive_max_pool",
        "src.backend.tensorflow.nn.conv",
        "src.backend.tensorflow.nn.depthwise_conv",
        "src.backend.tensorflow.nn.separable_conv",
        "src.backend.tensorflow.nn.conv_transpose",
        "src.backend.tensorflow.nn.one_hot",
        "src.backend.tensorflow.nn.multi_hot",
        "src.backend.tensorflow.nn.categorical_crossentropy",
        "src.backend.tensorflow.nn.sparse_categorical_crossentropy",
        "src.backend.tensorflow.nn.binary_crossentropy",
        "src.backend.tensorflow.nn.moments",
        "src.backend.tensorflow.nn.batch_normalization",
        "src.backend.tensorflow.nn.ctc_loss",
        "src.backend.tensorflow.nn.ctc_decode",
        "src.backend.tensorflow.nn.psnr",
        "src.backend.tensorflow.nn.dot_product_attention",
        "src.backend.tensorflow.nn.unfold",
        "src.backend.tensorflow.core.tree",
        "src.backend.tensorflow.core.KerasVariable",
        "src.backend.tensorflow.core.global_state",
        "src.backend.tensorflow.core.is_int_dtype",
        "src.backend.tensorflow.core.standardize_dtype",
        "src.backend.tensorflow.core.slice_along_axis",
        "src.backend.tensorflow.core.KerasTensor",
        "src.backend.tensorflow.core.base_name_scope",
        "src.backend.tensorflow.core.StatelessScope",
        "src.backend.tensorflow.core.in_stateless_scope",
        "src.backend.tensorflow.core.SymbolicScope",
        "src.backend.tensorflow.core.sparse_to_dense",
        "src.backend.tensorflow.core.auto_name",
        "src.backend.tensorflow.core.SUPPORTS_SPARSE_TENSORS",
        "src.backend.tensorflow.core.SUPPORTS_RAGGED_TENSORS",
        "src.backend.tensorflow.core.IS_THREAD_SAFE",
        "src.backend.tensorflow.core.Variable",
        "src.backend.tensorflow.core.convert_to_tensor",
        "src.backend.tensorflow.core.convert_to_numpy",
        "src.backend.tensorflow.core.is_tensor",
        "src.backend.tensorflow.core.shape",
        "src.backend.tensorflow.core.cast",
        "src.backend.tensorflow.core.compute_output_spec",
        "src.backend.tensorflow.core.cond",
        "src.backend.tensorflow.core.vectorized_map",
        "src.backend.tensorflow.core.map",
        "src.backend.tensorflow.core.scan",
        "src.backend.tensorflow.core.associative_scan",
        "src.backend.tensorflow.core.scatter",
        "src.backend.tensorflow.core.scatter_update",
        "src.backend.tensorflow.core.slice",
        "src.backend.tensorflow.core.slice_update",
        "src.backend.tensorflow.core.switch",
        "src.backend.tensorflow.core.while_loop",
        "src.backend.tensorflow.core.fori_loop",
        "src.backend.tensorflow.core.stop_gradient",
        "src.backend.tensorflow.core.unstack",
        "src.backend.tensorflow.core.random_seed_dtype",
        "src.backend.tensorflow.core.custom_gradient",
        "src.backend.tensorflow.core.remat",
        "src.backend.tensorflow.core.name_scope",
        "src.backend.tensorflow.core.device_scope",
        "src.backend.tensorflow.numpy.tree",
        "src.backend.tensorflow.numpy.config",
        "src.backend.tensorflow.numpy.standardize_dtype",
        "src.backend.tensorflow.numpy.dtypes",
        "src.backend.tensorflow.numpy.canonicalize_axis",
        "src.backend.tensorflow.numpy.to_tuple_or_list",
        "src.backend.tensorflow.numpy.vectorize_impl",
        "src.backend.tensorflow.numpy.sparse",
        "src.backend.tensorflow.numpy.cast",
        "src.backend.tensorflow.numpy.convert_to_tensor",
        "src.backend.tensorflow.numpy.shape_op",
        "src.backend.tensorflow.numpy.rot90",
        "src.backend.tensorflow.numpy.add",
        "src.backend.tensorflow.numpy.bartlett",
        "src.backend.tensorflow.numpy.hamming",
        "src.backend.tensorflow.numpy.hanning",
        "src.backend.tensorflow.numpy.heaviside",
        "src.backend.tensorflow.numpy.kaiser",
        "src.backend.tensorflow.numpy.bincount",
        "src.backend.tensorflow.numpy.einsum",
        "src.backend.tensorflow.numpy.subtract",
        "src.backend.tensorflow.numpy.matmul",
        "src.backend.tensorflow.numpy.multiply",
        "src.backend.tensorflow.numpy.mean",
        "src.backend.tensorflow.numpy.max",
        "src.backend.tensorflow.numpy.ones",
        "src.backend.tensorflow.numpy.zeros",
        "src.backend.tensorflow.numpy.absolute",
        "src.backend.tensorflow.numpy.abs",
        "src.backend.tensorflow.numpy.all",
        "src.backend.tensorflow.numpy.angle",
        "src.backend.tensorflow.numpy.any",
        "src.backend.tensorflow.numpy.amax",
        "src.backend.tensorflow.numpy.amin",
        "src.backend.tensorflow.numpy.append",
        "src.backend.tensorflow.numpy.arange",
        "src.backend.tensorflow.numpy.arccos",
        "src.backend.tensorflow.numpy.arccosh",
        "src.backend.tensorflow.numpy.arcsin",
        "src.backend.tensorflow.numpy.arcsinh",
        "src.backend.tensorflow.numpy.arctan",
        "src.backend.tensorflow.numpy.arctan2",
        "src.backend.tensorflow.numpy.arctanh",
        "src.backend.tensorflow.numpy.argmax",
        "src.backend.tensorflow.numpy.argmin",
        "src.backend.tensorflow.numpy.argsort",
        "src.backend.tensorflow.numpy.array",
        "src.backend.tensorflow.numpy.view",
        "src.backend.tensorflow.numpy.average",
        "src.backend.tensorflow.numpy.bitwise_and",
        "src.backend.tensorflow.numpy.bitwise_invert",
        "src.backend.tensorflow.numpy.bitwise_not",
        "src.backend.tensorflow.numpy.bitwise_or",
        "src.backend.tensorflow.numpy.bitwise_xor",
        "src.backend.tensorflow.numpy.bitwise_left_shift",
        "src.backend.tensorflow.numpy.left_shift",
        "src.backend.tensorflow.numpy.bitwise_right_shift",
        "src.backend.tensorflow.numpy.right_shift",
        "src.backend.tensorflow.numpy.blackman",
        "src.backend.tensorflow.numpy.broadcast_to",
        "src.backend.tensorflow.numpy.cbrt",
        "src.backend.tensorflow.numpy.ceil",
        "src.backend.tensorflow.numpy.clip",
        "src.backend.tensorflow.numpy.concatenate",
        "src.backend.tensorflow.numpy.conjugate",
        "src.backend.tensorflow.numpy.conj",
        "src.backend.tensorflow.numpy.copy",
        "src.backend.tensorflow.numpy.cos",
        "src.backend.tensorflow.numpy.cosh",
        "src.backend.tensorflow.numpy.count_nonzero",
        "src.backend.tensorflow.numpy.cross",
        "src.backend.tensorflow.numpy.cumprod",
        "src.backend.tensorflow.numpy.cumsum",
        "src.backend.tensorflow.numpy.deg2rad",
        "src.backend.tensorflow.numpy.diag",
        "src.backend.tensorflow.numpy.diagflat",
        "src.backend.tensorflow.numpy.diagonal",
        "src.backend.tensorflow.numpy.diff",
        "src.backend.tensorflow.numpy.digitize",
        "src.backend.tensorflow.numpy.dot",
        "src.backend.tensorflow.numpy.empty",
        "src.backend.tensorflow.numpy.empty_like",
        "src.backend.tensorflow.numpy.equal",
        "src.backend.tensorflow.numpy.exp",
        "src.backend.tensorflow.numpy.exp2",
        "src.backend.tensorflow.numpy.expand_dims",
        "src.backend.tensorflow.numpy.expm1",
        "src.backend.tensorflow.numpy.flip",
        "src.backend.tensorflow.numpy.floor",
        "src.backend.tensorflow.numpy.full",
        "src.backend.tensorflow.numpy.full_like",
        "src.backend.tensorflow.numpy.gcd",
        "src.backend.tensorflow.numpy.greater",
        "src.backend.tensorflow.numpy.greater_equal",
        "src.backend.tensorflow.numpy.hstack",
        "src.backend.tensorflow.numpy.hypot",
        "src.backend.tensorflow.numpy.identity",
        "src.backend.tensorflow.numpy.imag",
        "src.backend.tensorflow.numpy.isclose",
        "src.backend.tensorflow.numpy.isfinite",
        "src.backend.tensorflow.numpy.isin",
        "src.backend.tensorflow.numpy.isinf",
        "src.backend.tensorflow.numpy.isnan",
        "src.backend.tensorflow.numpy.isneginf",
        "src.backend.tensorflow.numpy.isposinf",
        "src.backend.tensorflow.numpy.isreal",
        "src.backend.tensorflow.numpy.kron",
        "src.backend.tensorflow.numpy.lcm",
        "src.backend.tensorflow.numpy.ldexp",
        "src.backend.tensorflow.numpy.less",
        "src.backend.tensorflow.numpy.less_equal",
        "src.backend.tensorflow.numpy.linspace",
        "src.backend.tensorflow.numpy.log",
        "src.backend.tensorflow.numpy.log10",
        "src.backend.tensorflow.numpy.log1p",
        "src.backend.tensorflow.numpy.log2",
        "src.backend.tensorflow.numpy.logaddexp",
        "src.backend.tensorflow.numpy.logaddexp2",
        "src.backend.tensorflow.numpy.logical_and",
        "src.backend.tensorflow.numpy.logical_not",
        "src.backend.tensorflow.numpy.logical_or",
        "src.backend.tensorflow.numpy.logspace",
        "src.backend.tensorflow.numpy.maximum",
        "src.backend.tensorflow.numpy.median",
        "src.backend.tensorflow.numpy.meshgrid",
        "src.backend.tensorflow.numpy.min",
        "src.backend.tensorflow.numpy.minimum",
        "src.backend.tensorflow.numpy.mod",
        "src.backend.tensorflow.numpy.moveaxis",
        "src.backend.tensorflow.numpy.nan_to_num",
        "src.backend.tensorflow.numpy.ndim",
        "src.backend.tensorflow.numpy.nonzero",
        "src.backend.tensorflow.numpy.not_equal",
        "src.backend.tensorflow.numpy.ones_like",
        "src.backend.tensorflow.numpy.zeros_like",
        "src.backend.tensorflow.numpy.outer",
        "src.backend.tensorflow.numpy.pad",
        "src.backend.tensorflow.numpy.prod",
        "src.backend.tensorflow.numpy.quantile",
        "src.backend.tensorflow.numpy.ravel",
        "src.backend.tensorflow.numpy.unravel_index",
        "src.backend.tensorflow.numpy.real",
        "src.backend.tensorflow.numpy.reciprocal",
        "src.backend.tensorflow.numpy.repeat",
        "src.backend.tensorflow.numpy.reshape",
        "src.backend.tensorflow.numpy.roll",
        "src.backend.tensorflow.numpy.searchsorted",
        "src.backend.tensorflow.numpy.sign",
        "src.backend.tensorflow.numpy.signbit",
        "src.backend.tensorflow.numpy.sin",
        "src.backend.tensorflow.numpy.sinh",
        "src.backend.tensorflow.numpy.size",
        "src.backend.tensorflow.numpy.sort",
        "src.backend.tensorflow.numpy.split",
        "src.backend.tensorflow.numpy.array_split",
        "src.backend.tensorflow.numpy.stack",
        "src.backend.tensorflow.numpy.std",
        "src.backend.tensorflow.numpy.swapaxes",
        "src.backend.tensorflow.numpy.take",
        "src.backend.tensorflow.numpy.take_along_axis",
        "src.backend.tensorflow.numpy.tan",
        "src.backend.tensorflow.numpy.tanh",
        "src.backend.tensorflow.numpy.tensordot",
        "src.backend.tensorflow.numpy.round",
        "src.backend.tensorflow.numpy.tile",
        "src.backend.tensorflow.numpy.trace",
        "src.backend.tensorflow.numpy.tri",
        "src.backend.tensorflow.numpy.tril",
        "src.backend.tensorflow.numpy.triu",
        "src.backend.tensorflow.numpy.trunc",
        "src.backend.tensorflow.numpy.vdot",
        "src.backend.tensorflow.numpy.inner",
        "src.backend.tensorflow.numpy.vstack",
        "src.backend.tensorflow.numpy.vectorize",
        "src.backend.tensorflow.numpy.where",
        "src.backend.tensorflow.numpy.divide",
        "src.backend.tensorflow.numpy.divide_no_nan",
        "src.backend.tensorflow.numpy.true_divide",
        "src.backend.tensorflow.numpy.power",
        "src.backend.tensorflow.numpy.negative",
        "src.backend.tensorflow.numpy.square",
        "src.backend.tensorflow.numpy.sqrt",
        "src.backend.tensorflow.numpy.squeeze",
        "src.backend.tensorflow.numpy.transpose",
        "src.backend.tensorflow.numpy.trapezoid",
        "src.backend.tensorflow.numpy.vander",
        "src.backend.tensorflow.numpy.var",
        "src.backend.tensorflow.numpy.sum",
        "src.backend.tensorflow.numpy.eye",
        "src.backend.tensorflow.numpy.floor_divide",
        "src.backend.tensorflow.numpy.logical_xor",
        "src.backend.tensorflow.numpy.corrcoef",
        "src.backend.tensorflow.numpy.correlate",
        "src.backend.tensorflow.numpy.select",
        "src.backend.tensorflow.numpy.slogdet",
        "src.backend.tensorflow.numpy.argpartition",
        "src.backend.tensorflow.numpy.histogram",
        "src.backend.tensorflow.trackable.tracking",
        "src.backend.tensorflow.trackable.KerasAutoTrackable",
        "src.backend.tensorflow.trackable.sticky_attribute_assignment",
        "src.backend.tensorflow.export.TFExportArchive",
        "src.backend.tensorflow.random.standardize_dtype",
        "src.backend.tensorflow.random.floatx",
        "src.backend.tensorflow.random.SeedGenerator",
        "src.backend.tensorflow.random.draw_seed",
        "src.backend.tensorflow.random.make_default_seed",
        "src.backend.tensorflow.random.normal",
        "src.backend.tensorflow.random.uniform",
        "src.backend.tensorflow.random.categorical",
        "src.backend.tensorflow.random.randint",
        "src.backend.tensorflow.random.truncated_normal",
        "src.backend.tensorflow.random.dropout",
        "src.backend.tensorflow.random.shuffle",
        "src.backend.tensorflow.random.gamma",
        "src.backend.tensorflow.random.binomial",
        "src.backend.tensorflow.random.beta",
        "src.backend.tensorflow.linalg.config",
        "src.backend.tensorflow.linalg.standardize_dtype",
        "src.backend.tensorflow.linalg.dtypes",
        "src.backend.tensorflow.linalg.cast",
        "src.backend.tensorflow.linalg.convert_to_tensor",
        "src.backend.tensorflow.linalg.cholesky",
        "src.backend.tensorflow.linalg.cholesky_inverse",
        "src.backend.tensorflow.linalg.det",
        "src.backend.tensorflow.linalg.eig",
        "src.backend.tensorflow.linalg.eigh",
        "src.backend.tensorflow.linalg.inv",
        "src.backend.tensorflow.linalg.lu_factor",
        "src.backend.tensorflow.linalg.norm",
        "src.backend.tensorflow.linalg.qr",
        "src.backend.tensorflow.linalg.solve",
        "src.backend.tensorflow.linalg.solve_triangular",
        "src.backend.tensorflow.linalg.svd",
        "src.backend.tensorflow.linalg.lstsq",
        "src.backend.tensorflow.linalg.jvp",
        "src.backend.tensorflow.sparse.ones_bool",
        "src.backend.tensorflow.sparse.ones_int8",
        "src.backend.tensorflow.sparse.zeros_int8",
        "src.backend.tensorflow.sparse.ones_like_int8",
        "src.backend.tensorflow.sparse.zeros_like_int8",
        "src.backend.tensorflow.sparse.sparse_to_dense",
        "src.backend.tensorflow.sparse.sparse_with_values",
        "src.backend.tensorflow.sparse.broadcast_scalar_to_sparse_shape",
        "src.backend.tensorflow.sparse.sparse_subtract",
        "src.backend.tensorflow.sparse.sparse_union_indices_and_values",
        "src.backend.tensorflow.sparse.indexed_slices_union_indices_and_values",
        "src.backend.tensorflow.sparse.sparse_intersection_indices_and_values",
        "src.backend.tensorflow.sparse.indexed_slices_intersection_indices_and_values",
        "src.backend.tensorflow.sparse.densifying_unary",
        "src.backend.tensorflow.sparse.elementwise_unary",
        "src.backend.tensorflow.sparse.elementwise_binary_union",
        "src.backend.tensorflow.sparse.elementwise_binary_intersection",
        "src.backend.tensorflow.sparse.elementwise_division",
        "src.backend.tensorflow.layer.tree",
        "src.backend.tensorflow.layer.KerasAutoTrackable",
        "src.backend.tensorflow.layer.tf_utils",
        "src.backend.tensorflow.layer.tracking",
        "src.backend.tensorflow.layer.TFLayer",
        "src.backend.tensorflow.optimizer.backend",
        "src.backend.tensorflow.optimizer.KerasAutoTrackable",
        "src.backend.tensorflow.optimizer.base_optimizer",
        "src.backend.tensorflow.optimizer.TFOptimizer",
        "src.backend.tensorflow.optimizer.filter_empty_gradients",
        "src.backend.tensorflow.math.config",
        "src.backend.tensorflow.math.standardize_dtype",
        "src.backend.tensorflow.math.dtypes",
        "src.backend.tensorflow.math.cast",
        "src.backend.tensorflow.math.convert_to_tensor",
        "src.backend.tensorflow.math.segment_sum",
        "src.backend.tensorflow.math.segment_max",
        "src.backend.tensorflow.math.top_k",
        "src.backend.tensorflow.math.in_top_k",
        "src.backend.tensorflow.math.logsumexp",
        "src.backend.tensorflow.math.qr",
        "src.backend.tensorflow.math.extract_sequences",
        "src.backend.tensorflow.math.fft",
        "src.backend.tensorflow.math.fft2",
        "src.backend.tensorflow.math.ifft2",
        "src.backend.tensorflow.math.rfft",
        "src.backend.tensorflow.math.irfft",
        "src.backend.tensorflow.math.stft",
        "src.backend.tensorflow.math.istft",
        "src.backend.tensorflow.math.rsqrt",
        "src.backend.tensorflow.math.erf",
        "src.backend.tensorflow.math.erfinv",
        "src.backend.tensorflow.math.solve",
        "src.backend.tensorflow.math.norm",
        "src.backend.tensorflow.math.logdet",
        "src.backend.tensorflow.trainer.callbacks_module",
        "src.backend.tensorflow.trainer.metrics_module",
        "src.backend.tensorflow.trainer.optimizers_module",
        "src.backend.tensorflow.trainer.tree",
        "src.backend.tensorflow.trainer.config",
        "src.backend.tensorflow.trainer.loss_module",
        "src.backend.tensorflow.trainer.base_trainer",
        "src.backend.tensorflow.trainer.array_slicing",
        "src.backend.tensorflow.trainer.data_adapter_utils",
        "src.backend.tensorflow.trainer.EpochIterator",
        "src.backend.tensorflow.trainer.traceback_utils",
        "src.backend.tensorflow.trainer.TensorFlowTrainer",
        "src.backend.tensorflow.trainer.TFEpochIterator",
        "src.backend.tensorflow.trainer.reduce_per_replica",
        "src.backend.tensorflow.trainer.concat",
        "src.backend.tensorflow.trainer.convert_to_np_if_not_ragged",
        "src.backend.tensorflow.trainer.potentially_ragged_concat",
        "src.backend.tensorflow.distribution_lib.list_devices",
        "src.backend.tensorflow.distribution_lib.distribute_value",
        "src.backend.tensorflow.image.backend",
        "src.backend.tensorflow.image.convert_to_tensor",
        "src.backend.tensorflow.image.moveaxis",
        "src.backend.tensorflow.image.draw_seed",
        "src.backend.tensorflow.image.RESIZE_INTERPOLATIONS",
        "src.backend.tensorflow.image.AFFINE_TRANSFORM_INTERPOLATIONS",
        "src.backend.tensorflow.image.AFFINE_TRANSFORM_FILL_MODES",
        "src.backend.tensorflow.image.MAP_COORDINATES_FILL_MODES",
        "src.backend.tensorflow.image.SCALE_AND_TRANSLATE_METHODS",
        "src.backend.tensorflow.image.rgb_to_grayscale",
        "src.backend.tensorflow.image.rgb_to_hsv",
        "src.backend.tensorflow.image.hsv_to_rgb",
        "src.backend.tensorflow.image.resize",
        "src.backend.tensorflow.image.affine_transform",
        "src.backend.tensorflow.image.perspective_transform",
        "src.backend.tensorflow.image.compute_homography_matrix",
        "src.backend.tensorflow.image.map_coordinates",
        "src.backend.tensorflow.image.gaussian_blur",
        "src.backend.tensorflow.image.elastic_transform",
        "src.backend.tensorflow.image.scale_and_translate",
        "src.backend.tensorflow.tensorboard.tf",
        "src.backend.tensorflow.tensorboard.start_trace",
        "src.backend.tensorflow.tensorboard.stop_trace",
        "src.backend.tensorflow.tensorboard.start_batch_trace",
        "src.backend.tensorflow.tensorboard.stop_batch_trace",
        "src.backend.jax.is_nnx_enabled",
        "src.backend.jax.IS_THREAD_SAFE",
        "src.backend.jax.SUPPORTS_RAGGED_TENSORS",
        "src.backend.jax.SUPPORTS_SPARSE_TENSORS",
        "src.backend.jax.Variable",
        "src.backend.jax.cast",
        "src.backend.jax.compute_output_spec",
        "src.backend.jax.cond",
        "src.backend.jax.convert_to_numpy",
        "src.backend.jax.convert_to_tensor",
        "src.backend.jax.device_scope",
        "src.backend.jax.is_tensor",
        "src.backend.jax.name_scope",
        "src.backend.jax.random_seed_dtype",
        "src.backend.jax.scatter",
        "src.backend.jax.shape",
        "src.backend.jax.stop_gradient",
        "src.backend.jax.vectorized_map",
        "src.backend.jax.cudnn_ok",
        "src.backend.jax.gru",
        "src.backend.jax.lstm",
        "src.backend.jax.rnn.tree",
        "src.backend.jax.rnn.stateless_scope",
        "src.backend.jax.rnn.rnn",
        "src.backend.jax.rnn.cudnn_ok",
        "src.backend.jax.rnn.lstm",
        "src.backend.jax.rnn.gru",
        "src.backend.jax.rnn.unstack",
        "src.backend.jax.nn.backend",
        "src.backend.jax.nn.compute_adaptive_pooling_window_sizes",
        "src.backend.jax.nn.compute_conv_transpose_padding_args_for_jax",
        "src.backend.jax.nn.cast",
        "src.backend.jax.nn.convert_to_tensor",
        "src.backend.jax.nn.relu",
        "src.backend.jax.nn.relu6",
        "src.backend.jax.nn.sigmoid",
        "src.backend.jax.nn.sparse_sigmoid",
        "src.backend.jax.nn.tanh",
        "src.backend.jax.nn.tanh_shrink",
        "src.backend.jax.nn.softplus",
        "src.backend.jax.nn.softsign",
        "src.backend.jax.nn.soft_shrink",
        "src.backend.jax.nn.sparse_plus",
        "src.backend.jax.nn.silu",
        "src.backend.jax.nn.squareplus",
        "src.backend.jax.nn.log_sigmoid",
        "src.backend.jax.nn.leaky_relu",
        "src.backend.jax.nn.hard_sigmoid",
        "src.backend.jax.nn.hard_silu",
        "src.backend.jax.nn.elu",
        "src.backend.jax.nn.selu",
        "src.backend.jax.nn.gelu",
        "src.backend.jax.nn.celu",
        "src.backend.jax.nn.glu",
        "src.backend.jax.nn.hard_tanh",
        "src.backend.jax.nn.hard_shrink",
        "src.backend.jax.nn.threshold",
        "src.backend.jax.nn.softmax",
        "src.backend.jax.nn.log_softmax",
        "src.backend.jax.nn.sparsemax",
        "src.backend.jax.nn.max_pool",
        "src.backend.jax.nn.average_pool",
        "src.backend.jax.nn.adaptive_average_pool",
        "src.backend.jax.nn.adaptive_max_pool",
        "src.backend.jax.nn.conv",
        "src.backend.jax.nn.depthwise_conv",
        "src.backend.jax.nn.separable_conv",
        "src.backend.jax.nn.conv_transpose",
        "src.backend.jax.nn.one_hot",
        "src.backend.jax.nn.multi_hot",
        "src.backend.jax.nn.categorical_crossentropy",
        "src.backend.jax.nn.sparse_categorical_crossentropy",
        "src.backend.jax.nn.binary_crossentropy",
        "src.backend.jax.nn.moments",
        "src.backend.jax.nn.batch_normalization",
        "src.backend.jax.nn.ctc_loss",
        "src.backend.jax.nn.ctc_decode",
        "src.backend.jax.nn.psnr",
        "src.backend.jax.nn.wrap_flash_attention",
        "src.backend.jax.nn.dot_product_attention",
        "src.backend.jax.nn.unfold",
        "src.backend.jax.core.tree",
        "src.backend.jax.core.config",
        "src.backend.jax.core.KerasVariable",
        "src.backend.jax.core.global_state",
        "src.backend.jax.core.standardize_dtype",
        "src.backend.jax.core.KerasTensor",
        "src.backend.jax.core.base_name_scope",
        "src.backend.jax.core.StatelessScope",
        "src.backend.jax.core.get_stateless_scope",
        "src.backend.jax.core.in_stateless_scope",
        "src.backend.jax.core.SymbolicScope",
        "src.backend.jax.core.distribution_lib",
        "src.backend.jax.core.SUPPORTS_SPARSE_TENSORS",
        "src.backend.jax.core.SUPPORTS_RAGGED_TENSORS",
        "src.backend.jax.core.IS_THREAD_SAFE",
        "src.backend.jax.core.JaxVariable",
        "src.backend.jax.core.Variable",
        "src.backend.jax.core.NnxVariable",
        "src.backend.jax.core.should_shard_at_init",
        "src.backend.jax.core.convert_to_tensor",
        "src.backend.jax.core.convert_to_numpy",
        "src.backend.jax.core.is_tensor",
        "src.backend.jax.core.shape",
        "src.backend.jax.core.cast",
        "src.backend.jax.core.compute_output_spec",
        "src.backend.jax.core.cond",
        "src.backend.jax.core.vectorized_map",
        "src.backend.jax.core.map",
        "src.backend.jax.core.scan",
        "src.backend.jax.core.associative_scan",
        "src.backend.jax.core.scatter",
        "src.backend.jax.core.scatter_update",
        "src.backend.jax.core.slice",
        "src.backend.jax.core.slice_update",
        "src.backend.jax.core.switch",
        "src.backend.jax.core.while_loop",
        "src.backend.jax.core.fori_loop",
        "src.backend.jax.core.stop_gradient",
        "src.backend.jax.core.unstack",
        "src.backend.jax.core.random_seed_dtype",
        "src.backend.jax.core.custom_gradient",
        "src.backend.jax.core.remat",
        "src.backend.jax.core.name_scope",
        "src.backend.jax.core.device_scope",
        "src.backend.jax.numpy.config",
        "src.backend.jax.numpy.dtypes",
        "src.backend.jax.numpy.canonicalize_axis",
        "src.backend.jax.numpy.to_tuple_or_list",
        "src.backend.jax.numpy.standardize_dtype",
        "src.backend.jax.numpy.nn",
        "src.backend.jax.numpy.sparse",
        "src.backend.jax.numpy.cast",
        "src.backend.jax.numpy.convert_to_tensor",
        "src.backend.jax.numpy.rot90",
        "src.backend.jax.numpy.add",
        "src.backend.jax.numpy.bartlett",
        "src.backend.jax.numpy.hamming",
        "src.backend.jax.numpy.hanning",
        "src.backend.jax.numpy.heaviside",
        "src.backend.jax.numpy.hypot",
        "src.backend.jax.numpy.kaiser",
        "src.backend.jax.numpy.bincount",
        "src.backend.jax.numpy.einsum",
        "src.backend.jax.numpy.subtract",
        "src.backend.jax.numpy.matmul",
        "src.backend.jax.numpy.multiply",
        "src.backend.jax.numpy.mean",
        "src.backend.jax.numpy.max",
        "src.backend.jax.numpy.ones",
        "src.backend.jax.numpy.zeros",
        "src.backend.jax.numpy.absolute",
        "src.backend.jax.numpy.abs",
        "src.backend.jax.numpy.all",
        "src.backend.jax.numpy.angle",
        "src.backend.jax.numpy.any",
        "src.backend.jax.numpy.amax",
        "src.backend.jax.numpy.amin",
        "src.backend.jax.numpy.append",
        "src.backend.jax.numpy.arange",
        "src.backend.jax.numpy.arccos",
        "src.backend.jax.numpy.arccosh",
        "src.backend.jax.numpy.arcsin",
        "src.backend.jax.numpy.arcsinh",
        "src.backend.jax.numpy.arctan",
        "src.backend.jax.numpy.arctan2",
        "src.backend.jax.numpy.arctanh",
        "src.backend.jax.numpy.argmax",
        "src.backend.jax.numpy.argmin",
        "src.backend.jax.numpy.argsort",
        "src.backend.jax.numpy.array",
        "src.backend.jax.numpy.view",
        "src.backend.jax.numpy.average",
        "src.backend.jax.numpy.bitwise_and",
        "src.backend.jax.numpy.bitwise_invert",
        "src.backend.jax.numpy.bitwise_not",
        "src.backend.jax.numpy.bitwise_or",
        "src.backend.jax.numpy.bitwise_xor",
        "src.backend.jax.numpy.bitwise_left_shift",
        "src.backend.jax.numpy.left_shift",
        "src.backend.jax.numpy.bitwise_right_shift",
        "src.backend.jax.numpy.right_shift",
        "src.backend.jax.numpy.blackman",
        "src.backend.jax.numpy.broadcast_to",
        "src.backend.jax.numpy.cbrt",
        "src.backend.jax.numpy.ceil",
        "src.backend.jax.numpy.clip",
        "src.backend.jax.numpy.concatenate",
        "src.backend.jax.numpy.conjugate",
        "src.backend.jax.numpy.conj",
        "src.backend.jax.numpy.copy",
        "src.backend.jax.numpy.cos",
        "src.backend.jax.numpy.cosh",
        "src.backend.jax.numpy.count_nonzero",
        "src.backend.jax.numpy.cross",
        "src.backend.jax.numpy.cumprod",
        "src.backend.jax.numpy.cumsum",
        "src.backend.jax.numpy.deg2rad",
        "src.backend.jax.numpy.diag",
        "src.backend.jax.numpy.diagflat",
        "src.backend.jax.numpy.diagonal",
        "src.backend.jax.numpy.diff",
        "src.backend.jax.numpy.digitize",
        "src.backend.jax.numpy.dot",
        "src.backend.jax.numpy.empty",
        "src.backend.jax.numpy.empty_like",
        "src.backend.jax.numpy.equal",
        "src.backend.jax.numpy.exp",
        "src.backend.jax.numpy.exp2",
        "src.backend.jax.numpy.expand_dims",
        "src.backend.jax.numpy.expm1",
        "src.backend.jax.numpy.flip",
        "src.backend.jax.numpy.floor",
        "src.backend.jax.numpy.full",
        "src.backend.jax.numpy.full_like",
        "src.backend.jax.numpy.gcd",
        "src.backend.jax.numpy.greater",
        "src.backend.jax.numpy.greater_equal",
        "src.backend.jax.numpy.hstack",
        "src.backend.jax.numpy.identity",
        "src.backend.jax.numpy.imag",
        "src.backend.jax.numpy.isclose",
        "src.backend.jax.numpy.isfinite",
        "src.backend.jax.numpy.isin",
        "src.backend.jax.numpy.isinf",
        "src.backend.jax.numpy.isnan",
        "src.backend.jax.numpy.isneginf",
        "src.backend.jax.numpy.isposinf",
        "src.backend.jax.numpy.isreal",
        "src.backend.jax.numpy.kron",
        "src.backend.jax.numpy.lcm",
        "src.backend.jax.numpy.ldexp",
        "src.backend.jax.numpy.less",
        "src.backend.jax.numpy.less_equal",
        "src.backend.jax.numpy.linspace",
        "src.backend.jax.numpy.log",
        "src.backend.jax.numpy.log10",
        "src.backend.jax.numpy.log1p",
        "src.backend.jax.numpy.log2",
        "src.backend.jax.numpy.logaddexp",
        "src.backend.jax.numpy.logaddexp2",
        "src.backend.jax.numpy.logical_and",
        "src.backend.jax.numpy.logical_not",
        "src.backend.jax.numpy.logical_or",
        "src.backend.jax.numpy.logspace",
        "src.backend.jax.numpy.maximum",
        "src.backend.jax.numpy.median",
        "src.backend.jax.numpy.meshgrid",
        "src.backend.jax.numpy.min",
        "src.backend.jax.numpy.minimum",
        "src.backend.jax.numpy.mod",
        "src.backend.jax.numpy.moveaxis",
        "src.backend.jax.numpy.nan_to_num",
        "src.backend.jax.numpy.ndim",
        "src.backend.jax.numpy.nonzero",
        "src.backend.jax.numpy.not_equal",
        "src.backend.jax.numpy.ones_like",
        "src.backend.jax.numpy.zeros_like",
        "src.backend.jax.numpy.outer",
        "src.backend.jax.numpy.pad",
        "src.backend.jax.numpy.prod",
        "src.backend.jax.numpy.quantile",
        "src.backend.jax.numpy.ravel",
        "src.backend.jax.numpy.unravel_index",
        "src.backend.jax.numpy.real",
        "src.backend.jax.numpy.reciprocal",
        "src.backend.jax.numpy.repeat",
        "src.backend.jax.numpy.reshape",
        "src.backend.jax.numpy.roll",
        "src.backend.jax.numpy.searchsorted",
        "src.backend.jax.numpy.sign",
        "src.backend.jax.numpy.signbit",
        "src.backend.jax.numpy.sin",
        "src.backend.jax.numpy.sinh",
        "src.backend.jax.numpy.size",
        "src.backend.jax.numpy.sort",
        "src.backend.jax.numpy.split",
        "src.backend.jax.numpy.array_split",
        "src.backend.jax.numpy.stack",
        "src.backend.jax.numpy.std",
        "src.backend.jax.numpy.swapaxes",
        "src.backend.jax.numpy.take",
        "src.backend.jax.numpy.take_along_axis",
        "src.backend.jax.numpy.tan",
        "src.backend.jax.numpy.tanh",
        "src.backend.jax.numpy.tensordot",
        "src.backend.jax.numpy.round",
        "src.backend.jax.numpy.tile",
        "src.backend.jax.numpy.trace",
        "src.backend.jax.numpy.tri",
        "src.backend.jax.numpy.tril",
        "src.backend.jax.numpy.triu",
        "src.backend.jax.numpy.trunc",
        "src.backend.jax.numpy.vdot",
        "src.backend.jax.numpy.inner",
        "src.backend.jax.numpy.vstack",
        "src.backend.jax.numpy.vectorize",
        "src.backend.jax.numpy.where",
        "src.backend.jax.numpy.divide",
        "src.backend.jax.numpy.divide_no_nan",
        "src.backend.jax.numpy.true_divide",
        "src.backend.jax.numpy.power",
        "src.backend.jax.numpy.negative",
        "src.backend.jax.numpy.square",
        "src.backend.jax.numpy.sqrt",
        "src.backend.jax.numpy.squeeze",
        "src.backend.jax.numpy.transpose",
        "src.backend.jax.numpy.trapezoid",
        "src.backend.jax.numpy.vander",
        "src.backend.jax.numpy.var",
        "src.backend.jax.numpy.sum",
        "src.backend.jax.numpy.eye",
        "src.backend.jax.numpy.floor_divide",
        "src.backend.jax.numpy.logical_xor",
        "src.backend.jax.numpy.corrcoef",
        "src.backend.jax.numpy.correlate",
        "src.backend.jax.numpy.select",
        "src.backend.jax.numpy.slogdet",
        "src.backend.jax.numpy.argpartition",
        "src.backend.jax.numpy.histogram",
        "src.backend.jax.export.tree",
        "src.backend.jax.export.StatelessScope",
        "src.backend.jax.export.tf",
        "src.backend.jax.export.JaxExportArchive",
        "src.backend.jax.random.floatx",
        "src.backend.jax.random.SeedGenerator",
        "src.backend.jax.random.draw_seed",
        "src.backend.jax.random.make_default_seed",
        "src.backend.jax.random.jax_draw_seed",
        "src.backend.jax.random.normal",
        "src.backend.jax.random.uniform",
        "src.backend.jax.random.categorical",
        "src.backend.jax.random.randint",
        "src.backend.jax.random.truncated_normal",
        "src.backend.jax.random.dropout",
        "src.backend.jax.random.shuffle",
        "src.backend.jax.random.gamma",
        "src.backend.jax.random.binomial",
        "src.backend.jax.random.beta",
        "src.backend.jax.linalg.config",
        "src.backend.jax.linalg.standardize_dtype",
        "src.backend.jax.linalg.dtypes",
        "src.backend.jax.linalg.cast",
        "src.backend.jax.linalg.convert_to_tensor",
        "src.backend.jax.linalg.cholesky",
        "src.backend.jax.linalg.cholesky_inverse",
        "src.backend.jax.linalg.det",
        "src.backend.jax.linalg.eig",
        "src.backend.jax.linalg.eigh",
        "src.backend.jax.linalg.inv",
        "src.backend.jax.linalg.lu_factor",
        "src.backend.jax.linalg.norm",
        "src.backend.jax.linalg.qr",
        "src.backend.jax.linalg.solve",
        "src.backend.jax.linalg.solve_triangular",
        "src.backend.jax.linalg.svd",
        "src.backend.jax.linalg.lstsq",
        "src.backend.jax.linalg.jvp",
        "src.backend.jax.sparse.jax_utils",
        "src.backend.jax.sparse.axis_shape_dims_for_broadcast_in_dim",
        "src.backend.jax.sparse.bcoo_add_indices",
        "src.backend.jax.sparse.densifying_unary",
        "src.backend.jax.sparse.elementwise_unary",
        "src.backend.jax.sparse.elementwise_binary_union",
        "src.backend.jax.sparse.elementwise_division",
        "src.backend.jax.layer.is_nnx_enabled",
        "src.backend.jax.layer.BaseLayer",
        "src.backend.jax.layer.JaxLayer",
        "src.backend.jax.optimizer.base_optimizer",
        "src.backend.jax.optimizer.JaxOptimizer",
        "src.backend.jax.math.config",
        "src.backend.jax.math.standardize_dtype",
        "src.backend.jax.math.dtypes",
        "src.backend.jax.math.cast",
        "src.backend.jax.math.convert_to_tensor",
        "src.backend.jax.math.scipy",
        "src.backend.jax.math.segment_sum",
        "src.backend.jax.math.segment_max",
        "src.backend.jax.math.top_k",
        "src.backend.jax.math.in_top_k",
        "src.backend.jax.math.logsumexp",
        "src.backend.jax.math.qr",
        "src.backend.jax.math.extract_sequences",
        "src.backend.jax.math.fft",
        "src.backend.jax.math.fft2",
        "src.backend.jax.math.ifft2",
        "src.backend.jax.math.rfft",
        "src.backend.jax.math.irfft",
        "src.backend.jax.math.stft",
        "src.backend.jax.math.istft",
        "src.backend.jax.math.rsqrt",
        "src.backend.jax.math.erf",
        "src.backend.jax.math.erfinv",
        "src.backend.jax.math.solve",
        "src.backend.jax.math.norm",
        "src.backend.jax.math.logdet",
        "src.backend.jax.trainer.backend",
        "src.backend.jax.trainer.callbacks_module",
        "src.backend.jax.trainer.optimizers_module",
        "src.backend.jax.trainer.tree",
        "src.backend.jax.trainer.config",
        "src.backend.jax.trainer.jax_distribution_lib",
        "src.backend.jax.trainer.is_nnx_enabled",
        "src.backend.jax.trainer.distribution_lib",
        "src.backend.jax.trainer.base_trainer",
        "src.backend.jax.trainer.array_slicing",
        "src.backend.jax.trainer.data_adapter_utils",
        "src.backend.jax.trainer.EpochIterator",
        "src.backend.jax.trainer.traceback_utils",
        "src.backend.jax.trainer.jit",
        "src.backend.jax.trainer.JAXTrainer",
        "src.backend.jax.trainer.JAXEpochIterator",
        "src.backend.jax.distribution_lib.global_state",
        "src.backend.jax.distribution_lib.seed_generator",
        "src.backend.jax.distribution_lib.jax_utils",
        "src.backend.jax.distribution_lib.rng_utils",
        "src.backend.jax.distribution_lib.list_devices",
        "src.backend.jax.distribution_lib.get_device_count",
        "src.backend.jax.distribution_lib.distribute_variable",
        "src.backend.jax.distribution_lib.distribute_tensor",
        "src.backend.jax.distribution_lib.distribute_data_input",
        "src.backend.jax.distribution_lib.initialize_rng",
        "src.backend.jax.distribution_lib.initialize",
        "src.backend.jax.distribution_lib.num_processes",
        "src.backend.jax.distribution_lib.process_id",
        "src.backend.jax.image.backend",
        "src.backend.jax.image.convert_to_tensor",
        "src.backend.jax.image.draw_seed",
        "src.backend.jax.image.RESIZE_INTERPOLATIONS",
        "src.backend.jax.image.AFFINE_TRANSFORM_INTERPOLATIONS",
        "src.backend.jax.image.AFFINE_TRANSFORM_FILL_MODES",
        "src.backend.jax.image.MAP_COORDINATES_FILL_MODES",
        "src.backend.jax.image.SCALE_AND_TRANSLATE_METHODS",
        "src.backend.jax.image.rgb_to_grayscale",
        "src.backend.jax.image.rgb_to_hsv",
        "src.backend.jax.image.hsv_to_rgb",
        "src.backend.jax.image.resize",
        "src.backend.jax.image.affine_transform",
        "src.backend.jax.image.perspective_transform",
        "src.backend.jax.image.compute_homography_matrix",
        "src.backend.jax.image.map_coordinates",
        "src.backend.jax.image.gaussian_blur",
        "src.backend.jax.image.elastic_transform",
        "src.backend.jax.image.scale_and_translate",
        "src.backend.jax.tensorboard.jax",
        "src.backend.jax.tensorboard.start_trace",
        "src.backend.jax.tensorboard.stop_trace",
        "src.backend.jax.tensorboard.start_batch_trace",
        "src.backend.jax.tensorboard.stop_batch_trace",
        "src.backend.IS_THREAD_SAFE",
        "src.backend.SUPPORTS_RAGGED_TENSORS",
        "src.backend.SUPPORTS_SPARSE_TENSORS",
        "src.backend.cast",
        "src.backend.compute_output_spec",
        "src.backend.cond",
        "src.backend.convert_to_numpy",
        "src.backend.convert_to_tensor",
        "src.backend.device_scope",
        "src.backend.is_tensor",
        "src.backend.random_seed_dtype",
        "src.backend.scatter",
        "src.backend.shape",
        "src.backend.stop_gradient",
        "src.backend.vectorized_map",
        "src.backend.cudnn_ok",
        "src.backend.gru",
        "src.backend.lstm",
        "src.backend.rnn",
        "src.backend.nn",
        "src.backend.core",
        "src.backend.random",
        "src.backend.linalg",
        "src.backend.math",
        "src.backend.image",
        "src.backend.tensorboard",
        "src.backend.is_nnx_enabled",
        "src.backend.to_torch_dtype",
        "src.models.Functional",
        "src.models.Model",
        "src.models.Sequential",
        "src.models.variable_mapping.Layer",
        "src.models.variable_mapping.Metric",
        "src.models.variable_mapping.Optimizer",
        "src.models.variable_mapping.saving_lib",
        "src.models.variable_mapping.KerasSaveable",
        "src.models.variable_mapping.map_saveable_variables",
        "src.models.variable_mapping.map_container_variables",
        "src.models.cloning.backend",
        "src.models.cloning.tree",
        "src.models.cloning.keras_export",
        "src.models.cloning.Input",
        "src.models.cloning.InputLayer",
        "src.models.cloning.Functional",
        "src.models.cloning.functional_like_constructor",
        "src.models.cloning.Sequential",
        "src.models.cloning.serialization_lib",
        "src.models.cloning.clone_model",
        "src.models.model.backend",
        "src.models.model.keras_export",
        "src.models.model.Layer",
        "src.models.model.map_saveable_variables",
        "src.models.model.gptq_quantize",
        "src.models.model.should_quantize_layer",
        "src.models.model.saving_api",
        "src.models.model.base_trainer",
        "src.models.model.summary_utils",
        "src.models.model.traceback_utils",
        "src.models.model.Trainer",
        "src.models.model.Model",
        "src.models.model.model_from_json",
        "src.models.model.functional_init_arguments",
        "src.models.model.inject_functional_model_class",
        "src.models.sequential.backend",
        "src.models.sequential.tree",
        "src.models.sequential.keras_export",
        "src.models.sequential.global_state",
        "src.models.sequential.standardize_shape",
        "src.models.sequential.InputLayer",
        "src.models.sequential.Layer",
        "src.models.sequential.saving_utils",
        "src.models.sequential.legacy_serialization",
        "src.models.sequential.Functional",
        "src.models.sequential.Model",
        "src.models.sequential.serialization_lib",
        "src.models.sequential.Sequential",
        "src.models.functional.backend",
        "src.models.functional.ops",
        "src.models.functional.tree",
        "src.models.functional.global_state",
        "src.models.functional.Input",
        "src.models.functional.InputLayer",
        "src.models.functional.InputSpec",
        "src.models.functional.Layer",
        "src.models.functional.saving_utils",
        "src.models.functional.legacy_serialization",
        "src.models.functional.Model",
        "src.models.functional.Function",
        "src.models.functional.make_node_key",
        "src.models.functional.KerasHistory",
        "src.models.functional.Node",
        "src.models.functional.Operation",
        "src.models.functional.serialization_lib",
        "src.models.functional.tracking",
        "src.models.functional.Functional",
        "src.models.functional.functional_from_config",
        "src.models.functional.operation_fn",
        "src.models.functional.functional_like_constructor",
        "src.models.functional.unpack_singleton",
        "src.models.functional.serialize_node",
        "src.models.functional.deserialize_node",
        "src.models.functional.is_input_keras_tensor",
        "src.models.functional.clone_single_keras_tensor",
        "src.models.functional.clone_keras_tensors",
        "src.models.functional.find_nodes_by_inputs_and_outputs",
        "src.models.functional.clone_graph_nodes",
        "src.optimizers.keras_export",
        "src.optimizers.Adadelta",
        "src.optimizers.Adafactor",
        "src.optimizers.Adagrad",
        "src.optimizers.Adam",
        "src.optimizers.Adamax",
        "src.optimizers.AdamW",
        "src.optimizers.Ftrl",
        "src.optimizers.Lion",
        "src.optimizers.LossScaleOptimizer",
        "src.optimizers.Muon",
        "src.optimizers.Nadam",
        "src.optimizers.Optimizer",
        "src.optimizers.RMSprop",
        "src.optimizers.SGD",
        "src.optimizers.serialization_lib",
        "src.optimizers.ALL_OBJECTS",
        "src.optimizers.ALL_OBJECTS_DICT",
        "src.optimizers.serialize",
        "src.optimizers.deserialize",
        "src.optimizers.get",
        "src.optimizers.LegacyOptimizerWarning",
        "src.optimizers.rmsprop.ops",
        "src.optimizers.rmsprop.keras_export",
        "src.optimizers.rmsprop.optimizer",
        "src.optimizers.rmsprop.RMSprop",
        "src.optimizers.lion.ops",
        "src.optimizers.lion.keras_export",
        "src.optimizers.lion.optimizer",
        "src.optimizers.lion.Lion",
        "src.optimizers.lamb.ops",
        "src.optimizers.lamb.keras_export",
        "src.optimizers.lamb.optimizer",
        "src.optimizers.lamb.Lamb",
        "src.optimizers.adafactor.backend",
        "src.optimizers.adafactor.ops",
        "src.optimizers.adafactor.keras_export",
        "src.optimizers.adafactor.optimizer",
        "src.optimizers.adafactor.Adafactor",
        "src.optimizers.base_optimizer.backend",
        "src.optimizers.base_optimizer.initializers",
        "src.optimizers.base_optimizer.ops",
        "src.optimizers.base_optimizer.learning_rate_schedule",
        "src.optimizers.base_optimizer.serialization_lib",
        "src.optimizers.base_optimizer.KerasSaveable",
        "src.optimizers.base_optimizer.tracking",
        "src.optimizers.base_optimizer.auto_name",
        "src.optimizers.base_optimizer.BaseOptimizer",
        "src.optimizers.base_optimizer.base_optimizer_keyword_args",
        "src.optimizers.base_optimizer.global_norm",
        "src.optimizers.base_optimizer.clip_by_global_norm",
        "src.optimizers.sgd.ops",
        "src.optimizers.sgd.keras_export",
        "src.optimizers.sgd.optimizer",
        "src.optimizers.sgd.SGD",
        "src.optimizers.adamax.ops",
        "src.optimizers.adamax.keras_export",
        "src.optimizers.adamax.optimizer",
        "src.optimizers.adamax.Adamax",
        "src.optimizers.ftrl.initializers",
        "src.optimizers.ftrl.ops",
        "src.optimizers.ftrl.keras_export",
        "src.optimizers.ftrl.optimizer",
        "src.optimizers.ftrl.Ftrl",
        "src.optimizers.adagrad.initializers",
        "src.optimizers.adagrad.ops",
        "src.optimizers.adagrad.keras_export",
        "src.optimizers.adagrad.optimizer",
        "src.optimizers.adagrad.Adagrad",
        "src.optimizers.adamw.keras_export",
        "src.optimizers.adamw.adam",
        "src.optimizers.adamw.optimizer",
        "src.optimizers.adamw.AdamW",
        "src.optimizers.adam.ops",
        "src.optimizers.adam.keras_export",
        "src.optimizers.adam.optimizer",
        "src.optimizers.adam.Adam",
        "src.optimizers.optimizer.backend",
        "src.optimizers.optimizer.keras_export",
        "src.optimizers.optimizer.base_optimizer",
        "src.optimizers.optimizer.BackendOptimizer",
        "src.optimizers.optimizer.Optimizer",
        "src.optimizers.optimizer.base_optimizer_keyword_args",
        "src.optimizers.nadam.backend",
        "src.optimizers.nadam.ops",
        "src.optimizers.nadam.keras_export",
        "src.optimizers.nadam.optimizer",
        "src.optimizers.nadam.Nadam",
        "src.optimizers.muon.ops",
        "src.optimizers.muon.keras_export",
        "src.optimizers.muon.optimizer",
        "src.optimizers.muon.Muon",
        "src.optimizers.loss_scale_optimizer.backend",
        "src.optimizers.loss_scale_optimizer.initializers",
        "src.optimizers.loss_scale_optimizer.ops",
        "src.optimizers.loss_scale_optimizer.keras_export",
        "src.optimizers.loss_scale_optimizer.optimizer",
        "src.optimizers.loss_scale_optimizer.serialization_lib",
        "src.optimizers.loss_scale_optimizer.tracking",
        "src.optimizers.loss_scale_optimizer.LossScaleOptimizer",
        "src.optimizers.adadelta.ops",
        "src.optimizers.adadelta.keras_export",
        "src.optimizers.adadelta.optimizer",
        "src.optimizers.adadelta.Adadelta",
        "src.optimizers.schedules.CosineDecay",
        "src.optimizers.schedules.CosineDecayRestarts",
        "src.optimizers.schedules.ExponentialDecay",
        "src.optimizers.schedules.InverseTimeDecay",
        "src.optimizers.schedules.PiecewiseConstantDecay",
        "src.optimizers.schedules.PolynomialDecay",
        "src.optimizers.schedules.learning_rate_schedule.ops",
        "src.optimizers.schedules.learning_rate_schedule.keras_export",
        "src.optimizers.schedules.learning_rate_schedule.serialization_lib",
        "src.optimizers.schedules.learning_rate_schedule.LearningRateSchedule",
        "src.optimizers.schedules.learning_rate_schedule.ExponentialDecay",
        "src.optimizers.schedules.learning_rate_schedule.PiecewiseConstantDecay",
        "src.optimizers.schedules.learning_rate_schedule.PolynomialDecay",
        "src.optimizers.schedules.learning_rate_schedule.InverseTimeDecay",
        "src.optimizers.schedules.learning_rate_schedule.CosineDecay",
        "src.optimizers.schedules.learning_rate_schedule.CosineDecayRestarts",
        "src.optimizers.schedules.learning_rate_schedule.serialize",
        "src.optimizers.schedules.learning_rate_schedule.deserialize",
        "src.trainers.compile_utils.losses_module",
        "src.trainers.compile_utils.metrics_module",
        "src.trainers.compile_utils.ops",
        "src.trainers.compile_utils.tree",
        "src.trainers.compile_utils.KerasTensor",
        "src.trainers.compile_utils.loss_module",
        "src.trainers.compile_utils.get_object_name",
        "src.trainers.compile_utils.Tracker",
        "src.trainers.compile_utils.MetricsList",
        "src.trainers.compile_utils.is_function_like",
        "src.trainers.compile_utils.is_binary_or_sparse_categorical",
        "src.trainers.compile_utils.get_metric",
        "src.trainers.compile_utils.get_loss",
        "src.trainers.compile_utils.CompileMetrics",
        "src.trainers.compile_utils.CompileLoss",
        "src.trainers.epoch_iterator.config",
        "src.trainers.epoch_iterator.data_adapters",
        "src.trainers.epoch_iterator.EpochIterator",
        "src.trainers.trainer.backend",
        "src.trainers.trainer.metrics_module",
        "src.trainers.trainer.ops",
        "src.trainers.trainer.optimizers",
        "src.trainers.trainer.tree",
        "src.trainers.trainer.LossScaleOptimizer",
        "src.trainers.trainer.serialization_lib",
        "src.trainers.trainer.CompileLoss",
        "src.trainers.trainer.CompileMetrics",
        "src.trainers.trainer.data_adapter_utils",
        "src.trainers.trainer.python_utils",
        "src.trainers.trainer.traceback_utils",
        "src.trainers.trainer.tracking",
        "src.trainers.trainer.Trainer",
        "src.trainers.trainer.model_supports_jit",
        "src.trainers.data_adapters.distribution_lib",
        "src.trainers.data_adapters.ArrayDataAdapter",
        "src.trainers.data_adapters.GeneratorDataAdapter",
        "src.trainers.data_adapters.GrainDatasetAdapter",
        "src.trainers.data_adapters.PyDatasetAdapter",
        "src.trainers.data_adapters.TFDatasetAdapter",
        "src.trainers.data_adapters.TorchDataLoaderAdapter",
        "src.trainers.data_adapters.get_data_adapter",
        "src.trainers.data_adapters.raise_unsupported_arg",
        "src.trainers.data_adapters.is_tf_dataset",
        "src.trainers.data_adapters.is_torch_dataloader",
        "src.trainers.data_adapters.is_grain_dataset",
        "src.trainers.data_adapters.torch_data_loader_adapter.tree",
        "src.trainers.data_adapters.torch_data_loader_adapter.data_adapter_utils",
        "src.trainers.data_adapters.torch_data_loader_adapter.DataAdapter",
        "src.trainers.data_adapters.torch_data_loader_adapter.TorchDataLoaderAdapter",
        "src.trainers.data_adapters.data_adapter.DataAdapter",
        "src.trainers.data_adapters.generator_data_adapter.tree",
        "src.trainers.data_adapters.generator_data_adapter.data_adapter_utils",
        "src.trainers.data_adapters.generator_data_adapter.DataAdapter",
        "src.trainers.data_adapters.generator_data_adapter.GeneratorDataAdapter",
        "src.trainers.data_adapters.generator_data_adapter.peek_and_restore",
        "src.trainers.data_adapters.tf_dataset_adapter.tree",
        "src.trainers.data_adapters.tf_dataset_adapter.data_adapter_utils",
        "src.trainers.data_adapters.tf_dataset_adapter.DataAdapter",
        "src.trainers.data_adapters.tf_dataset_adapter.TFDatasetAdapter",
        "src.trainers.data_adapters.tf_dataset_adapter.make_class_weight_map_fn",
        "src.trainers.data_adapters.array_data_adapter.tree",
        "src.trainers.data_adapters.array_data_adapter.array_slicing",
        "src.trainers.data_adapters.array_data_adapter.data_adapter_utils",
        "src.trainers.data_adapters.array_data_adapter.DataAdapter",
        "src.trainers.data_adapters.array_data_adapter.ArrayDataAdapter",
        "src.trainers.data_adapters.array_data_adapter.can_convert_arrays",
        "src.trainers.data_adapters.grain_dataset_adapter.tree",
        "src.trainers.data_adapters.grain_dataset_adapter.data_adapter_utils",
        "src.trainers.data_adapters.grain_dataset_adapter.DataAdapter",
        "src.trainers.data_adapters.grain_dataset_adapter.grain",
        "src.trainers.data_adapters.grain_dataset_adapter.tf",
        "src.trainers.data_adapters.grain_dataset_adapter.GrainDatasetAdapter",
        "src.trainers.data_adapters.array_slicing.backend",
        "src.trainers.data_adapters.array_slicing.tree",
        "src.trainers.data_adapters.array_slicing.data_adapter_utils",
        "src.trainers.data_adapters.array_slicing.tf",
        "src.trainers.data_adapters.array_slicing.ARRAY_TYPES",
        "src.trainers.data_adapters.array_slicing.Sliceable",
        "src.trainers.data_adapters.array_slicing.NumpySliceable",
        "src.trainers.data_adapters.array_slicing.TensorflowSliceable",
        "src.trainers.data_adapters.array_slicing.TensorflowRaggedSliceable",
        "src.trainers.data_adapters.array_slicing.TensorflowSparseSliceable",
        "src.trainers.data_adapters.array_slicing.JaxSparseSliceable",
        "src.trainers.data_adapters.array_slicing.TorchSliceable",
        "src.trainers.data_adapters.array_slicing.PandasSliceable",
        "src.trainers.data_adapters.array_slicing.PandasDataFrameSliceable",
        "src.trainers.data_adapters.array_slicing.PandasSeriesSliceable",
        "src.trainers.data_adapters.array_slicing.ScipySparseSliceable",
        "src.trainers.data_adapters.array_slicing.TensorflowSparseWrapper",
        "src.trainers.data_adapters.array_slicing.to_tensorflow_sparse_wrapper",
        "src.trainers.data_adapters.array_slicing.slice_tensorflow_sparse_wrapper",
        "src.trainers.data_adapters.array_slicing.can_slice_array",
        "src.trainers.data_adapters.array_slicing.convert_to_sliceable",
        "src.trainers.data_adapters.array_slicing.train_validation_split",
        "src.trainers.data_adapters.py_dataset_adapter.keras_export",
        "src.trainers.data_adapters.py_dataset_adapter.data_adapter_utils",
        "src.trainers.data_adapters.py_dataset_adapter.DataAdapter",
        "src.trainers.data_adapters.py_dataset_adapter.PyDataset",
        "src.trainers.data_adapters.py_dataset_adapter.PyDatasetAdapter",
        "src.trainers.data_adapters.py_dataset_adapter.get_pool_class",
        "src.trainers.data_adapters.py_dataset_adapter.get_worker_id_queue",
        "src.trainers.data_adapters.py_dataset_adapter.get_index",
        "src.trainers.data_adapters.py_dataset_adapter.PyDatasetEnqueuer",
        "src.trainers.data_adapters.py_dataset_adapter.OrderedEnqueuer",
        "src.trainers.data_adapters.py_dataset_adapter.init_pool_generator",
        "src.trainers.data_adapters.data_adapter_utils.backend",
        "src.trainers.data_adapters.data_adapter_utils.ops",
        "src.trainers.data_adapters.data_adapter_utils.tree",
        "src.trainers.data_adapters.data_adapter_utils.keras_export",
        "src.trainers.data_adapters.data_adapter_utils.NUM_BATCHES_FOR_TENSOR_SPEC",
        "src.trainers.data_adapters.data_adapter_utils.unpack_x_y_sample_weight",
        "src.trainers.data_adapters.data_adapter_utils.pack_x_y_sample_weight",
        "src.trainers.data_adapters.data_adapter_utils.list_to_tuple",
        "src.trainers.data_adapters.data_adapter_utils.check_data_cardinality",
        "src.trainers.data_adapters.data_adapter_utils.class_weight_to_sample_weights",
        "src.trainers.data_adapters.data_adapter_utils.get_keras_tensor_spec",
        "src.trainers.data_adapters.data_adapter_utils.convert_to_tf_tensor_spec",
        "src.trainers.data_adapters.data_adapter_utils.get_tensor_spec",
        "src.trainers.data_adapters.data_adapter_utils.get_jax_iterator",
        "src.trainers.data_adapters.data_adapter_utils.get_numpy_iterator",
        "src.trainers.data_adapters.data_adapter_utils.get_torch_dataloader",
        "src.trainers.data_adapters.data_adapter_utils.is_tensorflow_tensor",
        "src.trainers.data_adapters.data_adapter_utils.is_tensorflow_ragged",
        "src.trainers.data_adapters.data_adapter_utils.is_tensorflow_sparse",
        "src.trainers.data_adapters.data_adapter_utils.is_jax_array",
        "src.trainers.data_adapters.data_adapter_utils.is_jax_sparse",
        "src.trainers.data_adapters.data_adapter_utils.is_torch_tensor",
        "src.trainers.data_adapters.data_adapter_utils.is_scipy_sparse",
        "src.trainers.data_adapters.data_adapter_utils.scipy_sparse_to_tf_sparse",
        "src.trainers.data_adapters.data_adapter_utils.scipy_sparse_to_jax_sparse",
        "src.trainers.data_adapters.data_adapter_utils.tf_sparse_to_jax_sparse",
        "src.trainers.data_adapters.data_adapter_utils.jax_sparse_to_tf_sparse",
        "src.regularizers.keras_export",
        "src.regularizers.L1",
        "src.regularizers.L1L2",
        "src.regularizers.L2",
        "src.regularizers.OrthogonalRegularizer",
        "src.regularizers.Regularizer",
        "src.regularizers.serialization_lib",
        "src.regularizers.to_snake_case",
        "src.regularizers.ALL_OBJECTS",
        "src.regularizers.ALL_OBJECTS_DICT",
        "src.regularizers.serialize",
        "src.regularizers.deserialize",
        "src.regularizers.get",
        "src.regularizers.regularizers.ops",
        "src.regularizers.regularizers.keras_export",
        "src.regularizers.regularizers.normalize",
        "src.regularizers.regularizers.Regularizer",
        "src.regularizers.regularizers.L1L2",
        "src.regularizers.regularizers.L1",
        "src.regularizers.regularizers.L2",
        "src.regularizers.regularizers.OrthogonalRegularizer",
        "src.regularizers.regularizers.validate_float_arg",
        "src.dtype_policies.backend",
        "src.dtype_policies.keras_export",
        "src.dtype_policies.QUANTIZATION_MODES",
        "src.dtype_policies.DTypePolicy",
        "src.dtype_policies.FloatDTypePolicy",
        "src.dtype_policies.GPTQDTypePolicy",
        "src.dtype_policies.QuantizedDTypePolicy",
        "src.dtype_policies.QuantizedFloat8DTypePolicy",
        "src.dtype_policies.DTypePolicyMap",
        "src.dtype_policies.ALL_OBJECTS",
        "src.dtype_policies.ALL_OBJECTS_DICT",
        "src.dtype_policies.serialize",
        "src.dtype_policies.deserialize",
        "src.dtype_policies.get",
        "src.dtype_policies.dtype_policy.backend",
        "src.dtype_policies.dtype_policy.ops",
        "src.dtype_policies.dtype_policy.keras_export",
        "src.dtype_policies.dtype_policy.global_state",
        "src.dtype_policies.dtype_policy.QUANTIZATION_MODES",
        "src.dtype_policies.dtype_policy.DTypePolicy",
        "src.dtype_policies.dtype_policy.FloatDTypePolicy",
        "src.dtype_policies.dtype_policy.QuantizedDTypePolicy",
        "src.dtype_policies.dtype_policy.QuantizedFloat8DTypePolicy",
        "src.dtype_policies.dtype_policy.GPTQDTypePolicy",
        "src.dtype_policies.dtype_policy.set_dtype_policy",
        "src.dtype_policies.dtype_policy.dtype_policy",
        "src.dtype_policies.dtype_policy_map.dtype_policies",
        "src.dtype_policies.dtype_policy_map.keras_export",
        "src.dtype_policies.dtype_policy_map.DTypePolicy",
        "src.dtype_policies.dtype_policy_map.DTypePolicyMap",
        "src.applications.inception_resnet_v2.backend",
        "src.applications.inception_resnet_v2.layers",
        "src.applications.inception_resnet_v2.keras_export",
        "src.applications.inception_resnet_v2.imagenet_utils",
        "src.applications.inception_resnet_v2.Layer",
        "src.applications.inception_resnet_v2.Functional",
        "src.applications.inception_resnet_v2.operation_utils",
        "src.applications.inception_resnet_v2.file_utils",
        "src.applications.inception_resnet_v2.BASE_WEIGHT_URL",
        "src.applications.inception_resnet_v2.InceptionResNetV2",
        "src.applications.inception_resnet_v2.conv2d_bn",
        "src.applications.inception_resnet_v2.CustomScaleLayer",
        "src.applications.inception_resnet_v2.inception_resnet_block",
        "src.applications.inception_resnet_v2.preprocess_input",
        "src.applications.inception_resnet_v2.decode_predictions",
        "src.applications.imagenet_utils.activations",
        "src.applications.imagenet_utils.backend",
        "src.applications.imagenet_utils.ops",
        "src.applications.imagenet_utils.keras_export",
        "src.applications.imagenet_utils.file_utils",
        "src.applications.imagenet_utils.CLASS_INDEX",
        "src.applications.imagenet_utils.CLASS_INDEX_PATH",
        "src.applications.imagenet_utils.PREPROCESS_INPUT_DOC",
        "src.applications.imagenet_utils.PREPROCESS_INPUT_MODE_DOC",
        "src.applications.imagenet_utils.PREPROCESS_INPUT_DEFAULT_ERROR_DOC",
        "src.applications.imagenet_utils.PREPROCESS_INPUT_ERROR_DOC",
        "src.applications.imagenet_utils.PREPROCESS_INPUT_RET_DOC_TF",
        "src.applications.imagenet_utils.PREPROCESS_INPUT_RET_DOC_TORCH",
        "src.applications.imagenet_utils.PREPROCESS_INPUT_RET_DOC_CAFFE",
        "src.applications.imagenet_utils.preprocess_input",
        "src.applications.imagenet_utils.decode_predictions",
        "src.applications.imagenet_utils.obtain_input_shape",
        "src.applications.imagenet_utils.correct_pad",
        "src.applications.imagenet_utils.validate_activation",
        "src.applications.densenet.backend",
        "src.applications.densenet.layers",
        "src.applications.densenet.keras_export",
        "src.applications.densenet.imagenet_utils",
        "src.applications.densenet.Functional",
        "src.applications.densenet.operation_utils",
        "src.applications.densenet.file_utils",
        "src.applications.densenet.BASE_WEIGHTS_PATH",
        "src.applications.densenet.DENSENET121_WEIGHT_PATH",
        "src.applications.densenet.DENSENET121_WEIGHT_PATH_NO_TOP",
        "src.applications.densenet.DENSENET169_WEIGHT_PATH",
        "src.applications.densenet.DENSENET169_WEIGHT_PATH_NO_TOP",
        "src.applications.densenet.DENSENET201_WEIGHT_PATH",
        "src.applications.densenet.DENSENET201_WEIGHT_PATH_NO_TOP",
        "src.applications.densenet.dense_block",
        "src.applications.densenet.transition_block",
        "src.applications.densenet.conv_block",
        "src.applications.densenet.DenseNet",
        "src.applications.densenet.DenseNet121",
        "src.applications.densenet.DenseNet169",
        "src.applications.densenet.DenseNet201",
        "src.applications.densenet.preprocess_input",
        "src.applications.densenet.decode_predictions",
        "src.applications.densenet.DOC",
        "src.applications.mobilenet_v2.backend",
        "src.applications.mobilenet_v2.layers",
        "src.applications.mobilenet_v2.keras_export",
        "src.applications.mobilenet_v2.imagenet_utils",
        "src.applications.mobilenet_v2.Functional",
        "src.applications.mobilenet_v2.operation_utils",
        "src.applications.mobilenet_v2.file_utils",
        "src.applications.mobilenet_v2.BASE_WEIGHT_PATH",
        "src.applications.mobilenet_v2.MobileNetV2",
        "src.applications.mobilenet_v2.preprocess_input",
        "src.applications.mobilenet_v2.decode_predictions",
        "src.applications.mobilenet_v3.backend",
        "src.applications.mobilenet_v3.layers",
        "src.applications.mobilenet_v3.keras_export",
        "src.applications.mobilenet_v3.imagenet_utils",
        "src.applications.mobilenet_v3.Functional",
        "src.applications.mobilenet_v3.operation_utils",
        "src.applications.mobilenet_v3.file_utils",
        "src.applications.mobilenet_v3.BASE_WEIGHT_PATH",
        "src.applications.mobilenet_v3.WEIGHTS_HASHES",
        "src.applications.mobilenet_v3.BASE_DOCSTRING",
        "src.applications.mobilenet_v3.MobileNetV3",
        "src.applications.mobilenet_v3.MobileNetV3Small",
        "src.applications.mobilenet_v3.MobileNetV3Large",
        "src.applications.mobilenet_v3.relu",
        "src.applications.mobilenet_v3.hard_sigmoid",
        "src.applications.mobilenet_v3.hard_swish",
        "src.applications.mobilenet_v3.preprocess_input",
        "src.applications.mobilenet_v3.decode_predictions",
        "src.applications.efficientnet.backend",
        "src.applications.efficientnet.layers",
        "src.applications.efficientnet.keras_export",
        "src.applications.efficientnet.imagenet_utils",
        "src.applications.efficientnet.Functional",
        "src.applications.efficientnet.operation_utils",
        "src.applications.efficientnet.file_utils",
        "src.applications.efficientnet.BASE_WEIGHTS_PATH",
        "src.applications.efficientnet.WEIGHTS_HASHES",
        "src.applications.efficientnet.DEFAULT_BLOCKS_ARGS",
        "src.applications.efficientnet.CONV_KERNEL_INITIALIZER",
        "src.applications.efficientnet.DENSE_KERNEL_INITIALIZER",
        "src.applications.efficientnet.BASE_DOCSTRING",
        "src.applications.efficientnet.IMAGENET_STDDEV_RGB",
        "src.applications.efficientnet.EfficientNet",
        "src.applications.efficientnet.block",
        "src.applications.efficientnet.EfficientNetB0",
        "src.applications.efficientnet.EfficientNetB1",
        "src.applications.efficientnet.EfficientNetB2",
        "src.applications.efficientnet.EfficientNetB3",
        "src.applications.efficientnet.EfficientNetB4",
        "src.applications.efficientnet.EfficientNetB5",
        "src.applications.efficientnet.EfficientNetB6",
        "src.applications.efficientnet.EfficientNetB7",
        "src.applications.efficientnet.preprocess_input",
        "src.applications.efficientnet.decode_predictions",
        "src.applications.mobilenet.backend",
        "src.applications.mobilenet.layers",
        "src.applications.mobilenet.keras_export",
        "src.applications.mobilenet.imagenet_utils",
        "src.applications.mobilenet.Functional",
        "src.applications.mobilenet.operation_utils",
        "src.applications.mobilenet.file_utils",
        "src.applications.mobilenet.BASE_WEIGHT_PATH",
        "src.applications.mobilenet.MobileNet",
        "src.applications.mobilenet.preprocess_input",
        "src.applications.mobilenet.decode_predictions",
        "src.applications.inception_v3.backend",
        "src.applications.inception_v3.layers",
        "src.applications.inception_v3.keras_export",
        "src.applications.inception_v3.imagenet_utils",
        "src.applications.inception_v3.Functional",
        "src.applications.inception_v3.operation_utils",
        "src.applications.inception_v3.file_utils",
        "src.applications.inception_v3.WEIGHTS_PATH",
        "src.applications.inception_v3.WEIGHTS_PATH_NO_TOP",
        "src.applications.inception_v3.InceptionV3",
        "src.applications.inception_v3.conv2d_bn",
        "src.applications.inception_v3.preprocess_input",
        "src.applications.inception_v3.decode_predictions",
        "src.applications.resnet.backend",
        "src.applications.resnet.layers",
        "src.applications.resnet.keras_export",
        "src.applications.resnet.imagenet_utils",
        "src.applications.resnet.Functional",
        "src.applications.resnet.operation_utils",
        "src.applications.resnet.file_utils",
        "src.applications.resnet.BASE_WEIGHTS_PATH",
        "src.applications.resnet.WEIGHTS_HASHES",
        "src.applications.resnet.ResNet",
        "src.applications.resnet.residual_block_v1",
        "src.applications.resnet.stack_residual_blocks_v1",
        "src.applications.resnet.residual_block_v2",
        "src.applications.resnet.stack_residual_blocks_v2",
        "src.applications.resnet.ResNet50",
        "src.applications.resnet.ResNet101",
        "src.applications.resnet.ResNet152",
        "src.applications.resnet.preprocess_input",
        "src.applications.resnet.decode_predictions",
        "src.applications.resnet.DOC",
        "src.applications.vgg16.backend",
        "src.applications.vgg16.layers",
        "src.applications.vgg16.keras_export",
        "src.applications.vgg16.imagenet_utils",
        "src.applications.vgg16.Functional",
        "src.applications.vgg16.operation_utils",
        "src.applications.vgg16.file_utils",
        "src.applications.vgg16.WEIGHTS_PATH",
        "src.applications.vgg16.WEIGHTS_PATH_NO_TOP",
        "src.applications.vgg16.VGG16",
        "src.applications.vgg16.preprocess_input",
        "src.applications.vgg16.decode_predictions",
        "src.applications.efficientnet_v2.backend",
        "src.applications.efficientnet_v2.initializers",
        "src.applications.efficientnet_v2.layers",
        "src.applications.efficientnet_v2.keras_export",
        "src.applications.efficientnet_v2.imagenet_utils",
        "src.applications.efficientnet_v2.Functional",
        "src.applications.efficientnet_v2.operation_utils",
        "src.applications.efficientnet_v2.file_utils",
        "src.applications.efficientnet_v2.BASE_WEIGHTS_PATH",
        "src.applications.efficientnet_v2.WEIGHTS_HASHES",
        "src.applications.efficientnet_v2.DEFAULT_BLOCKS_ARGS",
        "src.applications.efficientnet_v2.CONV_KERNEL_INITIALIZER",
        "src.applications.efficientnet_v2.DENSE_KERNEL_INITIALIZER",
        "src.applications.efficientnet_v2.BASE_DOCSTRING",
        "src.applications.efficientnet_v2.round_filters",
        "src.applications.efficientnet_v2.round_repeats",
        "src.applications.efficientnet_v2.MBConvBlock",
        "src.applications.efficientnet_v2.FusedMBConvBlock",
        "src.applications.efficientnet_v2.EfficientNetV2",
        "src.applications.efficientnet_v2.EfficientNetV2B0",
        "src.applications.efficientnet_v2.EfficientNetV2B1",
        "src.applications.efficientnet_v2.EfficientNetV2B2",
        "src.applications.efficientnet_v2.EfficientNetV2B3",
        "src.applications.efficientnet_v2.EfficientNetV2S",
        "src.applications.efficientnet_v2.EfficientNetV2M",
        "src.applications.efficientnet_v2.EfficientNetV2L",
        "src.applications.efficientnet_v2.preprocess_input",
        "src.applications.efficientnet_v2.decode_predictions",
        "src.applications.vgg19.backend",
        "src.applications.vgg19.layers",
        "src.applications.vgg19.keras_export",
        "src.applications.vgg19.imagenet_utils",
        "src.applications.vgg19.Functional",
        "src.applications.vgg19.operation_utils",
        "src.applications.vgg19.file_utils",
        "src.applications.vgg19.WEIGHTS_PATH",
        "src.applications.vgg19.WEIGHTS_PATH_NO_TOP",
        "src.applications.vgg19.VGG19",
        "src.applications.vgg19.preprocess_input",
        "src.applications.vgg19.decode_predictions",
        "src.applications.xception.backend",
        "src.applications.xception.layers",
        "src.applications.xception.keras_export",
        "src.applications.xception.imagenet_utils",
        "src.applications.xception.Functional",
        "src.applications.xception.operation_utils",
        "src.applications.xception.file_utils",
        "src.applications.xception.WEIGHTS_PATH",
        "src.applications.xception.WEIGHTS_PATH_NO_TOP",
        "src.applications.xception.Xception",
        "src.applications.xception.preprocess_input",
        "src.applications.xception.decode_predictions",
        "src.applications.nasnet.backend",
        "src.applications.nasnet.layers",
        "src.applications.nasnet.keras_export",
        "src.applications.nasnet.imagenet_utils",
        "src.applications.nasnet.Functional",
        "src.applications.nasnet.operation_utils",
        "src.applications.nasnet.file_utils",
        "src.applications.nasnet.BASE_WEIGHTS_PATH",
        "src.applications.nasnet.NASNET_MOBILE_WEIGHT_PATH",
        "src.applications.nasnet.NASNET_MOBILE_WEIGHT_PATH_NO_TOP",
        "src.applications.nasnet.NASNET_LARGE_WEIGHT_PATH",
        "src.applications.nasnet.NASNET_LARGE_WEIGHT_PATH_NO_TOP",
        "src.applications.nasnet.NASNet",
        "src.applications.nasnet.NASNetMobile",
        "src.applications.nasnet.NASNetLarge",
        "src.applications.nasnet.preprocess_input",
        "src.applications.nasnet.decode_predictions",
        "src.applications.resnet_v2.keras_export",
        "src.applications.resnet_v2.imagenet_utils",
        "src.applications.resnet_v2.resnet",
        "src.applications.resnet_v2.ResNet50V2",
        "src.applications.resnet_v2.ResNet101V2",
        "src.applications.resnet_v2.ResNet152V2",
        "src.applications.resnet_v2.preprocess_input",
        "src.applications.resnet_v2.decode_predictions",
        "src.applications.resnet_v2.DOC",
        "src.applications.convnext.backend",
        "src.applications.convnext.initializers",
        "src.applications.convnext.layers",
        "src.applications.convnext.ops",
        "src.applications.convnext.random",
        "src.applications.convnext.keras_export",
        "src.applications.convnext.imagenet_utils",
        "src.applications.convnext.Layer",
        "src.applications.convnext.Functional",
        "src.applications.convnext.Sequential",
        "src.applications.convnext.operation_utils",
        "src.applications.convnext.file_utils",
        "src.applications.convnext.BASE_WEIGHTS_PATH",
        "src.applications.convnext.WEIGHTS_HASHES",
        "src.applications.convnext.MODEL_CONFIGS",
        "src.applications.convnext.BASE_DOCSTRING",
        "src.applications.convnext.StochasticDepth",
        "src.applications.convnext.LayerScale",
        "src.applications.convnext.ConvNeXtBlock",
        "src.applications.convnext.PreStem",
        "src.applications.convnext.Head",
        "src.applications.convnext.ConvNeXt",
        "src.applications.convnext.ConvNeXtTiny",
        "src.applications.convnext.ConvNeXtSmall",
        "src.applications.convnext.ConvNeXtBase",
        "src.applications.convnext.ConvNeXtLarge",
        "src.applications.convnext.ConvNeXtXLarge",
        "src.applications.convnext.preprocess_input",
        "src.applications.convnext.decode_predictions",
        "src.ops.cast",
        "src.ops.cond",
        "src.ops.is_tensor",
        "src.ops.name_scope",
        "src.ops.random",
        "src.ops.einops.keras_export",
        "src.ops.einops.KerasTensor",
        "src.ops.einops.any_symbolic_tensors",
        "src.ops.einops.shape",
        "src.ops.einops.prod",
        "src.ops.einops.reshape",
        "src.ops.einops.transpose",
        "src.ops.einops.Operation",
        "src.ops.einops.Rearrange",
        "src.ops.einops.rearrange",
        "src.ops.nn.backend",
        "src.ops.nn.keras_export",
        "src.ops.nn.KerasTensor",
        "src.ops.nn.any_symbolic_tensors",
        "src.ops.nn.config",
        "src.ops.nn.standardize_data_format",
        "src.ops.nn.compute_conv_transpose_output_shape",
        "src.ops.nn.operation_utils",
        "src.ops.nn.Operation",
        "src.ops.nn.reduce_shape",
        "src.ops.nn.is_continuous_axis",
        "src.ops.nn.Relu",
        "src.ops.nn.relu",
        "src.ops.nn.Relu6",
        "src.ops.nn.relu6",
        "src.ops.nn.Sigmoid",
        "src.ops.nn.sigmoid",
        "src.ops.nn.SparseSigmoid",
        "src.ops.nn.sparse_sigmoid",
        "src.ops.nn.Softplus",
        "src.ops.nn.softplus",
        "src.ops.nn.Softsign",
        "src.ops.nn.softsign",
        "src.ops.nn.SoftShrink",
        "src.ops.nn.soft_shrink",
        "src.ops.nn.SparsePlus",
        "src.ops.nn.sparse_plus",
        "src.ops.nn.Silu",
        "src.ops.nn.silu",
        "src.ops.nn.Squareplus",
        "src.ops.nn.squareplus",
        "src.ops.nn.LogSigmoid",
        "src.ops.nn.log_sigmoid",
        "src.ops.nn.LeakyRelu",
        "src.ops.nn.leaky_relu",
        "src.ops.nn.HardSigmoid",
        "src.ops.nn.hard_sigmoid",
        "src.ops.nn.HardSilu",
        "src.ops.nn.hard_silu",
        "src.ops.nn.Elu",
        "src.ops.nn.elu",
        "src.ops.nn.Selu",
        "src.ops.nn.selu",
        "src.ops.nn.Gelu",
        "src.ops.nn.gelu",
        "src.ops.nn.Celu",
        "src.ops.nn.celu",
        "src.ops.nn.Glu",
        "src.ops.nn.glu",
        "src.ops.nn.TanhShrink",
        "src.ops.nn.tanh_shrink",
        "src.ops.nn.HardTanh",
        "src.ops.nn.hard_tanh",
        "src.ops.nn.HardShrink",
        "src.ops.nn.hard_shrink",
        "src.ops.nn.Threshold",
        "src.ops.nn.threshold",
        "src.ops.nn.Softmax",
        "src.ops.nn.softmax",
        "src.ops.nn.LogSoftmax",
        "src.ops.nn.log_softmax",
        "src.ops.nn.Sparsemax",
        "src.ops.nn.sparsemax",
        "src.ops.nn.MaxPool",
        "src.ops.nn.max_pool",
        "src.ops.nn.AdaptiveMaxPool",
        "src.ops.nn.adaptive_max_pool",
        "src.ops.nn.AveragePool",
        "src.ops.nn.average_pool",
        "src.ops.nn.AdaptiveAveragePool",
        "src.ops.nn.adaptive_average_pool",
        "src.ops.nn.Conv",
        "src.ops.nn.conv",
        "src.ops.nn.DepthwiseConv",
        "src.ops.nn.depthwise_conv",
        "src.ops.nn.SeparableConv",
        "src.ops.nn.separable_conv",
        "src.ops.nn.ConvTranspose",
        "src.ops.nn.conv_transpose",
        "src.ops.nn.OneHot",
        "src.ops.nn.one_hot",
        "src.ops.nn.BinaryCrossentropy",
        "src.ops.nn.binary_crossentropy",
        "src.ops.nn.CategoricalCrossentropy",
        "src.ops.nn.categorical_crossentropy",
        "src.ops.nn.SparseCategoricalCrossentropy",
        "src.ops.nn.sparse_categorical_crossentropy",
        "src.ops.nn.MultiHot",
        "src.ops.nn.multi_hot",
        "src.ops.nn.Moments",
        "src.ops.nn.moments",
        "src.ops.nn.BatchNorm",
        "src.ops.nn.batch_normalization",
        "src.ops.nn.CTCLoss",
        "src.ops.nn.ctc_loss",
        "src.ops.nn.CTCDecode",
        "src.ops.nn.ctc_decode",
        "src.ops.nn.Normalize",
        "src.ops.nn.normalize",
        "src.ops.nn.PSNR",
        "src.ops.nn.psnr",
        "src.ops.nn.DotProductAttention",
        "src.ops.nn.dot_product_attention",
        "src.ops.nn.RMSNorm",
        "src.ops.nn.rms_normalization",
        "src.ops.nn.LayerNorm",
        "src.ops.nn.layer_normalization",
        "src.ops.nn.Polar",
        "src.ops.nn.polar",
        "src.ops.nn.Unfold",
        "src.ops.nn.unfold",
        "src.ops.core.backend",
        "src.ops.core.tree",
        "src.ops.core.keras_export",
        "src.ops.core.KerasTensor",
        "src.ops.core.any_symbolic_tensors",
        "src.ops.core.slice_along_axis",
        "src.ops.core.Operation",
        "src.ops.core.serialization_lib",
        "src.ops.core.traceback_utils",
        "src.ops.core.Map",
        "src.ops.core.map",
        "src.ops.core.Scan",
        "src.ops.core.scan",
        "src.ops.core.AssociativeScan",
        "src.ops.core.associative_scan",
        "src.ops.core.Scatter",
        "src.ops.core.scatter",
        "src.ops.core.ScatterUpdate",
        "src.ops.core.scatter_update",
        "src.ops.core.Slice",
        "src.ops.core.slice",
        "src.ops.core.SliceUpdate",
        "src.ops.core.slice_update",
        "src.ops.core.Switch",
        "src.ops.core.switch",
        "src.ops.core.WhileLoop",
        "src.ops.core.while_loop",
        "src.ops.core.StopGradient",
        "src.ops.core.stop_gradient",
        "src.ops.core.ForiLoop",
        "src.ops.core.fori_loop",
        "src.ops.core.Unstack",
        "src.ops.core.unstack",
        "src.ops.core.shape",
        "src.ops.core.dtype",
        "src.ops.core.Cast",
        "src.ops.core.cast",
        "src.ops.core.SaturateCast",
        "src.ops.core.saturate_cast",
        "src.ops.core.ConvertToTensor",
        "src.ops.core.convert_to_tensor",
        "src.ops.core.convert_to_numpy",
        "src.ops.core.Cond",
        "src.ops.core.cond",
        "src.ops.core.VectorizedMap",
        "src.ops.core.vectorized_map",
        "src.ops.core.is_tensor",
        "src.ops.core.custom_gradient",
        "src.ops.numpy.backend",
        "src.ops.numpy.keras_export",
        "src.ops.numpy.KerasTensor",
        "src.ops.numpy.any_symbolic_tensors",
        "src.ops.numpy.dtypes",
        "src.ops.numpy.canonicalize_axis",
        "src.ops.numpy.to_tuple_or_list",
        "src.ops.numpy.operation_utils",
        "src.ops.numpy.Operation",
        "src.ops.numpy.broadcast_shapes",
        "src.ops.numpy.reduce_shape",
        "src.ops.numpy.Rot90",
        "src.ops.numpy.rot90",
        "src.ops.numpy.shape_equal",
        "src.ops.numpy.Absolute",
        "src.ops.numpy.absolute",
        "src.ops.numpy.Abs",
        "src.ops.numpy.abs",
        "src.ops.numpy.Add",
        "src.ops.numpy.add",
        "src.ops.numpy.All",
        "src.ops.numpy.all",
        "src.ops.numpy.Angle",
        "src.ops.numpy.angle",
        "src.ops.numpy.Any",
        "src.ops.numpy.any",
        "src.ops.numpy.Amax",
        "src.ops.numpy.amax",
        "src.ops.numpy.Amin",
        "src.ops.numpy.amin",
        "src.ops.numpy.Append",
        "src.ops.numpy.append",
        "src.ops.numpy.Arange",
        "src.ops.numpy.arange",
        "src.ops.numpy.Arccos",
        "src.ops.numpy.arccos",
        "src.ops.numpy.Arccosh",
        "src.ops.numpy.arccosh",
        "src.ops.numpy.Arcsin",
        "src.ops.numpy.arcsin",
        "src.ops.numpy.Arcsinh",
        "src.ops.numpy.arcsinh",
        "src.ops.numpy.Arctan",
        "src.ops.numpy.arctan",
        "src.ops.numpy.Arctan2",
        "src.ops.numpy.arctan2",
        "src.ops.numpy.Arctanh",
        "src.ops.numpy.arctanh",
        "src.ops.numpy.Argmax",
        "src.ops.numpy.argmax",
        "src.ops.numpy.Argmin",
        "src.ops.numpy.argmin",
        "src.ops.numpy.Argsort",
        "src.ops.numpy.argsort",
        "src.ops.numpy.Array",
        "src.ops.numpy.array",
        "src.ops.numpy.View",
        "src.ops.numpy.view",
        "src.ops.numpy.Average",
        "src.ops.numpy.average",
        "src.ops.numpy.Bartlett",
        "src.ops.numpy.bartlett",
        "src.ops.numpy.Hamming",
        "src.ops.numpy.hamming",
        "src.ops.numpy.Hanning",
        "src.ops.numpy.hanning",
        "src.ops.numpy.Heaviside",
        "src.ops.numpy.heaviside",
        "src.ops.numpy.Kaiser",
        "src.ops.numpy.kaiser",
        "src.ops.numpy.Bincount",
        "src.ops.numpy.bincount",
        "src.ops.numpy.BitwiseAnd",
        "src.ops.numpy.bitwise_and",
        "src.ops.numpy.BitwiseInvert",
        "src.ops.numpy.bitwise_invert",
        "src.ops.numpy.BitwiseNot",
        "src.ops.numpy.bitwise_not",
        "src.ops.numpy.BitwiseOr",
        "src.ops.numpy.bitwise_or",
        "src.ops.numpy.BitwiseXor",
        "src.ops.numpy.bitwise_xor",
        "src.ops.numpy.BitwiseLeftShift",
        "src.ops.numpy.bitwise_left_shift",
        "src.ops.numpy.LeftShift",
        "src.ops.numpy.left_shift",
        "src.ops.numpy.BitwiseRightShift",
        "src.ops.numpy.bitwise_right_shift",
        "src.ops.numpy.RightShift",
        "src.ops.numpy.right_shift",
        "src.ops.numpy.Blackman",
        "src.ops.numpy.blackman",
        "src.ops.numpy.BroadcastTo",
        "src.ops.numpy.broadcast_to",
        "src.ops.numpy.Cbrt",
        "src.ops.numpy.cbrt",
        "src.ops.numpy.Ceil",
        "src.ops.numpy.ceil",
        "src.ops.numpy.Clip",
        "src.ops.numpy.clip",
        "src.ops.numpy.Concatenate",
        "src.ops.numpy.concatenate",
        "src.ops.numpy.Conjugate",
        "src.ops.numpy.conjugate",
        "src.ops.numpy.Conj",
        "src.ops.numpy.conj",
        "src.ops.numpy.Copy",
        "src.ops.numpy.copy",
        "src.ops.numpy.Cos",
        "src.ops.numpy.cos",
        "src.ops.numpy.Cosh",
        "src.ops.numpy.cosh",
        "src.ops.numpy.CountNonzero",
        "src.ops.numpy.count_nonzero",
        "src.ops.numpy.Cross",
        "src.ops.numpy.cross",
        "src.ops.numpy.Cumprod",
        "src.ops.numpy.cumprod",
        "src.ops.numpy.Cumsum",
        "src.ops.numpy.cumsum",
        "src.ops.numpy.Deg2rad",
        "src.ops.numpy.deg2rad",
        "src.ops.numpy.Diag",
        "src.ops.numpy.diag",
        "src.ops.numpy.Diagflat",
        "src.ops.numpy.diagflat",
        "src.ops.numpy.Diagonal",
        "src.ops.numpy.diagonal",
        "src.ops.numpy.Diff",
        "src.ops.numpy.diff",
        "src.ops.numpy.Digitize",
        "src.ops.numpy.digitize",
        "src.ops.numpy.Dot",
        "src.ops.numpy.dot",
        "src.ops.numpy.Einsum",
        "src.ops.numpy.einsum",
        "src.ops.numpy.empty",
        "src.ops.numpy.EmptyLike",
        "src.ops.numpy.empty_like",
        "src.ops.numpy.Equal",
        "src.ops.numpy.equal",
        "src.ops.numpy.Exp",
        "src.ops.numpy.exp",
        "src.ops.numpy.Exp2",
        "src.ops.numpy.exp2",
        "src.ops.numpy.ExpandDims",
        "src.ops.numpy.expand_dims",
        "src.ops.numpy.Expm1",
        "src.ops.numpy.expm1",
        "src.ops.numpy.Flip",
        "src.ops.numpy.flip",
        "src.ops.numpy.Floor",
        "src.ops.numpy.floor",
        "src.ops.numpy.Full",
        "src.ops.numpy.full",
        "src.ops.numpy.FullLike",
        "src.ops.numpy.full_like",
        "src.ops.numpy.Gcd",
        "src.ops.numpy.gcd",
        "src.ops.numpy.GetItem",
        "src.ops.numpy.get_item",
        "src.ops.numpy.Greater",
        "src.ops.numpy.greater",
        "src.ops.numpy.GreaterEqual",
        "src.ops.numpy.greater_equal",
        "src.ops.numpy.Hstack",
        "src.ops.numpy.hstack",
        "src.ops.numpy.Hypot",
        "src.ops.numpy.hypot",
        "src.ops.numpy.identity",
        "src.ops.numpy.Imag",
        "src.ops.numpy.imag",
        "src.ops.numpy.Isclose",
        "src.ops.numpy.isclose",
        "src.ops.numpy.Isfinite",
        "src.ops.numpy.isfinite",
        "src.ops.numpy.IsIn",
        "src.ops.numpy.isin",
        "src.ops.numpy.Isinf",
        "src.ops.numpy.isinf",
        "src.ops.numpy.Isnan",
        "src.ops.numpy.isnan",
        "src.ops.numpy.Isneginf",
        "src.ops.numpy.isneginf",
        "src.ops.numpy.Isposinf",
        "src.ops.numpy.isposinf",
        "src.ops.numpy.Isreal",
        "src.ops.numpy.isreal",
        "src.ops.numpy.Kron",
        "src.ops.numpy.kron",
        "src.ops.numpy.Lcm",
        "src.ops.numpy.lcm",
        "src.ops.numpy.Ldexp",
        "src.ops.numpy.ldexp",
        "src.ops.numpy.Less",
        "src.ops.numpy.less",
        "src.ops.numpy.LessEqual",
        "src.ops.numpy.less_equal",
        "src.ops.numpy.Linspace",
        "src.ops.numpy.linspace",
        "src.ops.numpy.Log",
        "src.ops.numpy.log",
        "src.ops.numpy.Log10",
        "src.ops.numpy.log10",
        "src.ops.numpy.Log1p",
        "src.ops.numpy.log1p",
        "src.ops.numpy.Log2",
        "src.ops.numpy.log2",
        "src.ops.numpy.Logaddexp",
        "src.ops.numpy.logaddexp",
        "src.ops.numpy.Logaddexp2",
        "src.ops.numpy.logaddexp2",
        "src.ops.numpy.LogicalAnd",
        "src.ops.numpy.logical_and",
        "src.ops.numpy.LogicalNot",
        "src.ops.numpy.logical_not",
        "src.ops.numpy.LogicalOr",
        "src.ops.numpy.logical_or",
        "src.ops.numpy.Logspace",
        "src.ops.numpy.logspace",
        "src.ops.numpy.Matmul",
        "src.ops.numpy.matmul",
        "src.ops.numpy.Max",
        "src.ops.numpy.max",
        "src.ops.numpy.Maximum",
        "src.ops.numpy.maximum",
        "src.ops.numpy.Median",
        "src.ops.numpy.median",
        "src.ops.numpy.Meshgrid",
        "src.ops.numpy.meshgrid",
        "src.ops.numpy.Min",
        "src.ops.numpy.min",
        "src.ops.numpy.Minimum",
        "src.ops.numpy.minimum",
        "src.ops.numpy.Mod",
        "src.ops.numpy.mod",
        "src.ops.numpy.Moveaxis",
        "src.ops.numpy.moveaxis",
        "src.ops.numpy.NanToNum",
        "src.ops.numpy.nan_to_num",
        "src.ops.numpy.Ndim",
        "src.ops.numpy.ndim",
        "src.ops.numpy.Nonzero",
        "src.ops.numpy.nonzero",
        "src.ops.numpy.NotEqual",
        "src.ops.numpy.not_equal",
        "src.ops.numpy.OnesLike",
        "src.ops.numpy.ones_like",
        "src.ops.numpy.ZerosLike",
        "src.ops.numpy.zeros_like",
        "src.ops.numpy.Outer",
        "src.ops.numpy.outer",
        "src.ops.numpy.Pad",
        "src.ops.numpy.pad",
        "src.ops.numpy.Prod",
        "src.ops.numpy.prod",
        "src.ops.numpy.Quantile",
        "src.ops.numpy.quantile",
        "src.ops.numpy.Ravel",
        "src.ops.numpy.ravel",
        "src.ops.numpy.UnravelIndex",
        "src.ops.numpy.unravel_index",
        "src.ops.numpy.Real",
        "src.ops.numpy.real",
        "src.ops.numpy.Reciprocal",
        "src.ops.numpy.reciprocal",
        "src.ops.numpy.Repeat",
        "src.ops.numpy.repeat",
        "src.ops.numpy.Reshape",
        "src.ops.numpy.reshape",
        "src.ops.numpy.Roll",
        "src.ops.numpy.roll",
        "src.ops.numpy.Round",
        "src.ops.numpy.round",
        "src.ops.numpy.SearchSorted",
        "src.ops.numpy.searchsorted",
        "src.ops.numpy.Sign",
        "src.ops.numpy.sign",
        "src.ops.numpy.Signbit",
        "src.ops.numpy.signbit",
        "src.ops.numpy.Sin",
        "src.ops.numpy.sin",
        "src.ops.numpy.Sinh",
        "src.ops.numpy.sinh",
        "src.ops.numpy.Size",
        "src.ops.numpy.size",
        "src.ops.numpy.Sort",
        "src.ops.numpy.sort",
        "src.ops.numpy.Split",
        "src.ops.numpy.split",
        "src.ops.numpy.Stack",
        "src.ops.numpy.stack",
        "src.ops.numpy.Std",
        "src.ops.numpy.std",
        "src.ops.numpy.Swapaxes",
        "src.ops.numpy.swapaxes",
        "src.ops.numpy.Take",
        "src.ops.numpy.take",
        "src.ops.numpy.TakeAlongAxis",
        "src.ops.numpy.take_along_axis",
        "src.ops.numpy.Tan",
        "src.ops.numpy.tan",
        "src.ops.numpy.Tanh",
        "src.ops.numpy.tanh",
        "src.ops.numpy.Tensordot",
        "src.ops.numpy.tensordot",
        "src.ops.numpy.Tile",
        "src.ops.numpy.tile",
        "src.ops.numpy.Trace",
        "src.ops.numpy.trace",
        "src.ops.numpy.tri",
        "src.ops.numpy.Tril",
        "src.ops.numpy.tril",
        "src.ops.numpy.Triu",
        "src.ops.numpy.triu",
        "src.ops.numpy.Trunc",
        "src.ops.numpy.trunc",
        "src.ops.numpy.Vdot",
        "src.ops.numpy.vdot",
        "src.ops.numpy.Inner",
        "src.ops.numpy.inner",
        "src.ops.numpy.vectorize",
        "src.ops.numpy.Vstack",
        "src.ops.numpy.vstack",
        "src.ops.numpy.Where",
        "src.ops.numpy.where",
        "src.ops.numpy.Subtract",
        "src.ops.numpy.subtract",
        "src.ops.numpy.Multiply",
        "src.ops.numpy.multiply",
        "src.ops.numpy.Divide",
        "src.ops.numpy.divide",
        "src.ops.numpy.DivideNoNan",
        "src.ops.numpy.divide_no_nan",
        "src.ops.numpy.TrueDivide",
        "src.ops.numpy.true_divide",
        "src.ops.numpy.Power",
        "src.ops.numpy.power",
        "src.ops.numpy.Negative",
        "src.ops.numpy.negative",
        "src.ops.numpy.Square",
        "src.ops.numpy.square",
        "src.ops.numpy.Sqrt",
        "src.ops.numpy.sqrt",
        "src.ops.numpy.Squeeze",
        "src.ops.numpy.squeeze",
        "src.ops.numpy.Transpose",
        "src.ops.numpy.transpose",
        "src.ops.numpy.Trapezoid",
        "src.ops.numpy.trapezoid",
        "src.ops.numpy.Mean",
        "src.ops.numpy.mean",
        "src.ops.numpy.Vander",
        "src.ops.numpy.vander",
        "src.ops.numpy.Var",
        "src.ops.numpy.var",
        "src.ops.numpy.Sum",
        "src.ops.numpy.sum",
        "src.ops.numpy.zeros",
        "src.ops.numpy.ones",
        "src.ops.numpy.eye",
        "src.ops.numpy.FloorDivide",
        "src.ops.numpy.floor_divide",
        "src.ops.numpy.LogicalXor",
        "src.ops.numpy.logical_xor",
        "src.ops.numpy.Corrcoef",
        "src.ops.numpy.corrcoef",
        "src.ops.numpy.Correlate",
        "src.ops.numpy.correlate",
        "src.ops.numpy.Select",
        "src.ops.numpy.select",
        "src.ops.numpy.Slogdet",
        "src.ops.numpy.slogdet",
        "src.ops.numpy.Argpartition",
        "src.ops.numpy.argpartition",
        "src.ops.numpy.Histogram",
        "src.ops.numpy.histogram",
        "src.ops.numpy.ArraySplit",
        "src.ops.numpy.array_split",
        "src.ops.operation.backend",
        "src.ops.operation.dtype_policies",
        "src.ops.operation.tree",
        "src.ops.operation.keras_export",
        "src.ops.operation.any_symbolic_tensors",
        "src.ops.operation.is_nnx_enabled",
        "src.ops.operation.Node",
        "src.ops.operation.KerasSaveable",
        "src.ops.operation.python_utils",
        "src.ops.operation.traceback_utils",
        "src.ops.operation.auto_name",
        "src.ops.operation.Operation",
        "src.ops.linalg.backend",
        "src.ops.linalg.tree",
        "src.ops.linalg.keras_export",
        "src.ops.linalg.KerasTensor",
        "src.ops.linalg.any_symbolic_tensors",
        "src.ops.linalg.Operation",
        "src.ops.linalg.reduce_shape",
        "src.ops.linalg.Cholesky",
        "src.ops.linalg.cholesky",
        "src.ops.linalg.CholeskyInverse",
        "src.ops.linalg.cholesky_inverse",
        "src.ops.linalg.Det",
        "src.ops.linalg.det",
        "src.ops.linalg.Eig",
        "src.ops.linalg.eig",
        "src.ops.linalg.Eigh",
        "src.ops.linalg.eigh",
        "src.ops.linalg.Inv",
        "src.ops.linalg.inv",
        "src.ops.linalg.LuFactor",
        "src.ops.linalg.lu_factor",
        "src.ops.linalg.Norm",
        "src.ops.linalg.norm",
        "src.ops.linalg.Qr",
        "src.ops.linalg.qr",
        "src.ops.linalg.Solve",
        "src.ops.linalg.solve",
        "src.ops.linalg.SolveTriangular",
        "src.ops.linalg.solve_triangular",
        "src.ops.linalg.SVD",
        "src.ops.linalg.svd",
        "src.ops.linalg.Lstsq",
        "src.ops.linalg.lstsq",
        "src.ops.linalg.JVP",
        "src.ops.linalg.jvp",
        "src.ops.symbolic_arguments.tree",
        "src.ops.symbolic_arguments.KerasTensor",
        "src.ops.symbolic_arguments.SymbolicArguments",
        "src.ops.math.backend",
        "src.ops.math.keras_export",
        "src.ops.math.KerasTensor",
        "src.ops.math.any_symbolic_tensors",
        "src.ops.math.Operation",
        "src.ops.math.reduce_shape",
        "src.ops.math.SegmentReduction",
        "src.ops.math.SegmentSum",
        "src.ops.math.segment_sum",
        "src.ops.math.SegmentMax",
        "src.ops.math.segment_max",
        "src.ops.math.TopK",
        "src.ops.math.top_k",
        "src.ops.math.InTopK",
        "src.ops.math.in_top_k",
        "src.ops.math.Logsumexp",
        "src.ops.math.logsumexp",
        "src.ops.math.ExtractSequences",
        "src.ops.math.extract_sequences",
        "src.ops.math.FFT",
        "src.ops.math.fft",
        "src.ops.math.FFT2",
        "src.ops.math.fft2",
        "src.ops.math.IFFT2",
        "src.ops.math.ifft2",
        "src.ops.math.RFFT",
        "src.ops.math.rfft",
        "src.ops.math.IRFFT",
        "src.ops.math.irfft",
        "src.ops.math.STFT",
        "src.ops.math.stft",
        "src.ops.math.ISTFT",
        "src.ops.math.istft",
        "src.ops.math.Rsqrt",
        "src.ops.math.rsqrt",
        "src.ops.math.Erf",
        "src.ops.math.erf",
        "src.ops.math.Erfinv",
        "src.ops.math.erfinv",
        "src.ops.math.Logdet",
        "src.ops.math.logdet",
        "src.ops.math.ViewAsComplex",
        "src.ops.math.ViewAsReal",
        "src.ops.math.view_as_complex",
        "src.ops.math.view_as_real",
        "src.ops.operation_utils.tree",
        "src.ops.operation_utils.keras_export",
        "src.ops.operation_utils.canonicalize_axis",
        "src.ops.operation_utils.to_tuple_or_list",
        "src.ops.operation_utils.broadcast_shapes",
        "src.ops.operation_utils.compute_expand_dims_output_shape",
        "src.ops.operation_utils.compute_pooling_output_shape",
        "src.ops.operation_utils.compute_conv_output_shape",
        "src.ops.operation_utils.compute_matmul_output_shape",
        "src.ops.operation_utils.compute_reshape_output_shape",
        "src.ops.operation_utils.compute_transpose_output_shape",
        "src.ops.operation_utils.compute_take_along_axis_output_shape",
        "src.ops.operation_utils.reduce_shape",
        "src.ops.operation_utils.get_source_inputs",
        "src.ops.node.tree",
        "src.ops.node.KerasTensor",
        "src.ops.node.SymbolicArguments",
        "src.ops.node.Node",
        "src.ops.node.KerasHistory",
        "src.ops.node.is_keras_tensor",
        "src.ops.image.backend",
        "src.ops.image.ops",
        "src.ops.image.keras_export",
        "src.ops.image.KerasTensor",
        "src.ops.image.any_symbolic_tensors",
        "src.ops.image.Operation",
        "src.ops.image.compute_conv_output_shape",
        "src.ops.image.RGBToGrayscale",
        "src.ops.image.rgb_to_grayscale",
        "src.ops.image.RGBToHSV",
        "src.ops.image.rgb_to_hsv",
        "src.ops.image.HSVToRGB",
        "src.ops.image.hsv_to_rgb",
        "src.ops.image.Resize",
        "src.ops.image.resize",
        "src.ops.image.AffineTransform",
        "src.ops.image.affine_transform",
        "src.ops.image.ExtractPatches",
        "src.ops.image.extract_patches",
        "src.ops.image.ExtractPatches3D",
        "src.ops.image.extract_patches_3d",
        "src.ops.image.MapCoordinates",
        "src.ops.image.map_coordinates",
        "src.ops.image.PadImages",
        "src.ops.image.pad_images",
        "src.ops.image.CropImages",
        "src.ops.image.crop_images",
        "src.ops.image.PerspectiveTransform",
        "src.ops.image.perspective_transform",
        "src.ops.image.GaussianBlur",
        "src.ops.image.gaussian_blur",
        "src.ops.image.ElasticTransform",
        "src.ops.image.elastic_transform",
        "src.ops.image.ScaleAndTranslate",
        "src.ops.image.scale_and_translate",
        "src.ops.function.tree",
        "src.ops.function.keras_export",
        "src.ops.function.KerasTensor",
        "src.ops.function.backend",
        "src.ops.function.is_nnx_enabled",
        "src.ops.function.Operation",
        "src.ops.function.Function",
        "src.ops.function.make_node_key",
        "src.ops.function.map_graph",
        "src.ops.ml_dtypes",
        "src.ops.np",
        "src.ops.backend",
        "src.ops.tree",
        "src.ops.keras_export",
        "src.ops.KerasTensor",
        "src.ops.any_symbolic_tensors",
        "src.ops.slice_along_axis",
        "src.ops.Operation",
        "src.ops.serialization_lib",
        "src.ops.traceback_utils",
        "src.ops.Map",
        "src.ops.map",
        "src.ops.Scan",
        "src.ops.scan",
        "src.ops.AssociativeScan",
        "src.ops.associative_scan",
        "src.ops.Scatter",
        "src.ops.scatter",
        "src.ops.ScatterUpdate",
        "src.ops.scatter_update",
        "src.ops.Slice",
        "src.ops.slice",
        "src.ops.SliceUpdate",
        "src.ops.slice_update",
        "src.ops.Switch",
        "src.ops.switch",
        "src.ops.WhileLoop",
        "src.ops.while_loop",
        "src.ops.StopGradient",
        "src.ops.stop_gradient",
        "src.ops.ForiLoop",
        "src.ops.fori_loop",
        "src.ops.Unstack",
        "src.ops.unstack",
        "src.ops.shape",
        "src.ops.dtype",
        "src.ops.Cast",
        "src.ops.SaturateCast",
        "src.ops.saturate_cast",
        "src.ops.ConvertToTensor",
        "src.ops.convert_to_tensor",
        "src.ops.convert_to_numpy",
        "src.ops.Cond",
        "src.ops.VectorizedMap",
        "src.ops.vectorized_map",
        "src.ops.custom_gradient",
        "src.ops.reduce_shape",
        "src.ops.Cholesky",
        "src.ops.cholesky",
        "src.ops.CholeskyInverse",
        "src.ops.cholesky_inverse",
        "src.ops.Det",
        "src.ops.det",
        "src.ops.Eig",
        "src.ops.eig",
        "src.ops.Eigh",
        "src.ops.eigh",
        "src.ops.Inv",
        "src.ops.inv",
        "src.ops.LuFactor",
        "src.ops.lu_factor",
        "src.ops.Norm",
        "src.ops.norm",
        "src.ops.Qr",
        "src.ops.qr",
        "src.ops.Solve",
        "src.ops.solve",
        "src.ops.SolveTriangular",
        "src.ops.solve_triangular",
        "src.ops.SVD",
        "src.ops.svd",
        "src.ops.Lstsq",
        "src.ops.lstsq",
        "src.ops.JVP",
        "src.ops.jvp",
        "src.ops.SegmentReduction",
        "src.ops.SegmentSum",
        "src.ops.segment_sum",
        "src.ops.SegmentMax",
        "src.ops.segment_max",
        "src.ops.TopK",
        "src.ops.top_k",
        "src.ops.InTopK",
        "src.ops.in_top_k",
        "src.ops.Logsumexp",
        "src.ops.logsumexp",
        "src.ops.ExtractSequences",
        "src.ops.extract_sequences",
        "src.ops.FFT",
        "src.ops.fft",
        "src.ops.FFT2",
        "src.ops.fft2",
        "src.ops.IFFT2",
        "src.ops.ifft2",
        "src.ops.RFFT",
        "src.ops.rfft",
        "src.ops.IRFFT",
        "src.ops.irfft",
        "src.ops.STFT",
        "src.ops.stft",
        "src.ops.ISTFT",
        "src.ops.istft",
        "src.ops.Rsqrt",
        "src.ops.rsqrt",
        "src.ops.Erf",
        "src.ops.erf",
        "src.ops.Erfinv",
        "src.ops.erfinv",
        "src.ops.Logdet",
        "src.ops.logdet",
        "src.ops.ViewAsComplex",
        "src.ops.ViewAsReal",
        "src.ops.view_as_complex",
        "src.ops.view_as_real",
        "src.ops.warnings",
        "src.ops.config",
        "src.ops.standardize_data_format",
        "src.ops.compute_conv_transpose_output_shape",
        "src.ops.is_continuous_axis",
        "src.ops.Relu",
        "src.ops.relu",
        "src.ops.Relu6",
        "src.ops.relu6",
        "src.ops.Sigmoid",
        "src.ops.sigmoid",
        "src.ops.SparseSigmoid",
        "src.ops.sparse_sigmoid",
        "src.ops.Softplus",
        "src.ops.softplus",
        "src.ops.Softsign",
        "src.ops.softsign",
        "src.ops.SoftShrink",
        "src.ops.soft_shrink",
        "src.ops.SparsePlus",
        "src.ops.sparse_plus",
        "src.ops.Silu",
        "src.ops.silu",
        "src.ops.Squareplus",
        "src.ops.squareplus",
        "src.ops.LogSigmoid",
        "src.ops.log_sigmoid",
        "src.ops.LeakyRelu",
        "src.ops.leaky_relu",
        "src.ops.HardSigmoid",
        "src.ops.hard_sigmoid",
        "src.ops.HardSilu",
        "src.ops.hard_silu",
        "src.ops.Elu",
        "src.ops.elu",
        "src.ops.Selu",
        "src.ops.selu",
        "src.ops.Gelu",
        "src.ops.gelu",
        "src.ops.Celu",
        "src.ops.celu",
        "src.ops.Glu",
        "src.ops.glu",
        "src.ops.TanhShrink",
        "src.ops.tanh_shrink",
        "src.ops.HardTanh",
        "src.ops.hard_tanh",
        "src.ops.HardShrink",
        "src.ops.hard_shrink",
        "src.ops.Threshold",
        "src.ops.threshold",
        "src.ops.Softmax",
        "src.ops.softmax",
        "src.ops.LogSoftmax",
        "src.ops.log_softmax",
        "src.ops.Sparsemax",
        "src.ops.sparsemax",
        "src.ops.MaxPool",
        "src.ops.max_pool",
        "src.ops.AdaptiveMaxPool",
        "src.ops.adaptive_max_pool",
        "src.ops.AveragePool",
        "src.ops.average_pool",
        "src.ops.AdaptiveAveragePool",
        "src.ops.adaptive_average_pool",
        "src.ops.Conv",
        "src.ops.conv",
        "src.ops.DepthwiseConv",
        "src.ops.depthwise_conv",
        "src.ops.SeparableConv",
        "src.ops.separable_conv",
        "src.ops.ConvTranspose",
        "src.ops.conv_transpose",
        "src.ops.OneHot",
        "src.ops.one_hot",
        "src.ops.BinaryCrossentropy",
        "src.ops.binary_crossentropy",
        "src.ops.CategoricalCrossentropy",
        "src.ops.categorical_crossentropy",
        "src.ops.SparseCategoricalCrossentropy",
        "src.ops.sparse_categorical_crossentropy",
        "src.ops.MultiHot",
        "src.ops.multi_hot",
        "src.ops.Moments",
        "src.ops.moments",
        "src.ops.BatchNorm",
        "src.ops.batch_normalization",
        "src.ops.CTCLoss",
        "src.ops.ctc_loss",
        "src.ops.CTCDecode",
        "src.ops.ctc_decode",
        "src.ops.Normalize",
        "src.ops.normalize",
        "src.ops.PSNR",
        "src.ops.psnr",
        "src.ops.DotProductAttention",
        "src.ops.dot_product_attention",
        "src.ops.RMSNorm",
        "src.ops.rms_normalization",
        "src.ops.LayerNorm",
        "src.ops.layer_normalization",
        "src.ops.Polar",
        "src.ops.polar",
        "src.ops.Unfold",
        "src.ops.unfold",
        "src.ops.builtins",
        "src.ops.re",
        "src.ops.dtypes",
        "src.ops.canonicalize_axis",
        "src.ops.to_tuple_or_list",
        "src.ops.broadcast_shapes",
        "src.ops.Rot90",
        "src.ops.rot90",
        "src.ops.shape_equal",
        "src.ops.Absolute",
        "src.ops.absolute",
        "src.ops.Abs",
        "src.ops.abs",
        "src.ops.Add",
        "src.ops.add",
        "src.ops.All",
        "src.ops.all",
        "src.ops.Angle",
        "src.ops.angle",
        "src.ops.Any",
        "src.ops.any",
        "src.ops.Amax",
        "src.ops.amax",
        "src.ops.Amin",
        "src.ops.amin",
        "src.ops.Append",
        "src.ops.append",
        "src.ops.Arange",
        "src.ops.arange",
        "src.ops.Arccos",
        "src.ops.arccos",
        "src.ops.Arccosh",
        "src.ops.arccosh",
        "src.ops.Arcsin",
        "src.ops.arcsin",
        "src.ops.Arcsinh",
        "src.ops.arcsinh",
        "src.ops.Arctan",
        "src.ops.arctan",
        "src.ops.Arctan2",
        "src.ops.arctan2",
        "src.ops.Arctanh",
        "src.ops.arctanh",
        "src.ops.Argmax",
        "src.ops.argmax",
        "src.ops.Argmin",
        "src.ops.argmin",
        "src.ops.Argsort",
        "src.ops.argsort",
        "src.ops.Array",
        "src.ops.array",
        "src.ops.View",
        "src.ops.view",
        "src.ops.Average",
        "src.ops.average",
        "src.ops.Bartlett",
        "src.ops.bartlett",
        "src.ops.Hamming",
        "src.ops.hamming",
        "src.ops.Hanning",
        "src.ops.hanning",
        "src.ops.Heaviside",
        "src.ops.heaviside",
        "src.ops.Kaiser",
        "src.ops.kaiser",
        "src.ops.Bincount",
        "src.ops.bincount",
        "src.ops.BitwiseAnd",
        "src.ops.bitwise_and",
        "src.ops.BitwiseInvert",
        "src.ops.bitwise_invert",
        "src.ops.BitwiseNot",
        "src.ops.bitwise_not",
        "src.ops.BitwiseOr",
        "src.ops.bitwise_or",
        "src.ops.BitwiseXor",
        "src.ops.bitwise_xor",
        "src.ops.BitwiseLeftShift",
        "src.ops.bitwise_left_shift",
        "src.ops.LeftShift",
        "src.ops.left_shift",
        "src.ops.BitwiseRightShift",
        "src.ops.bitwise_right_shift",
        "src.ops.RightShift",
        "src.ops.right_shift",
        "src.ops.Blackman",
        "src.ops.blackman",
        "src.ops.BroadcastTo",
        "src.ops.broadcast_to",
        "src.ops.Cbrt",
        "src.ops.cbrt",
        "src.ops.Ceil",
        "src.ops.ceil",
        "src.ops.Clip",
        "src.ops.clip",
        "src.ops.Concatenate",
        "src.ops.concatenate",
        "src.ops.Conjugate",
        "src.ops.conjugate",
        "src.ops.Conj",
        "src.ops.conj",
        "src.ops.Copy",
        "src.ops.copy",
        "src.ops.Cos",
        "src.ops.cos",
        "src.ops.Cosh",
        "src.ops.cosh",
        "src.ops.CountNonzero",
        "src.ops.count_nonzero",
        "src.ops.Cross",
        "src.ops.cross",
        "src.ops.Cumprod",
        "src.ops.cumprod",
        "src.ops.Cumsum",
        "src.ops.cumsum",
        "src.ops.Deg2rad",
        "src.ops.deg2rad",
        "src.ops.Diag",
        "src.ops.diag",
        "src.ops.Diagflat",
        "src.ops.diagflat",
        "src.ops.Diagonal",
        "src.ops.diagonal",
        "src.ops.Diff",
        "src.ops.diff",
        "src.ops.Digitize",
        "src.ops.digitize",
        "src.ops.Dot",
        "src.ops.dot",
        "src.ops.Einsum",
        "src.ops.einsum",
        "src.ops.empty",
        "src.ops.EmptyLike",
        "src.ops.empty_like",
        "src.ops.Equal",
        "src.ops.equal",
        "src.ops.Exp",
        "src.ops.exp",
        "src.ops.Exp2",
        "src.ops.exp2",
        "src.ops.ExpandDims",
        "src.ops.expand_dims",
        "src.ops.Expm1",
        "src.ops.expm1",
        "src.ops.Flip",
        "src.ops.flip",
        "src.ops.Floor",
        "src.ops.floor",
        "src.ops.Full",
        "src.ops.full",
        "src.ops.FullLike",
        "src.ops.full_like",
        "src.ops.Gcd",
        "src.ops.gcd",
        "src.ops.GetItem",
        "src.ops.get_item",
        "src.ops.Greater",
        "src.ops.greater",
        "src.ops.GreaterEqual",
        "src.ops.greater_equal",
        "src.ops.Hstack",
        "src.ops.hstack",
        "src.ops.Hypot",
        "src.ops.hypot",
        "src.ops.identity",
        "src.ops.Imag",
        "src.ops.imag",
        "src.ops.Isclose",
        "src.ops.isclose",
        "src.ops.Isfinite",
        "src.ops.isfinite",
        "src.ops.IsIn",
        "src.ops.isin",
        "src.ops.Isinf",
        "src.ops.isinf",
        "src.ops.Isnan",
        "src.ops.isnan",
        "src.ops.Isneginf",
        "src.ops.isneginf",
        "src.ops.Isposinf",
        "src.ops.isposinf",
        "src.ops.Isreal",
        "src.ops.isreal",
        "src.ops.Kron",
        "src.ops.kron",
        "src.ops.Lcm",
        "src.ops.lcm",
        "src.ops.Ldexp",
        "src.ops.ldexp",
        "src.ops.Less",
        "src.ops.less",
        "src.ops.LessEqual",
        "src.ops.less_equal",
        "src.ops.Linspace",
        "src.ops.linspace",
        "src.ops.Log",
        "src.ops.log",
        "src.ops.Log10",
        "src.ops.log10",
        "src.ops.Log1p",
        "src.ops.log1p",
        "src.ops.Log2",
        "src.ops.log2",
        "src.ops.Logaddexp",
        "src.ops.logaddexp",
        "src.ops.Logaddexp2",
        "src.ops.logaddexp2",
        "src.ops.LogicalAnd",
        "src.ops.logical_and",
        "src.ops.LogicalNot",
        "src.ops.logical_not",
        "src.ops.LogicalOr",
        "src.ops.logical_or",
        "src.ops.Logspace",
        "src.ops.logspace",
        "src.ops.Matmul",
        "src.ops.matmul",
        "src.ops.Max",
        "src.ops.max",
        "src.ops.Maximum",
        "src.ops.maximum",
        "src.ops.Median",
        "src.ops.median",
        "src.ops.Meshgrid",
        "src.ops.meshgrid",
        "src.ops.Min",
        "src.ops.min",
        "src.ops.Minimum",
        "src.ops.minimum",
        "src.ops.Mod",
        "src.ops.mod",
        "src.ops.Moveaxis",
        "src.ops.moveaxis",
        "src.ops.NanToNum",
        "src.ops.nan_to_num",
        "src.ops.Ndim",
        "src.ops.ndim",
        "src.ops.Nonzero",
        "src.ops.nonzero",
        "src.ops.NotEqual",
        "src.ops.not_equal",
        "src.ops.OnesLike",
        "src.ops.ones_like",
        "src.ops.ZerosLike",
        "src.ops.zeros_like",
        "src.ops.Outer",
        "src.ops.outer",
        "src.ops.Pad",
        "src.ops.pad",
        "src.ops.Prod",
        "src.ops.prod",
        "src.ops.Quantile",
        "src.ops.quantile",
        "src.ops.Ravel",
        "src.ops.ravel",
        "src.ops.UnravelIndex",
        "src.ops.unravel_index",
        "src.ops.Real",
        "src.ops.real",
        "src.ops.Reciprocal",
        "src.ops.reciprocal",
        "src.ops.Repeat",
        "src.ops.repeat",
        "src.ops.Reshape",
        "src.ops.reshape",
        "src.ops.Roll",
        "src.ops.roll",
        "src.ops.Round",
        "src.ops.round",
        "src.ops.SearchSorted",
        "src.ops.searchsorted",
        "src.ops.Sign",
        "src.ops.sign",
        "src.ops.Signbit",
        "src.ops.signbit",
        "src.ops.Sin",
        "src.ops.sin",
        "src.ops.Sinh",
        "src.ops.sinh",
        "src.ops.Size",
        "src.ops.size",
        "src.ops.Sort",
        "src.ops.sort",
        "src.ops.Split",
        "src.ops.split",
        "src.ops.Stack",
        "src.ops.stack",
        "src.ops.Std",
        "src.ops.std",
        "src.ops.Swapaxes",
        "src.ops.swapaxes",
        "src.ops.Take",
        "src.ops.take",
        "src.ops.TakeAlongAxis",
        "src.ops.take_along_axis",
        "src.ops.Tan",
        "src.ops.tan",
        "src.ops.Tanh",
        "src.ops.tanh",
        "src.ops.Tensordot",
        "src.ops.tensordot",
        "src.ops.Tile",
        "src.ops.tile",
        "src.ops.Trace",
        "src.ops.trace",
        "src.ops.tri",
        "src.ops.Tril",
        "src.ops.tril",
        "src.ops.Triu",
        "src.ops.triu",
        "src.ops.Trunc",
        "src.ops.trunc",
        "src.ops.Vdot",
        "src.ops.vdot",
        "src.ops.Inner",
        "src.ops.inner",
        "src.ops.vectorize",
        "src.ops.Vstack",
        "src.ops.vstack",
        "src.ops.Where",
        "src.ops.where",
        "src.ops.Subtract",
        "src.ops.subtract",
        "src.ops.Multiply",
        "src.ops.multiply",
        "src.ops.Divide",
        "src.ops.divide",
        "src.ops.DivideNoNan",
        "src.ops.divide_no_nan",
        "src.ops.TrueDivide",
        "src.ops.true_divide",
        "src.ops.Power",
        "src.ops.power",
        "src.ops.Negative",
        "src.ops.negative",
        "src.ops.Square",
        "src.ops.square",
        "src.ops.Sqrt",
        "src.ops.sqrt",
        "src.ops.Squeeze",
        "src.ops.squeeze",
        "src.ops.Transpose",
        "src.ops.transpose",
        "src.ops.Trapezoid",
        "src.ops.trapezoid",
        "src.ops.Mean",
        "src.ops.mean",
        "src.ops.Vander",
        "src.ops.vander",
        "src.ops.Var",
        "src.ops.var",
        "src.ops.Sum",
        "src.ops.sum",
        "src.ops.zeros",
        "src.ops.ones",
        "src.ops.eye",
        "src.ops.FloorDivide",
        "src.ops.floor_divide",
        "src.ops.LogicalXor",
        "src.ops.logical_xor",
        "src.ops.Corrcoef",
        "src.ops.corrcoef",
        "src.ops.Correlate",
        "src.ops.correlate",
        "src.ops.Select",
        "src.ops.select",
        "src.ops.Slogdet",
        "src.ops.slogdet",
        "src.ops.Argpartition",
        "src.ops.argpartition",
        "src.ops.Histogram",
        "src.ops.histogram",
        "src.ops.ArraySplit",
        "src.ops.array_split",
        "src.saving.CustomObjectScope",
        "src.saving.custom_object_scope",
        "src.saving.get_custom_objects",
        "src.saving.get_registered_name",
        "src.saving.get_registered_object",
        "src.saving.register_keras_serializable",
        "src.saving.load_model",
        "src.saving.deserialize_keras_object",
        "src.saving.serialize_keras_object",
        "src.saving.file_editor.backend",
        "src.saving.file_editor.keras_export",
        "src.saving.file_editor.saving_lib",
        "src.saving.file_editor.H5IOStore",
        "src.saving.file_editor.naming",
        "src.saving.file_editor.summary_utils",
        "src.saving.file_editor.is_ipython_notebook",
        "src.saving.file_editor.KerasFileEditor",
        "src.saving.file_editor.get_weight_spec_of_saveable",
        "src.saving.file_editor.get_weight_spec_of_container",
        "src.saving.file_editor.initialize_id_counter",
        "src.saving.file_editor.increment_id_counter",
        "src.saving.file_editor.get_id_counter",
        "src.saving.file_editor.display_weight",
        "src.saving.serialization_lib.api_export",
        "src.saving.serialization_lib.backend",
        "src.saving.serialization_lib.keras_export",
        "src.saving.serialization_lib.global_state",
        "src.saving.serialization_lib.object_registration",
        "src.saving.serialization_lib.KerasSaveable",
        "src.saving.serialization_lib.python_utils",
        "src.saving.serialization_lib.tf",
        "src.saving.serialization_lib.PLAIN_TYPES",
        "src.saving.serialization_lib.BUILTIN_MODULES",
        "src.saving.serialization_lib.LOADING_APIS",
        "src.saving.serialization_lib.SerializableDict",
        "src.saving.serialization_lib.SafeModeScope",
        "src.saving.serialization_lib.enable_unsafe_deserialization",
        "src.saving.serialization_lib.in_safe_mode",
        "src.saving.serialization_lib.ObjectSharingScope",
        "src.saving.serialization_lib.get_shared_object",
        "src.saving.serialization_lib.record_object_after_serialization",
        "src.saving.serialization_lib.record_object_after_deserialization",
        "src.saving.serialization_lib.serialize_keras_object",
        "src.saving.serialization_lib.get_build_and_compile_config",
        "src.saving.serialization_lib.serialize_with_public_class",
        "src.saving.serialization_lib.serialize_with_public_fn",
        "src.saving.serialization_lib.serialize_dict",
        "src.saving.serialization_lib.deserialize_keras_object",
        "src.saving.saving_api.keras_export",
        "src.saving.saving_api.legacy_h5_format",
        "src.saving.saving_api.saving_lib",
        "src.saving.saving_api.file_utils",
        "src.saving.saving_api.io_utils",
        "src.saving.saving_api.save_model",
        "src.saving.saving_api.load_model",
        "src.saving.saving_api.save_weights",
        "src.saving.saving_api.load_weights",
        "src.saving.keras_saveable.KerasSaveable",
        "src.saving.object_registration.keras_export",
        "src.saving.object_registration.global_state",
        "src.saving.object_registration.GLOBAL_CUSTOM_OBJECTS",
        "src.saving.object_registration.GLOBAL_CUSTOM_NAMES",
        "src.saving.object_registration.CustomObjectScope",
        "src.saving.object_registration.custom_object_scope",
        "src.saving.object_registration.get_custom_objects",
        "src.saving.object_registration.register_keras_serializable",
        "src.saving.object_registration.get_registered_name",
        "src.saving.object_registration.get_registered_object",
        "src.saving.saving_lib.backend",
        "src.saving.saving_lib.global_state",
        "src.saving.saving_lib.ObjectSharingScope",
        "src.saving.saving_lib.deserialize_keras_object",
        "src.saving.saving_lib.serialize_keras_object",
        "src.saving.saving_lib.dtype_utils",
        "src.saving.saving_lib.file_utils",
        "src.saving.saving_lib.io_utils",
        "src.saving.saving_lib.naming",
        "src.saving.saving_lib.plot_model",
        "src.saving.saving_lib.check_pydot",
        "src.saving.saving_lib.readable_memory_size",
        "src.saving.saving_lib.weight_memory_size",
        "src.saving.saving_lib.keras_version",
        "src.saving.saving_lib.save_model",
        "src.saving.saving_lib.load_model",
        "src.saving.saving_lib.save_weights_only",
        "src.saving.saving_lib.load_weights_only",
        "src.saving.saving_lib.DiskIOStore",
        "src.saving.saving_lib.H5IOStore",
        "src.saving.saving_lib.ShardedH5IOStore",
        "src.saving.saving_lib.NpzIOStore",
        "src.saving.saving_lib.get_temp_dir",
        "src.saving.saving_lib.get_attr_skipset",
        "src.saving.saving_lib.is_memory_sufficient",
        "src.export.LiteRTExporter",
        "src.export.export_litert",
        "src.export.export_onnx",
        "src.export.export_openvino",
        "src.export.ExportArchive",
        "src.export.export_saved_model",
        "src.export.TFSMLayer",
        "src.export.export_utils.backend",
        "src.export.export_utils.layers",
        "src.export.export_utils.models",
        "src.export.export_utils.ops",
        "src.export.export_utils.tree",
        "src.export.export_utils.tf",
        "src.export.export_utils.get_input_signature",
        "src.export.export_utils.make_input_spec",
        "src.export.export_utils.make_tf_tensor_spec",
        "src.export.export_utils.convert_spec_to_tensor",
        "src.export.onnx.backend",
        "src.export.onnx.tree",
        "src.export.onnx.convert_spec_to_tensor",
        "src.export.onnx.get_input_signature",
        "src.export.onnx.make_tf_tensor_spec",
        "src.export.onnx.DEFAULT_ENDPOINT_NAME",
        "src.export.onnx.ExportArchive",
        "src.export.onnx.patch_tf2onnx",
        "src.export.onnx.io_utils",
        "src.export.onnx.export_onnx",
        "src.export.onnx.get_concrete_fn",
        "src.export.openvino.backend",
        "src.export.openvino.tree",
        "src.export.openvino.convert_spec_to_tensor",
        "src.export.openvino.get_input_signature",
        "src.export.openvino.make_tf_tensor_spec",
        "src.export.openvino.DEFAULT_ENDPOINT_NAME",
        "src.export.openvino.ExportArchive",
        "src.export.openvino.io_utils",
        "src.export.openvino.export_openvino",
        "src.export.openvino.collect_names",
        "src.export.openvino.set_names",
        "src.export.openvino.get_concrete_fn",
        "src.export.tfsm_layer.backend",
        "src.export.tfsm_layer.layers",
        "src.export.tfsm_layer.keras_export",
        "src.export.tfsm_layer.serialization_lib",
        "src.export.tfsm_layer.tf",
        "src.export.tfsm_layer.TFSMLayer",
        "src.export.tf2onnx_lib.patch_tf2onnx",
        "src.export.saved_model.backend",
        "src.export.saved_model.layers",
        "src.export.saved_model.tree",
        "src.export.saved_model.keras_export",
        "src.export.saved_model.get_input_signature",
        "src.export.saved_model.make_tf_tensor_spec",
        "src.export.saved_model.io_utils",
        "src.export.saved_model.tf",
        "src.export.saved_model.BackendExportArchive",
        "src.export.saved_model.DEFAULT_ENDPOINT_NAME",
        "src.export.saved_model.ExportArchive",
        "src.export.saved_model.export_saved_model",
        "src.export.litert.layers",
        "src.export.litert.models",
        "src.export.litert.tree",
        "src.export.litert.get_input_signature",
        "src.export.litert.io_utils",
        "src.export.litert.tf",
        "src.export.litert.export_litert",
        "src.export.litert.LiteRTExporter",
        "src.random.categorical",
        "src.random.dropout",
        "src.random.gamma",
        "src.random.normal",
        "src.random.randint",
        "src.random.shuffle",
        "src.random.truncated_normal",
        "src.random.uniform",
        "src.random.SeedGenerator",
        "src.random.random.backend",
        "src.random.random.keras_export",
        "src.random.random.normal",
        "src.random.random.categorical",
        "src.random.random.uniform",
        "src.random.random.randint",
        "src.random.random.truncated_normal",
        "src.random.random.dropout",
        "src.random.random.shuffle",
        "src.random.random.gamma",
        "src.random.random.binomial",
        "src.random.random.beta",
        "src.random.seed_generator.backend",
        "src.random.seed_generator.keras_export",
        "src.random.seed_generator.global_state",
        "src.random.seed_generator.jax_utils",
        "src.random.seed_generator.auto_name",
        "src.random.seed_generator.GLOBAL_SEED_GENERATOR",
        "src.random.seed_generator.SeedGenerator",
        "src.random.seed_generator.global_seed_generator",
        "src.random.seed_generator.make_default_seed",
        "src.random.seed_generator.draw_seed",
        "src.initializers.backend",
        "src.initializers.ops",
        "src.initializers.keras_export",
        "src.initializers.STFT",
        "src.initializers.Constant",
        "src.initializers.Identity",
        "src.initializers.Ones",
        "src.initializers.Zeros",
        "src.initializers.Initializer",
        "src.initializers.GlorotNormal",
        "src.initializers.GlorotUniform",
        "src.initializers.HeNormal",
        "src.initializers.HeUniform",
        "src.initializers.LecunNormal",
        "src.initializers.LecunUniform",
        "src.initializers.Orthogonal",
        "src.initializers.RandomNormal",
        "src.initializers.RandomUniform",
        "src.initializers.TruncatedNormal",
        "src.initializers.VarianceScaling",
        "src.initializers.serialization_lib",
        "src.initializers.to_snake_case",
        "src.initializers.ALL_OBJECTS",
        "src.initializers.ALL_OBJECTS_DICT",
        "src.initializers.serialize",
        "src.initializers.deserialize",
        "src.initializers.get",
        "src.initializers.constant_initializers.ops",
        "src.initializers.constant_initializers.keras_export",
        "src.initializers.constant_initializers.standardize_dtype",
        "src.initializers.constant_initializers.Initializer",
        "src.initializers.constant_initializers.serialization_lib",
        "src.initializers.constant_initializers.scipy",
        "src.initializers.constant_initializers.Constant",
        "src.initializers.constant_initializers.Zeros",
        "src.initializers.constant_initializers.Ones",
        "src.initializers.constant_initializers.Identity",
        "src.initializers.constant_initializers.STFT",
        "src.initializers.random_initializers.ops",
        "src.initializers.random_initializers.keras_export",
        "src.initializers.random_initializers.random",
        "src.initializers.random_initializers.Initializer",
        "src.initializers.random_initializers.serialization_lib",
        "src.initializers.random_initializers.RandomInitializer",
        "src.initializers.random_initializers.RandomNormal",
        "src.initializers.random_initializers.TruncatedNormal",
        "src.initializers.random_initializers.RandomUniform",
        "src.initializers.random_initializers.VarianceScaling",
        "src.initializers.random_initializers.GlorotUniform",
        "src.initializers.random_initializers.GlorotNormal",
        "src.initializers.random_initializers.LecunNormal",
        "src.initializers.random_initializers.LecunUniform",
        "src.initializers.random_initializers.HeNormal",
        "src.initializers.random_initializers.HeUniform",
        "src.initializers.random_initializers.compute_fans",
        "src.initializers.random_initializers.Orthogonal",
        "src.initializers.initializer.keras_export",
        "src.initializers.initializer.Initializer",
    ]
    for name in names:
        node = TFNode("test_semantic", name, inputs=["a", "b", "c"])
        initial_len = len(builder.graph.nodes)
        KERAS_LAYERS_MAPPING[name](builder, node)
        assert len(builder.graph.nodes) > initial_len
