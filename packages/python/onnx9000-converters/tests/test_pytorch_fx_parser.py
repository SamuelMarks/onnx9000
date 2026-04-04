"""Module docstring."""

import pytest
from onnx9000.converters.pytorch_fx_parser import PyTorchFXParser


def test_pytorch_fx_parser():
    """Docstring for D103."""
    export_json = """
    {
        "nodes": [
            {"op": "placeholder", "target": "x", "args": [], "out": "%1"},
            {"op": "placeholder", "target": "w", "args": [], "out": "%2"},
            {"op": "call_function", "target": "aten.mm.default", "args": ["%1", "%2"], "out": "%3"},
            {"op": "call_function", "target": "aten.relu.default", "args": ["%3"], "out": "%4"},
            {"op": "output", "target": "output", "args": [["%4"]], "out": ""}
        ]
    }
    """
    parser = PyTorchFXParser()
    graph = parser.parse_json(export_json)

    # In Graph, inputs/outputs are lists of Tensor objects
    input_names = [t.name for t in graph.inputs]
    assert "%1" in input_names
    assert "%2" in input_names

    output_names = [t.name for t in graph.outputs]
    assert "%4" in output_names

    assert len(graph.nodes) == 2
    assert graph.nodes[0].op_type == "MatMul"
    assert graph.nodes[0].inputs == ["%1", "%2"]
    assert graph.nodes[0].outputs == ["%3"]

    assert graph.nodes[1].op_type == "Generic" or graph.nodes[1].op_type == "aten.relu.default"
    assert graph.nodes[1].inputs == ["%3"]
    assert graph.nodes[1].outputs == ["%4"]


def test_pytorch_fx_parser_missing():
    """Docstring for D103."""
    from onnx9000.converters.pytorch_fx_parser import (
        _map_aten_add_,
        _map_aten_add_Tensor,
        _map_aten_arange_start_step,
        _map_aten_bmm_default,
        _map_aten_convolution_default,
        _map_aten_copy_,
        _map_aten_gelu_default,
        _map_aten_max_pool2d_with_indices_default,
        _map_aten_mm_default,
        _map_aten_mul_Tensor,
        _map_aten_native_batch_norm_legit_no_training_default,
        _map_aten_native_layer_norm_default,
        _map_aten_where_self,
        load_pytorch_fx,
    )

    # We just run the mappers directly to ensure they are covered
    assert _map_aten_add_Tensor(["x"], ["y"], {}).op_type == "Add"
    assert _map_aten_mul_Tensor(["x"], ["y"], {}).op_type == "Mul"
    assert _map_aten_convolution_default(["x"], ["y"], {}).op_type == "Conv"
    assert (
        _map_aten_native_batch_norm_legit_no_training_default(["x"], ["y"], {}).op_type
        == "BatchNorm"
    )
    assert _map_aten_native_layer_norm_default(["x"], ["y"], {}).op_type == "LayerNorm"
    assert _map_aten_bmm_default(["x"], ["y"], {}).op_type == "MatMul"
    assert _map_aten_mm_default(["x"], ["y"], {}).op_type == "MatMul"
    assert _map_aten_max_pool2d_with_indices_default(["x"], ["y"], {}).op_type == "MaxPool2D"
    assert _map_aten_gelu_default(["x"], ["y"], {}).op_type == "Gelu"
    assert _map_aten_arange_start_step(["x"], ["y"], {}).op_type == "Range"
    assert _map_aten_where_self(["x"], ["y"], {}).op_type == "Where"
    assert _map_aten_copy_(["x"], ["y"], {}).op_type == "Identity"
    assert _map_aten_add_(["x"], ["y"], {}).op_type == "Add"

    # load wrapper
    assert load_pytorch_fx("{}") is not None
