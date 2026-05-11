"""Tests for CNTK mapper."""

from onnx9000.converters.cntk.mapper import CNTKMapper


def test_mapper_cntk_basic():
    """Test mapping CNTK model dict to Graph."""
    model_dict = {
        "inputs": [{"name": "input_x"}],
        "nodes": [
            {"op": "Convolution", "name": "conv1", "inputs": ["input_x"], "outputs": ["conv1_out"]},
            {
                "op": "Plus",
                "name": "plus1",
                "inputs": ["conv1_out", "bias"],
                "outputs": ["plus1_out"],
            },
        ],
        "outputs": [{"name": "plus1_out"}],
    }
    mapper = CNTKMapper(model_dict)
    graph = mapper.map()

    assert len(graph.inputs) == 1
    assert graph.inputs[0].name == "input_x"

    assert len(graph.nodes) == 2
    assert graph.nodes[0].op_type == "Conv"
    assert graph.nodes[1].op_type == "Add"

    assert len(graph.outputs) == 1
    assert graph.outputs[0].name == "plus1_out"
