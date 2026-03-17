"""Module providing core logic and structural definitions."""


def test_jax_importer_add_mul_i32() -> None:
    """Tests the test_jax_importer_add_mul_i32 functionality."""
    from onnx9000.converters.jax.importer import load_jax

    jaxpr_dict = {
        "invars": [{"name": "x", "shape": [1], "type": "i32"}],
        "outvars": [{"name": "z", "shape": [1], "type": "i32"}],
        "constvars": [],
        "eqns": [
            {
                "primitive": "add",
                "invars": [{"name": "x"}],
                "outvars": [{"name": "y", "shape": [1], "type": "i32"}],
            },
            {
                "primitive": "mul",
                "invars": [{"name": "y"}],
                "outvars": [{"name": "z", "shape": [1], "type": "i32"}],
            },
        ],
    }
    graph = load_jax(jaxpr_dict)
    assert len(graph.nodes) == 2
    assert graph.nodes[0].op_type == "Add"
    assert graph.nodes[1].op_type == "Mul"


def test_jax_importer_unknown_type() -> None:
    """Tests the test_jax_importer_unknown_type functionality."""
    from onnx9000.converters.jax.importer import load_jax

    jaxpr_dict = {"invars": [{"name": "x", "shape": [1], "type": "unknown_type"}]}
    graph = load_jax(jaxpr_dict)
    assert graph.tensors["x"].dtype == 1


def test_jax_importer_fallback_load() -> None:
    """Tests the test_jax_importer_fallback_load functionality."""
    from unittest.mock import patch

    from onnx9000.converters.jax.importer import load

    with patch("onnx9000.core.parser.core.load") as mock_load:
        mock_load.return_value = "onnx_graph"
        res = load("test.onnx")
        assert res == "onnx_graph"
