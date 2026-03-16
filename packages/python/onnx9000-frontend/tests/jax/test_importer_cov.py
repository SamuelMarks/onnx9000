"""Module providing core logic and structural definitions."""


def test_jax_importer_add_mul_i32() -> None:
    """Tests the test_jax_importer_add_mul_i32 functionality."""
    from onnx9000.core.dtypes import DType
    from onnx9000.frontend.jax.importer import JaxprImporter, _map_jax_type

    assert _map_jax_type("i32") == DType.INT32
    assert _map_jax_type("unknown") == DType.FLOAT32
    jaxpr = {
        "invars": [{"name": "a", "shape": [1], "type": "i32"}],
        "outvars": [{"name": "c"}],
        "eqns": [
            {
                "primitive": "add",
                "invars": [{"name": "a"}],
                "outvars": [{"name": "b", "shape": [1], "type": "i32"}],
            },
            {
                "primitive": "mul",
                "invars": [{"name": "b"}],
                "outvars": [{"name": "c", "shape": [1], "type": "i32"}],
            },
        ],
    }
    importer = JaxprImporter()
    g = importer.parse(jaxpr)
    ops = [n.op_type for n in g.nodes]
    assert "Add" in ops
    assert "Mul" in ops


def test_jax_importer_extra() -> None:
    """Tests the test_jax_importer_extra functionality."""
    from onnx9000.frontend.jax.importer import JaxprImporter

    jaxpr = {
        "invars": [],
        "outvars": [{"name": "o"}],
        "eqns": [
            {
                "primitive": "sub",
                "invars": [],
                "outvars": [{"name": "o", "shape": [1], "type": "f32"}],
            },
            {
                "primitive": "div",
                "invars": [],
                "outvars": [{"name": "o", "shape": [1], "type": "f32"}],
            },
            {
                "primitive": "max",
                "invars": [],
                "outvars": [{"name": "o", "shape": [1], "type": "f32"}],
            },
            {
                "primitive": "min",
                "invars": [],
                "outvars": [{"name": "o", "shape": [1], "type": "f32"}],
            },
            {
                "primitive": "dot_general",
                "invars": [],
                "outvars": [{"name": "o", "shape": [1], "type": "f32"}],
            },
            {
                "primitive": "broadcast_in_dim",
                "invars": [],
                "outvars": [{"name": "o", "shape": [1], "type": "f32"}],
                "params": {"broadcast_dimensions": []},
            },
            {
                "primitive": "xla_pmap",
                "invars": [],
                "outvars": [{"name": "o", "shape": [1], "type": "f32"}],
                "params": {"axis_name": "x"},
            },
            {
                "primitive": "grad_core",
                "invars": [],
                "outvars": [{"name": "o", "shape": [1], "type": "f32"}],
            },
        ],
    }
    importer = JaxprImporter()
    importer.parse(jaxpr)


def test_jax_importer_constvars_and_load() -> None:
    """Tests the test_jax_importer_constvars_and_load functionality."""
    from onnx9000.frontend.jax.importer import JaxprImporter, load

    jaxpr = {
        "invars": [],
        "constvars": [{"name": "c", "shape": [1], "type": "f32"}],
        "outvars": [],
        "eqns": [],
    }
    importer = JaxprImporter()
    importer.parse(jaxpr)
    assert load(jaxpr, format="jax") is not None
    assert load({"node": []}, format="tf") is None
    from unittest.mock import patch

    with patch("onnx9000.core.parser.core.load", return_value="onnx_loaded"):
        assert load("model.onnx", format="onnx") == "onnx_loaded"
