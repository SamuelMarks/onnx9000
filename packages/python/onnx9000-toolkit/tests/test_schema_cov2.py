"""Tests the schema cov2 module functionality."""

from onnx9000.toolkit.script.schema import SchemaRegistry


def test_schema_load_from_json() -> None:
    """Tests the schema load from json functionality."""
    reg = SchemaRegistry()
    reg.load_from_json(
        '[{"name": "CustomOp", "since_version": 1, "inputs": ["X"], "outputs": ["Y"]}]'
    )
    assert "CustomOp" in reg.schemas


def test_schema_coverage_misc():
    """Docstring for D103."""
    import json

    from onnx9000.toolkit.script.schema import OpSchema, SchemaRegistry

    # 49 `if not candidates:`
    reg = SchemaRegistry()
    reg.register(OpSchema("FutureOp", 99, [], [], []))
    assert reg.get_schema("FutureOp", 10) is None

    # 54-56 `item.get("outputs", []), item.get("attributes", [])` inside `load_from_json`
    reg.load_from_json(json.dumps([{"name": "NoOutsAttrOp", "since_version": 1, "inputs": []}]))
    schema = reg.get_schema("NoOutsAttrOp", 10)
    assert schema is not None
    assert schema.outputs == []
    assert schema.attributes == []
