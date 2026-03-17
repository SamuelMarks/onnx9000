"""Tests the schema cov2 module functionality."""

from onnx9000.toolkit.script.schema import SchemaRegistry


def test_schema_load_from_json() -> None:
    """Tests the schema load from json functionality."""
    reg = SchemaRegistry()
    reg.load_from_json(
        '[{"name": "CustomOp", "since_version": 1, "inputs": ["X"], "outputs": ["Y"]}]'
    )
    assert "CustomOp" in reg.schemas
