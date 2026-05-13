"""Tests for json_extract."""

import json

from onnx9000.core.ir import Graph, Tensor
from onnx9000.json_extract import _default_serializer, extract_json


def test_default_serializer() -> None:
    """Test the default serializer handling of specific objects."""
    assert _default_serializer(b"hello world") == "[Buffer: 11 bytes]"
    assert _default_serializer(bytearray(b"test")) == "[Buffer: 4 bytes]"
    assert set(_default_serializer({1, 2, 3})) == {1, 2, 3}

    class Dummy:
        def __init__(self):
            self.a = 1
            self._b = 2

    assert _default_serializer(Dummy()) == {"a": 1}
    assert _default_serializer(123) == "123"


def test_extract_json_basic() -> None:
    """Test extracting JSON from a basic graph."""
    graph = Graph("TestGraph")
    json_str = extract_json(graph)
    assert isinstance(json_str, str)

    data = json.loads(json_str)
    assert data["name"] == "TestGraph"
    assert "nodes" in data


def test_extract_json_with_tensor() -> None:
    """Test extracting JSON from a graph with buffers."""
    graph = Graph("TensorGraph")
    t = Tensor("weights", [2, 2], "float32")
    t.data = b"0000111122223333"  # 16 bytes
    graph.initializers.append(t)

    json_str = extract_json(graph)
    data = json.loads(json_str)

    init = data["initializers"][0]
    assert init["name"] == "weights"
    assert init["data"] == "[Buffer: 16 bytes]"
