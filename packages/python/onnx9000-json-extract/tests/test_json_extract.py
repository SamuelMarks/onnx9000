"""Comprehensive unit tests for `onnx9000.json_extract` and IR `to_dict()` methods."""

import json

from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Attribute, DynamicDim, Graph, Node, Tensor, ValueInfo
from onnx9000.json_extract import DictSerializableProtocol, _default_serializer, extract_json


def test_json_extract_and_ir_to_dict() -> None:
    """Verify _default_serializer and extract_json on Graph, Node, Tensor, ValueInfo, Attribute, DynamicDim."""
    assert DictSerializableProtocol.to_dict(None) == {}

    class CustomDictObj:
        """Represents custom dict obj, supporting operations such as to_dict."""

        def to_dict(self) -> dict[str, int]:
            """Execute to dict on CustomDictObj.

            Returns:
                Result of to dict (dict[str, int]).
            """
            return {"a": 1}

    assert _default_serializer(b"abc") == "[Buffer: 3 bytes]"
    assert _default_serializer(CustomDictObj()) == {"a": 1}
    assert _default_serializer({1}) == [1]
    assert _default_serializer(1.23) == "1.23"

    dim = DynamicDim("batch")
    assert dim.to_dict() == {"value": "batch"}

    attr_int = Attribute("axis", "INT", 1)
    assert attr_int.to_dict() == {"name": "axis", "attr_type": "INT", "value": 1}

    attr_complex = Attribute("sub", "INTS", [1, 2])
    assert attr_complex.to_dict()["value"] == "[1, 2]"

    vi = ValueInfo("in_0", (dim, 3), DType.FLOAT32)
    assert vi.to_dict()["shape"] == ["batch", 3]

    t = Tensor(
        "w", shape=(2, 2), dtype=DType.FLOAT32, is_initializer=True, data=b"\x00\x01\x02\x03"
    )
    assert t.to_dict()["data"] == "[Buffer: 4 bytes]"

    n = Node(
        "Add", inputs=[t, "in_0"], outputs=["out_0"], attributes={"axis": attr_int}, name="add_0"
    )
    assert n.to_dict()["inputs"] == ["w", "in_0"]

    g = Graph("test_json_graph")
    g.inputs.append(vi)
    g.outputs.append(ValueInfo("out_0", (dim, 3), DType.FLOAT32))
    g.add_tensor(t)
    g.add_node(n)

    json_str = extract_json(g, indent=2)
    parsed = json.loads(json_str)
    assert parsed["name"] == "test_json_graph"
    assert parsed["nodes"][0]["op_type"] == "Add"
    assert parsed["tensors"]["w"]["data"] == "[Buffer: 4 bytes]"
