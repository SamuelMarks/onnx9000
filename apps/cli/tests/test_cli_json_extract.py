"""Tests for the json-extract CLI subcommand."""

from __future__ import annotations

import argparse
import json
from unittest.mock import mock_open, patch

from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Attribute, DynamicDim, Graph, Node, Tensor, ValueInfo
from onnx9000_cli.main import json_extract_cmd


def test_json_extract_cmd_stdout() -> None:
    """Verify json_extract_cmd serializes a Graph via to_dict() to stdout."""
    args = argparse.Namespace(model="dummy.onnx", output=None)
    graph = Graph("test_graph")
    graph.inputs.append(ValueInfo("x", (DynamicDim("N"), 3), DType.FLOAT32))
    graph.nodes.append(
        Node("Relu", ["x"], ["y"], {"alpha": Attribute("alpha", "FLOAT", 0.1)}, name="relu_0")
    )

    with (
        patch("onnx9000_cli.main.load_onnx", return_value=graph),
        patch("builtins.print") as mock_print,
    ):
        json_extract_cmd(args)

    calls = mock_print.call_args_list
    out_str = calls[-1][0][0]
    parsed = json.loads(out_str)
    assert parsed["name"] == "test_graph"
    assert parsed["nodes"][0]["op_type"] == "Relu"


def test_json_extract_cmd_file() -> None:
    """Verify json_extract_cmd writes serialized JSON including Tensor raw bytes to a file."""
    args = argparse.Namespace(model="dummy.onnx", output="out.json")
    graph = Graph("test_graph")
    t = Tensor("w", (2,), DType.FLOAT32, data=b"buffer_data")
    graph.add_tensor(t)

    m_open = mock_open()
    with (
        patch("onnx9000_cli.main.load_onnx", return_value=graph),
        patch("builtins.open", m_open),
        patch("builtins.print"),
    ):
        json_extract_cmd(args)

    m_open.assert_called_once_with("out.json", "w")
    written = m_open().write.call_args[0][0]
    parsed = json.loads(written)
    assert parsed["tensors"]["w"]["data"] == "[Buffer: 11 bytes]"


def test_json_extract_cmd_set_and_fallback() -> None:
    """Verify default_serializer handles sets, raw bytes, DynamicDim, and string fallback."""
    args = argparse.Namespace(model="dummy.onnx", output=None)

    class CustomObj:
        """Custom non-JSON object for fallback serialization."""

        def __str__(self) -> str:
            """Return string representation."""
            return "CustomStr"

    payload = {
        "my_set": {"a"},
        "raw": b"1234",
        "dim": DynamicDim("B"),
        "attr": Attribute("k", "INT", 5),
        "custom": CustomObj(),
    }

    with (
        patch("onnx9000_cli.main.load_onnx", return_value=payload),
        patch("builtins.print") as mock_print,
    ):
        json_extract_cmd(args)

    out_str = mock_print.call_args_list[-1][0][0]
    parsed = json.loads(out_str)
    assert parsed["my_set"] == ["a"]
    assert parsed["raw"] == "[Buffer: 4 bytes]"
    assert parsed["dim"] == {"value": "B"}
    assert parsed["attr"]["value"] == 5
    assert parsed["custom"] == "CustomStr"
