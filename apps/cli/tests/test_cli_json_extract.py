import argparse
import json
from unittest.mock import MagicMock, mock_open, patch

from onnx9000_cli.main import json_extract_cmd


def test_json_extract_cmd_stdout():
    args = argparse.Namespace(model="dummy.onnx", output=None)
    mock_graph = MagicMock()
    mock_graph.__dict__ = {"test_attr": "value"}

    with (
        patch("onnx9000_cli.main.load_onnx", return_value=mock_graph),
        patch("onnx9000.core.parser.core.load", return_value=mock_graph),
        patch("builtins.print") as mock_print,
    ):
        json_extract_cmd(args)

    mock_print.assert_any_call('{\n  "test_attr": "value"\n}')


def test_json_extract_cmd_file():
    args = argparse.Namespace(model="dummy.onnx", output="out.json")
    mock_graph = MagicMock()
    mock_graph.__dict__ = {"test_attr": b"buffer_data"}

    m_open = mock_open()
    with (
        patch("onnx9000_cli.main.load_onnx", return_value=mock_graph),
        patch("onnx9000.core.parser.core.load", return_value=mock_graph),
        patch("builtins.open", m_open),
        patch("builtins.print"),
    ):
        json_extract_cmd(args)

    m_open.assert_called_once_with("out.json", "w")
    # check that we wrote [Buffer: 11 bytes] for the bytes object
    m_open().write.assert_called_once_with('{\n  "test_attr": "[Buffer: 11 bytes]"\n}')


def test_json_extract_cmd_set():
    args = argparse.Namespace(model="dummy.onnx", output=None)

    class DummyGraph:
        def __init__(self):
            self.my_set = {1, 2}
            self._hidden = True

    mock_graph = DummyGraph()

    with (
        patch("onnx9000_cli.main.load_onnx", return_value=mock_graph),
        patch("onnx9000.core.parser.core.load", return_value=mock_graph),
        patch("builtins.print") as mock_print,
    ):
        json_extract_cmd(args)

    # Set gets converted to list
    calls = mock_print.call_args_list
    out = calls[-1][0][0]

    # parse the json to check correctly regardless of order
    out_dict = json.loads(out)
    assert "my_set" in out_dict
    assert isinstance(out_dict["my_set"], list)
    assert set(out_dict["my_set"]) == {1, 2}
    assert "_hidden" not in out_dict


def test_json_extract_cmd_str_fallback():
    args = argparse.Namespace(model="dummy.onnx", output=None)
    mock_graph = MagicMock()

    # give it something that has no __dict__, isn't bytes/set to hit str() fallback
    # like a simple int or a custom class with __slots__
    class Slotted:
        __slots__ = ["val"]

        def __init__(self):
            self.val = 42

        def __str__(self):
            return "SlottedObj"

    mock_graph.__dict__ = {"test_attr": Slotted()}

    with (
        patch("onnx9000_cli.main.load_onnx", return_value=mock_graph),
        patch("onnx9000.core.parser.core.load", return_value=mock_graph),
        patch("builtins.print") as mock_print,
    ):
        json_extract_cmd(args)

    mock_print.assert_any_call('{\n  "test_attr": "SlottedObj"\n}')
