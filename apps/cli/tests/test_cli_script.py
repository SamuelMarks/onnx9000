import argparse
from unittest.mock import MagicMock, patch

from onnx9000_cli.main import script_cmd


def test_script_cmd_output():
    args = argparse.Namespace(input="dummy.py", output="out.onnx")
    with patch("onnx9000.toolkit.script.parse_and_compile") as mock_parse:
        mock_model = MagicMock()
        mock_parse.return_value = mock_model

        script_cmd(args)

        mock_parse.assert_called_once_with("dummy.py")
        mock_model.save.assert_called_once_with("out.onnx")


def test_script_cmd_no_output():
    args = argparse.Namespace(input="dummy.py")
    with patch("onnx9000.toolkit.script.parse_and_compile") as mock_parse:
        mock_model = MagicMock()
        mock_parse.return_value = mock_model

        script_cmd(args)

        mock_parse.assert_called_once_with("dummy.py")
        mock_model.save.assert_not_called()


def test_script_cmd_error():
    args = argparse.Namespace(input="dummy.py")
    with patch("onnx9000.toolkit.script.parse_and_compile") as mock_parse:
        mock_parse.side_effect = Exception("test error")
        with patch("sys.exit") as mock_exit:
            script_cmd(args)
            mock_exit.assert_called_once_with(1)
