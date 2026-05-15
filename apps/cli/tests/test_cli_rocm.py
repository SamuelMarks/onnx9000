import argparse
from unittest.mock import patch

from onnx9000_cli.main import rocm_cmd


def test_rocm_cmd():
    args = argparse.Namespace(model="dummy.onnx")
    with patch("builtins.print") as mock_print:
        with patch("onnx9000.backends.rocm.executor.Dispatcher") as mock_dispatcher:
            with patch("onnx9000_cli.main.load_onnx") as mock_load:
                rocm_cmd(args)

                mock_print.assert_any_call("Initializing ROCm execution for dummy.onnx")
                mock_load.assert_called_once_with("dummy.onnx")
                mock_dispatcher.assert_called_once()
                mock_print.assert_any_call("ROCm engine loaded.")
