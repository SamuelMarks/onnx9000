import argparse
from unittest.mock import MagicMock, patch

from onnx9000_cli.main import onnx2tf_cmd


def test_onnx2tf_cmd():
    args = argparse.Namespace(
        input="dummy.onnx",
        output="dummy.tflite",
        keep_nchw=True,
        int8=True,
        fp16=True,
        batch=1,
        disable_optimization=True,
        external_weights="weights.bin",
        progress=True,
        micro=False,
    )
    with patch("onnx9000.tflite_exporter.cli.main") as mock_main:
        onnx2tf_cmd(args)

        mock_main.assert_called_once()
        called_args = mock_main.call_args[0][0]
        assert "dummy.onnx" in called_args
        assert "-o" in called_args
        assert "dummy.tflite" in called_args
        assert "--int8" in called_args
