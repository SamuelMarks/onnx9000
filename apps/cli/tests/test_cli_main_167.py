"""Unit tests and verification suite for cli main 167 functionality."""

import argparse
from unittest.mock import MagicMock, patch

from onnx9000_cli.main import convert_cmd


def test_pytorch_convert_clean() -> None:
    """Verify that pytorch convert clean behaves as expected."""
    args = argparse.Namespace(src="model.pt", to="onnx", output="out.onnx")
    setattr(args, "from", "pytorch")

    class FakeModule:
        """Represents fake module."""

        pass

    mock_torch = MagicMock()
    mock_torch.export.load.side_effect = RuntimeError("no export")
    mock_torch.load.return_value = FakeModule()
    mock_torch.nn.Module = FakeModule
    mock_torch.fx.symbolic_trace.return_value = MagicMock()

    with patch.dict("sys.modules", {"torch": mock_torch}):
        with patch("onnx9000.converters.parsers.PyTorchFXParser"):
            with patch("onnx9000.core.exporter.export_graph"):
                convert_cmd(args)
                assert mock_torch.fx.symbolic_trace.call_count == 1
