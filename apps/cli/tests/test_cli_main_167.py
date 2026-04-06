import argparse
import sys
from unittest.mock import MagicMock, patch
from onnx9000_cli.main import convert_cmd


def test_pytorch_convert_clean():
    args = argparse.Namespace(src="model.pt", to="onnx", output="out.onnx")
    setattr(args, "from", "pytorch")

    class FakeModule:
        pass

    mock_torch = MagicMock()
    mock_torch.export.load.side_effect = Exception("no export")
    mock_torch.load.return_value = FakeModule()
    mock_torch.nn.Module = FakeModule
    mock_torch.fx.symbolic_trace.return_value = MagicMock()

    with patch.dict("sys.modules", {"torch": mock_torch}):
        with patch("onnx9000.converters.parsers.PyTorchFXParser"):
            with patch("onnx9000.core.exporter.export_graph"):
                convert_cmd(args)
                assert mock_torch.fx.symbolic_trace.call_count == 1
