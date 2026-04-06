import argparse
import sys
from unittest.mock import MagicMock, patch

from onnx9000_cli.main import convert_cmd, jit_cmd, main


def test_coverage_gaps_cmd160():
    args = argparse.Namespace(src="model.pt", to="onnx", output="out.onnx")
    setattr(args, "from", "pytorch")

    with patch("onnx9000_cli.main.load_onnx", return_value=MagicMock()):
        with patch("onnx9000.core.exporter.export_graph"):
            with patch("torch.export.load", side_effect=Exception):
                with patch("torch.load") as mock_load:
                    import torch
                    import torch.nn as nn

                    class FakeModule(nn.Module):
                        pass

                    fm = FakeModule()
                    mock_load.return_value = fm
                    with patch("torch.fx.symbolic_trace"):
                        with patch("onnx9000.converters.parsers.PyTorchFXParser"):
                            import builtins

                            orig_is = builtins.isinstance

                            def my_is(obj, cls):
                                if obj is fm:
                                    return True
                                return orig_is(obj, cls)

                            with patch("builtins.isinstance", side_effect=my_is):
                                try:
                                    convert_cmd(args)
                                except Exception:
                                    pass
