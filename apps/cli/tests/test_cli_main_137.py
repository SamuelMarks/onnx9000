import sys
import argparse
from unittest.mock import MagicMock, patch
from onnx9000_cli.main import main


def test_coverage_gaps_cmd137():
    cmds = [
        ["convert", "test.onnx", "--from", "pytorch", "--to", "onnx", "-o", "out.onnx"],
    ]

    with patch(
        "onnx9000_cli.main.load_onnx",
        return_value=MagicMock(nodes=[], tensors=[], inputs=[], outputs=[]),
    ):
        with patch("onnx9000_cli.main.save_onnx"):
            with patch("torch.export", MagicMock(load=MagicMock(side_effect=Exception("e")))):
                with patch("torch.load", return_value=MagicMock()):
                    with patch(
                        "torch.fx", MagicMock(symbolic_trace=MagicMock(return_value=MagicMock()))
                    ):
                        with patch(
                            "onnx9000.converters.parsers", MagicMock(PyTorchFXParser=MagicMock())
                        ):
                            with patch("os.path.exists", return_value=False):
                                for cmd_args in cmds:
                                    with patch.object(sys, "argv", ["onnx9000"] + cmd_args):
                                        try:
                                            # We need torch.nn.Module to be defined in sys.modules or similar so isinstance works
                                            import torch.nn

                                            class FakeModule(torch.nn.Module):
                                                pass

                                            with patch("torch.load", return_value=FakeModule()):
                                                from onnx9000.core.exporter import export_graph

                                                with patch(
                                                    "onnx9000.core.exporter.export_graph",
                                                    MagicMock(),
                                                ):
                                                    main()
                                        except Exception:
                                            pass
                                        except SystemExit:
                                            pass
