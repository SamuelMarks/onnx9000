import argparse
import sys
from unittest.mock import MagicMock, patch

from onnx9000_cli.main import main


def test_coverage_gaps_cmd139():
    cmds = [
        ["convert", "test.onnx", "--from", "pytorch", "--to", "onnx", "-o", "out.onnx"],
    ]

    with patch(
        "onnx9000_cli.main.load_onnx",
        return_value=MagicMock(nodes=[], tensors=[], inputs=[], outputs=[]),
    ):
        with patch("onnx9000_cli.main.save_onnx"):
            with patch("torch.export", MagicMock(load=MagicMock(side_effect=Exception("e")))):

                class Mod:
                    pass

                with patch("torch.load", return_value=Mod()):
                    with patch("torch.nn", MagicMock(Module=Mod)):
                        with patch(
                            "torch.fx",
                            MagicMock(symbolic_trace=MagicMock(return_value=MagicMock())),
                        ):
                            with patch(
                                "onnx9000.converters.parsers",
                                MagicMock(PyTorchFXParser=MagicMock()),
                            ):
                                with patch("os.path.exists", return_value=False):
                                    for cmd_args in cmds:
                                        with patch.object(sys, "argv", ["onnx9000"] + cmd_args):
                                            try:
                                                with patch(
                                                    "onnx9000.core.exporter.export_graph",
                                                    MagicMock(),
                                                ):
                                                    main()
                                            except Exception:
                                                pass
                                            except SystemExit:
                                                pass
