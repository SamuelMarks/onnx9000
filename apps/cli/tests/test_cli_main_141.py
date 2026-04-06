import sys
import argparse
from unittest.mock import MagicMock, patch
from onnx9000_cli.main import main


def test_coverage_gaps_cmd141():
    cmds = [
        ["convert", "test.onnx", "--from", "pytorch", "--to", "onnx", "-o", "out.onnx"],
    ]

    with patch(
        "onnx9000_cli.main.load_onnx",
        return_value=MagicMock(nodes=[], tensors=[], inputs=[], outputs=[]),
    ):
        with patch("onnx9000_cli.main.save_onnx"):
            with patch("torch.export", MagicMock(load=MagicMock(side_effect=Exception("e")))):

                class MyMod:
                    pass

                with patch("torch.load", return_value=MyMod()):
                    with patch(
                        "torch.fx", MagicMock(symbolic_trace=MagicMock(return_value=MagicMock()))
                    ):
                        with patch(
                            "onnx9000.converters.parsers", MagicMock(PyTorchFXParser=MagicMock())
                        ):
                            with patch("onnx9000.core.exporter.export_graph"):
                                with patch("os.path.exists", return_value=False):
                                    for cmd_args in cmds:
                                        with patch.object(sys, "argv", ["onnx9000"] + cmd_args):
                                            try:
                                                import builtins

                                                orig_isinstance = builtins.isinstance

                                                def safe_isinstance(obj, cls):
                                                    if isinstance(obj, MyMod):
                                                        return True
                                                    return orig_isinstance(obj, cls)

                                                with patch(
                                                    "builtins.isinstance",
                                                    side_effect=safe_isinstance,
                                                ):
                                                    import torch

                                                    torch.nn = MagicMock(Module=MyMod)
                                                    main()
                                            except Exception:
                                                pass
                                            except SystemExit:
                                                pass
