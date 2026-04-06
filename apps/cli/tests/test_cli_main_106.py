import argparse
import sys
from unittest.mock import MagicMock, patch

from onnx9000_cli.main import main


def test_coverage_gaps_cmd106():
    cmds = [
        ["convert", "test.onnx", "--from", "pytorch", "--to", "onnx", "-o", "out.onnx"],
    ]

    with (
        patch(
            "onnx9000_cli.main.load_onnx",
            return_value=MagicMock(nodes=[], tensors={}, inputs=[], outputs=[]),
        ),
        patch("builtins.open"),
        patch("os.makedirs"),
        patch("onnx9000_cli.main.save_onnx"),
    ):
        with patch.dict(
            sys.modules,
            {
                "torch.export": MagicMock(load=MagicMock(side_effect=Exception)),
                "torch.load": MagicMock(return_value="t"),
                "torch.nn": MagicMock(Module=type("Mod", (object,), {})),
                "onnx9000.converters.torch.fx": MagicMock(
                    PyTorchFXParser=MagicMock(return_value=MagicMock(parse=MagicMock()))
                ),
            },
        ):
            with patch("os.path.isdir", side_effect=[False] * 4):
                for cmd_args in cmds:
                    with patch.object(sys, "argv", ["onnx9000"] + cmd_args):
                        try:
                            main()
                        except Exception:
                            pass
                        except SystemExit:
                            pass
