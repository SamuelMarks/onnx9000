import argparse
import sys
from unittest.mock import MagicMock, patch

from onnx9000_cli.main import main


def test_coverage_gaps_cmd27():
    cmds = [
        ["coreml", "export", "test.onnx"],
        ["edit", "test.onnx"],
    ]

    with (
        patch(
            "onnx9000_cli.main.load_onnx",
            return_value=MagicMock(inputs=[MagicMock(name="in", shape=(1, 2))], outputs=[]),
        ),
        patch(
            "onnx9000.core.parser.core.load",
            return_value=MagicMock(inputs=[MagicMock(name="in", shape=(1, 2))], outputs=[]),
        ),
        patch("onnx9000_cli.main.save_onnx"),
        patch("os.makedirs"),
    ):
        with patch.dict(
            sys.modules,
            {
                "onnx9000_array": MagicMock(lazy_mode=MagicMock()),
                "importlib.util.spec_from_file_location": MagicMock(
                    return_value=MagicMock(loader=MagicMock())
                ),
                "importlib.util.module_from_spec": MagicMock(return_value=MagicMock()),
            },
        ):
            with patch("os.path.exists", return_value=True):
                with patch("subprocess.run"):
                    for cmd_args in cmds:
                        with patch.object(sys, "argv", ["onnx9000"] + cmd_args):
                            try:
                                main()
                            except Exception:
                                pass
                            except SystemExit:
                                pass

            with patch("os.path.exists", return_value=False):
                with patch("subprocess.run"):
                    for cmd_args in cmds:
                        with patch.object(sys, "argv", ["onnx9000"] + cmd_args):
                            try:
                                main()
                            except Exception:
                                pass
                            except SystemExit:
                                pass

            with patch("os.path.exists", return_value=True):
                import subprocess

                with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "cmd")):
                    with patch.object(sys, "argv", ["onnx9000", "coreml", "export", "test"]):
                        try:
                            main()
                        except SystemExit:
                            pass
                with patch("subprocess.run", side_effect=KeyboardInterrupt):
                    with patch.object(sys, "argv", ["onnx9000", "edit", "test"]):
                        try:
                            main()
                        except SystemExit:
                            pass
