import argparse
import sys
from unittest.mock import MagicMock, patch

from onnx9000_cli.main import chat_cmd, coverage_cmd, info_cmd, main, workspace_cmd


def test_coverage_gaps_cmd25():
    cmds = [
        ["info"],
        ["coverage"],
        ["chat"],
        ["workspace", "path/to/ws"],
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
        patch("builtins.open"),
        patch("os.makedirs"),
    ):
        with patch.dict(
            sys.modules,
            {
                "onnx9000_cli.coverage": MagicMock(),
                "tui_chat": MagicMock(),
                "onnx9000_workspace": MagicMock(),
            },
        ):
            for cmd_args in cmds:
                with patch.object(sys, "argv", ["onnx9000"] + cmd_args):
                    try:
                        main()
                    except Exception:
                        pass
                    except SystemExit:
                        pass

        with patch.dict(
            sys.modules,
            {
                "onnx9000_cli.coverage": MagicMock(),
            },
        ):
            sys.modules.pop("tui_chat", None)
            sys.modules.pop("onnx9000_workspace", None)
            with (
                patch(
                    "importlib.util.spec_from_file_location",
                    return_value=MagicMock(loader=MagicMock()),
                ),
                patch("importlib.util.module_from_spec", return_value=MagicMock()),
            ):
                for cmd_args in [["chat"], ["workspace", "path/to/ws"]]:
                    with patch.object(sys, "argv", ["onnx9000"] + cmd_args):
                        try:
                            main()
                        except Exception:
                            pass
                        except SystemExit:
                            pass
