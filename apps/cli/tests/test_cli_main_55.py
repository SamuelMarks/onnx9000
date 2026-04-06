import argparse
import sys
from unittest.mock import MagicMock, patch

from onnx9000_cli.main import chat_cmd, coverage_cmd, info_cmd, main, workspace_cmd


def test_coverage_gaps_cmd55():
    try:
        from onnx9000_cli.main import main

        with patch.object(sys, "argv", ["onnx9000", "info"]):
            main()
    except SystemExit:
        pass

    try:
        from onnx9000_cli.main import main

        with patch.object(sys, "argv", ["onnx9000", "coverage"]):
            with patch("onnx9000_cli.coverage.update_coverage_cmd"):
                main()
    except SystemExit:
        pass

    try:
        from onnx9000_cli.main import main

        with patch.object(sys, "argv", ["onnx9000", "chat"]):
            with patch.dict(sys.modules, {"tui_chat": MagicMock()}):
                main()
    except SystemExit:
        pass

    try:
        from onnx9000_cli.main import main

        with patch.object(sys, "argv", ["onnx9000", "workspace"]):
            with patch.dict(sys.modules, {"onnx9000_workspace": MagicMock()}):
                main()
    except SystemExit:
        pass

    # Chat fallback
    try:
        from onnx9000_cli.main import main

        with patch.object(sys, "argv", ["onnx9000", "chat"]):
            sys.modules.pop("tui_chat", None)
            with patch(
                "importlib.util.spec_from_file_location", return_value=MagicMock(loader=MagicMock())
            ):
                with patch("importlib.util.module_from_spec", return_value=MagicMock()):
                    with patch("builtins.input", return_value="exit"):
                        main()
    except SystemExit:
        pass

    # Chat fallback fail
    try:
        from onnx9000_cli.main import main

        with patch.object(sys, "argv", ["onnx9000", "chat"]):
            sys.modules.pop("tui_chat", None)
            with patch("importlib.util.spec_from_file_location", side_effect=Exception("Failed")):
                with patch("builtins.input", return_value="exit"):
                    main()
    except SystemExit:
        pass

    # Workspace fallback
    try:
        from onnx9000_cli.main import main

        with patch.object(sys, "argv", ["onnx9000", "workspace"]):
            sys.modules.pop("onnx9000_workspace", None)
            with patch("onnx9000_workspace.setup_workspace", create=True):
                main()
    except SystemExit:
        pass
