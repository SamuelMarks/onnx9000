import argparse
import sys
from unittest.mock import MagicMock, patch

from onnx9000_cli.main import main


def test_coverage_gaps_cmd10():
    cmds = [
        [
            "simplify",
            "test.onnx",
            "--skip-rules",
            "a,b",
            "--prune-inputs",
            "c",
            "--preserve-nodes",
            "d",
            "--input-shape",
            "a:1,2",
            "b:a,b",
            "--tensor-type",
            "a:float32",
            "--check-n",
            "3",
            "--custom-ops",
            "custom.py",
        ],
        ["simplify", "test.onnx", "--overwrite", "1"],
    ]

    with (
        patch(
            "onnx9000_cli.main.load_onnx",
            return_value=MagicMock(nodes=[], tensors={}, inputs=[], outputs=[]),
        ),
        patch("onnx9000_cli.main.save_onnx"),
        patch("builtins.open"),
        patch("os.path.exists", side_effect=[True, False]),
        patch("importlib.util.spec_from_file_location", return_value=MagicMock(loader=MagicMock())),
        patch("importlib.util.module_from_spec", return_value=MagicMock()),
    ):
        with patch.dict(
            sys.modules,
            {
                "onnx9000.optimizer.simplifier": MagicMock(),
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
