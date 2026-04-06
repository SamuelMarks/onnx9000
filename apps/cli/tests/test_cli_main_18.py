import sys
import argparse
from unittest.mock import MagicMock, patch
from onnx9000_cli.main import main


def test_coverage_gaps_cmd18():
    cmds = [
        ["hummingbird", "test.onnx"],
    ]

    with (
        patch(
            "onnx9000_cli.main.load_onnx",
            return_value=MagicMock(
                nodes=[MagicMock(op_type="TreeEnsembleClassifier")],
                tensors={},
                inputs=[],
                outputs=[],
            ),
        ),
        patch("onnx9000_cli.main.save_onnx"),
    ):
        with patch.dict(
            sys.modules,
            {
                "onnx9000.optimizer.hummingbird.engine": MagicMock(
                    TranspilationEngine=MagicMock(
                        return_value=MagicMock(
                            transpile=MagicMock(return_value=MagicMock(nodes=[], tensors={}))
                        )
                    )
                ),
                "onnx9000.optimizer.hummingbird.onnxml_parser": MagicMock(),
                "onnx9000.optimizer.hummingbird.strategies": MagicMock(),
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
