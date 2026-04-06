import sys
import argparse
from unittest.mock import MagicMock, patch
from onnx9000_cli.main import hummingbird_cmd


def test_coverage_gaps_cmd102():
    args = argparse.Namespace(model="test.onnx", output="out.onnx")
    with patch(
        "onnx9000_cli.main.load_onnx",
        return_value=MagicMock(
            nodes=[MagicMock(op_type="TreeEnsembleClassifier")], tensors={}, inputs=[], outputs=[]
        ),
    ):
        with patch("onnx9000_cli.main.save_onnx"):
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
                    "onnx9000.optimizer.hummingbird.onnxml_parser": MagicMock(
                        parse_onnxml_tree_ensemble=MagicMock()
                    ),
                    "onnx9000.optimizer.hummingbird.strategies": MagicMock(
                        TargetHardware=MagicMock(CPU=1)
                    ),
                },
            ):
                hummingbird_cmd(args)
