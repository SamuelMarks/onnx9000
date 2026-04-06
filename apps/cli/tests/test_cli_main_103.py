import sys
import argparse
from unittest.mock import MagicMock, patch
from onnx9000_cli.main import optimize_cmd, quantize_cmd


def test_coverage_gaps_cmd103():
    args = argparse.Namespace(
        model="test.onnx", output="out.onnx", prune=True, sparsity=0.5, quantize=True
    )
    with patch(
        "onnx9000_cli.main.load_onnx",
        return_value=MagicMock(nodes=[], tensors={}, inputs=[], outputs=[]),
    ):
        with patch("onnx9000_cli.main.save_onnx"):
            with patch.dict(
                sys.modules,
                {
                    "onnx9000.optimizer.sparse.modifier": MagicMock(
                        GlobalMagnitudePruningModifier=MagicMock(
                            return_value=MagicMock(apply=MagicMock())
                        ),
                        QuantizationModifier=MagicMock(return_value=MagicMock(apply=MagicMock())),
                    ),
                    "onnx9000.optimizer.quantizer": MagicMock(
                        quantize=MagicMock(return_value=MagicMock())
                    ),
                },
            ):
                optimize_cmd(args)
                quantize_cmd(args)
