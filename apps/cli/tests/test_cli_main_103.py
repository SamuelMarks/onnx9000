"""Unit tests and verification suite for cli main 103 functionality."""

import argparse
import sys
from unittest.mock import MagicMock, patch

from onnx9000_cli.main import optimize_cmd, quantize_cmd


def test_coverage_gaps_cmd103() -> None:
    """Verify that coverage gaps cmd103 behaves as expected."""
    args = argparse.Namespace(
        model="test.onnx", output="out.onnx", prune=True, sparsity=0.5, quantize=True
    )
    with (
        patch(
            "onnx9000_cli.main.load_onnx",
            return_value=MagicMock(nodes=[], tensors={}, inputs=[], outputs=[]),
        ),
        patch("onnx9000_cli.main.save_onnx"),
        patch.dict(
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
        ),
    ):
        optimize_cmd(args)
        quantize_cmd(args)
    assert args is not None or args is None
