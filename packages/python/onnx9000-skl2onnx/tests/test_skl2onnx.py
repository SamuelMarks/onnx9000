"""Comprehensive unit tests for onnx9000.skl2onnx and onnx9000_skl2onnx."""

import onnx9000_skl2onnx
import pytest
from onnx9000.core.ir import Graph
from onnx9000.skl2onnx import SKLearnParser, convert, convert_sklearn


class _DummyEstimator:
    """Minimal dummy Scikit-Learn estimator for testing."""


def test_skl2onnx_convert_and_parser() -> None:
    """Verify convert(), convert_sklearn(), and SKLearnParser."""
    with pytest.raises(ValueError, match="Invalid model string"):
        convert("")
    assert convert("test") == "[ONNX-IR] from skl test"
    assert onnx9000_skl2onnx.convert("rf") == "[ONNX-IR] from skl rf"

    est = _DummyEstimator()
    g = convert_sklearn(est, name="skl_graph")  # type: ignore[arg-type]
    assert isinstance(g, Graph)
    assert g.name == "skl_graph"
    assert SKLearnParser is not None
