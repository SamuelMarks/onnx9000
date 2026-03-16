"""Test Dropout."""

from onnx9000.core.dtypes import DType
from onnx9000.frontend.frontend.builder import GraphBuilder, Tracing
from onnx9000.frontend.frontend.nn.dropout import Dropout, Dropout2d
from onnx9000.frontend.frontend.tensor import Tensor


def test_dropout() -> None:
    """Tests the test_dropout functionality."""
    d1 = Dropout(0.5)
    d2 = Dropout2d(0.2)
    builder = GraphBuilder("dummy")
    with Tracing(builder):
        x = Tensor((20, 16, 50, 50), DType.FLOAT32, "x")
        y1 = d1(x)
        y2 = d2(x)
        d1.eval()
        y3 = d1(x)
    assert y1 is not None
    assert y2 is not None
    assert y3 is not None


def test_dropout_single_res() -> None:
    """Tests the test_dropout_single_res functionality."""
    from unittest.mock import patch
    from onnx9000.frontend.frontend.nn.dropout import Dropout

    d = Dropout()
    with patch("onnx9000.frontend.frontend.utils.record_op", return_value="fake_tensor"):
        assert d(None) == "fake_tensor"
