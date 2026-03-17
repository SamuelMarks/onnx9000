"""Module providing core logic and structural definitions."""

from onnx9000.converters.frontend.nn.conv import ConvTranspose1d, ConvTranspose2d

"Test Conv."
from onnx9000.core.dtypes import DType
from onnx9000.converters.frontend.builder import GraphBuilder, Tracing
from onnx9000.converters.frontend.nn.conv import Conv1d, Conv2d, Conv3d
from onnx9000.converters.frontend.tensor import Tensor


def test_conv() -> None:
    """Tests the test_conv functionality."""
    c1 = Conv1d(16, 33, 3, bias=False)
    c2 = Conv2d(16, 33, 3, padding=1)
    c3 = Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(1, 2, 0))
    assert c1.weight.shape == (33, 16, 3)
    assert c2.weight.shape == (33, 16, 3, 3)
    assert c3.weight.shape == (33, 16, 3, 5, 2)
    assert c1.bias is None
    assert c2.bias.shape == (33,)
    builder = GraphBuilder("dummy")
    with Tracing(builder):
        x1 = Tensor((20, 16, 50), DType.FLOAT32, "x1")
        x2 = Tensor((20, 16, 50, 50), DType.FLOAT32, "x2")
        x3 = Tensor((20, 16, 10, 50, 50), DType.FLOAT32, "x3")
        y1 = c1(x1)
        y2 = c2(x2)
        y3 = c3(x3)
    assert y1 is not None
    assert y2 is not None
    assert y3 is not None


def test_conv_transpose() -> None:
    """Tests the test_conv_transpose functionality."""
    ct1 = ConvTranspose1d(16, 33, 3, stride=2, output_padding=1, bias=False)
    ct2 = ConvTranspose2d(16, 33, 3, padding=1)
    assert ct1.weight.shape == (16, 33, 3)
    assert ct2.weight.shape == (16, 33, 3, 3)
    assert ct1.bias is None
    assert ct2.bias.shape == (33,)
    from onnx9000.converters.frontend.builder import GraphBuilder, Tracing

    builder = GraphBuilder("dummy")
    with Tracing(builder):
        x1 = Tensor((20, 16, 50), DType.FLOAT32, "x1")
        x2 = Tensor((20, 16, 50, 50), DType.FLOAT32, "x2")
        y1 = ct1(x1)
        y2 = ct2(x2)
    assert y1 is not None
    assert y2 is not None
