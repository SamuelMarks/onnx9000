"""Test state_dict parity."""

from onnx9000.frontends.frontend.nn.module import Module
from onnx9000.frontends.frontend.tensor import Parameter, Tensor
from onnx9000.frontends.frontend.nn.conv import Conv2d
from onnx9000.frontends.frontend.nn.linear import Linear
from onnx9000.frontends.frontend.nn.normalization import BatchNorm2d
from onnx9000.frontends.frontend.nn.containers import Sequential


class MyModel(Module):
    """Provides semantic functionality and verification."""

    def __init__(self):
        """Provides semantic functionality and verification."""
        super().__init__()
        self.conv1 = Conv2d(3, 16, 3)
        self.bn1 = BatchNorm2d(16)
        self.seq = Sequential(Linear(16, 32), Linear(32, 10))


def test_state_dict_keys():
    """Provides semantic functionality and verification."""
    m = MyModel()
    sd = m.state_dict()
    expected_keys = {
        "conv1.weight",
        "conv1.bias",
        "bn1.weight",
        "bn1.bias",
        "bn1.running_mean",
        "bn1.running_var",
        "seq.0.weight",
        "seq.0.bias",
        "seq.1.weight",
        "seq.1.bias",
    }
    assert set(sd.keys()) == expected_keys


def test_conv2d_parity():
    """Provides semantic functionality and verification."""
    from onnx9000 import GraphBuilder, Tracing
    from onnx9000.core.dtypes import DType

    conv = Conv2d(3, 16, 3, padding=1, bias=False)
    assert conv.weight.shape == (16, 3, 3, 3)
    assert conv.bias is None
    builder = GraphBuilder("dummy")
    with Tracing(builder):
        x = Tensor((1, 3, 224, 224), DType.FLOAT32, "x")
        y = conv(x)
    assert y is not None
