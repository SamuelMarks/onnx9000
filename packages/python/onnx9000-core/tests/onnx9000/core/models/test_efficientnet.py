"""Unit tests for EfficientNet model in `onnx9000.core.models.efficientnet`."""

from onnx9000.core.ir import Tensor, Variable
from onnx9000.core.models.efficientnet import (
    EfficientNet,
    MBConv,
    SqueezeExcitation,
    efficientnet_b0,
    get_param,
)


def test_efficientnet_model() -> None:
    """Verify SqueezeExcitation, MBConv, EfficientNet, and efficientnet_b0."""
    assert isinstance(get_param("w", [8]), Variable)

    se = SqueezeExcitation(in_channels=16, squeeze_channels=4, prefix="se")
    x16 = Tensor("x16", (1, 16, 8, 8), 1)
    assert isinstance(se(x16), Tensor)

    mb = MBConv(16, 16, expand_ratio=2, stride=1, kernel_size=3, prefix="mb")
    assert isinstance(mb(x16), Tensor)

    model = efficientnet_b0(num_classes=10)
    assert isinstance(model, EfficientNet)
    x = Tensor("x", (1, 3, 224, 224), 1)
    assert isinstance(model(x), Tensor)
