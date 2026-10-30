"""Unit tests for ResNet model in `onnx9000.core.models.resnet`."""

from onnx9000.core.ir import Tensor, Variable
from onnx9000.core.models.resnet import BasicBlock, ResNet, get_param, resnet18, resnet50


def test_resnet_model() -> None:
    """Verify BasicBlock, ResNet, resnet18, and resnet50."""
    assert isinstance(get_param("w", [8]), Variable)

    blk = BasicBlock(inplanes=64, planes=64, stride=1, downsample=True, prefix="b0")
    x_blk = Tensor("x_blk", (1, 64, 56, 56), 1)
    assert isinstance(blk(x_blk), Tensor)

    m18 = resnet18(num_classes=10)
    assert isinstance(m18, ResNet)
    x = Tensor("x", (1, 3, 224, 224), 1)
    assert isinstance(m18(x), Tensor)

    m50 = resnet50(num_classes=10)
    assert isinstance(m50, ResNet)
