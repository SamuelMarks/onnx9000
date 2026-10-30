"""Unit tests for ConvNeXt model in `onnx9000.core.models.convnext`."""

from onnx9000.core.ir import Tensor, Variable
from onnx9000.core.models.convnext import ConvNeXt, ConvNeXtBlock, convnext_tiny, get_param


def test_convnext_model() -> None:
    """Verify ConvNeXtBlock, ConvNeXt, and convnext_tiny."""
    assert isinstance(get_param("w", [8]), Variable)

    blk = ConvNeXtBlock(dim=32, prefix="b0")
    x_blk = Tensor("x_blk", (1, 32, 14, 14), 1)
    assert isinstance(blk(x_blk), Tensor)

    model = convnext_tiny(in_chans=3, num_classes=10)
    assert isinstance(model, ConvNeXt)
    x = Tensor("x", (1, 3, 224, 224), 1)
    assert isinstance(model(x), Tensor)
