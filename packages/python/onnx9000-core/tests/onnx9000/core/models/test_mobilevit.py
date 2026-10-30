"""Unit tests for MobileViT model in `onnx9000.core.models.mobilevit`."""

from onnx9000.core.ir import Tensor, Variable
from onnx9000.core.models.mobilevit import MobileViT, MobileViTBlock, get_param, mobilevit_s


def test_mobilevit_model() -> None:
    """Verify MobileViTBlock, MobileViT, and mobilevit_s."""
    assert isinstance(get_param("w", [8]), Variable)

    blk = MobileViTBlock(dim=16, out_dim=32, num_heads=4, mlp_dim=32, prefix="b0")
    x_blk = Tensor("x_blk", (1, 16, 14, 14), 1)
    assert isinstance(blk(x_blk), Tensor)

    model = mobilevit_s(num_classes=10)
    assert isinstance(model, MobileViT)
    x = Tensor("x", (1, 3, 224, 224), 1)
    assert isinstance(model(x), Tensor)
