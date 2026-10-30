"""Unit tests for SwinTransformer model in `onnx9000.core.models.swin`."""

from onnx9000.core.ir import Tensor, Variable
from onnx9000.core.models.swin import (
    SwinTransformer,
    SwinTransformerBlock,
    WindowAttention,
    get_param,
    swin_t,
)


def test_swin_model() -> None:
    """Verify WindowAttention, SwinTransformerBlock, SwinTransformer, and swin_t."""
    assert isinstance(get_param("w", [8]), Variable)

    wa = WindowAttention(dim=32, window_size=(7, 7), num_heads=4, prefix="wa")
    x_seq = Tensor("x_seq", (1, 49, 32), 1)
    assert isinstance(wa(x_seq), Tensor)

    blk = SwinTransformerBlock(
        dim=32, input_resolution=(14, 14), num_heads=4, window_size=7, shift_size=3, prefix="b0"
    )
    assert isinstance(blk(x_seq), Tensor)

    model = swin_t(embed_dim=24, depths=[1, 1, 1, 1], num_heads=[2, 2, 2, 2], num_classes=10)
    assert isinstance(model, SwinTransformer)
    x = Tensor("x", (1, 3, 224, 224), 1)
    assert isinstance(model(x), Tensor)
