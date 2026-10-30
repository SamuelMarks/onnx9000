"""Unit tests for MaskedAutoencoderViT in `onnx9000.core.models.mae`."""

from onnx9000.core.ir import Tensor, Variable
from onnx9000.core.models.mae import MaskedAutoencoderViT, get_param, mae_vit_base_patch16


def test_mae_model() -> None:
    """Verify MaskedAutoencoderViT and mae_vit_base_patch16."""
    assert isinstance(get_param("w", [8]), Variable)

    model = MaskedAutoencoderViT(
        img_size=32,
        patch_size=16,
        embed_dim=64,
        depth=1,
        num_heads=4,
        decoder_embed_dim=32,
        decoder_depth=1,
        decoder_num_heads=4,
    )
    assert isinstance(model, MaskedAutoencoderViT)
    x = Tensor("x", (1, 3, 32, 32), 1)
    mask_idx = Tensor("mask_idx", (1, 2), 7)
    assert isinstance(model(x, mask_idx), Tensor)
    assert isinstance(mae_vit_base_patch16(), MaskedAutoencoderViT)
