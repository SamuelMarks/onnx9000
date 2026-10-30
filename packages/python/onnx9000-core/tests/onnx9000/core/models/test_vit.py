"""Unit tests for VisionTransformer (ViT) model in `onnx9000.core.models.vit`."""

from onnx9000.core.ir import Tensor, Variable
from onnx9000.core.models.vit import (
    Block,
    PatchEmbed,
    VisionTransformer,
    get_param,
    vit_base_patch16_224,
)


def test_vit_model() -> None:
    """Verify PatchEmbed, Block, VisionTransformer, and vit_base_patch16_224."""
    assert isinstance(get_param("w", [8]), Variable)

    pe = PatchEmbed(img_size=32, patch_size=16, in_chans=3, embed_dim=64, prefix="pe")
    x_img = Tensor("x_img", (1, 3, 32, 32), 1)
    patches = pe(x_img)
    assert isinstance(patches, Tensor)

    blk = Block(dim=64, num_heads=4, prefix="b0")
    assert isinstance(blk(patches), Tensor)

    model = VisionTransformer(
        img_size=32, patch_size=16, embed_dim=64, depth=1, num_heads=4, num_classes=10
    )
    assert isinstance(model, VisionTransformer)
    assert isinstance(model(x_img), Tensor)
    assert isinstance(vit_base_patch16_224(), VisionTransformer)
