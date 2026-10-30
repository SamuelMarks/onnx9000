"""Unit tests for CLIP model in `onnx9000.core.models.clip`."""

from onnx9000.core.ir import Tensor, Variable
from onnx9000.core.models.clip import CLIP, clip_vit_base_patch16, get_param


def test_clip_model() -> None:
    """Verify CLIP instantiation and forward execution."""
    p = get_param("logit_scale", [1])
    assert isinstance(p, Variable)

    model = clip_vit_base_patch16(
        vision_layers=1,
        text_layers=1,
        embed_dim=64,
        vision_width=64,
        vision_heads=4,
        text_width=64,
        text_heads=4,
        vocab_size=100,
    )
    assert isinstance(model, CLIP)
    img = Tensor("img", (1, 3, 224, 224), 1)
    txt = Tensor("txt", (1, 16), 7)
    img_feat, txt_feat = model(img, txt)
    assert isinstance(img_feat, Tensor)
    assert isinstance(txt_feat, Tensor)
