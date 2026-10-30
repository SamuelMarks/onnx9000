"""Unit tests for Diffusion Transformer (DiT) model in `onnx9000.core.models.dit`."""

from onnx9000.core.ir import Tensor, Variable
from onnx9000.core.models.dit import DiT, DiTBlock, dit_xl_2, get_param


def test_dit_model() -> None:
    """Verify DiTBlock, DiT, and dit_xl_2."""
    assert isinstance(get_param("w", [8]), Variable)

    blk = DiTBlock(hidden_size=64, num_heads=4, prefix="b0")
    x_tok = Tensor("x_tok", (1, 16, 64), 1)
    c_emb = Tensor("c_emb", (1, 64), 1)
    assert isinstance(blk(x_tok, c_emb), Tensor)

    model = DiT(input_size=16, patch_size=4, in_channels=4, hidden_size=64, depth=1, num_heads=4)
    x_img = Tensor("x_img", (1, 4, 16, 16), 1)
    t_emb = Tensor("t_emb", (1, 64), 1)
    assert isinstance(model(x_img, t_emb), Tensor)
    assert isinstance(dit_xl_2(), DiT)
