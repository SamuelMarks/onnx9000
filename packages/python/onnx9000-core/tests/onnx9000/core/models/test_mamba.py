"""Unit tests for Mamba SSM model in `onnx9000.core.models.mamba`."""

from onnx9000.core.ir import Tensor, Variable
from onnx9000.core.models.mamba import Mamba, MambaBlock, get_param, mamba_130m


def test_mamba_model() -> None:
    """Verify MambaBlock, Mamba, and mamba_130m."""
    assert isinstance(get_param("w", [8]), Variable)

    blk = MambaBlock(d_model=32, d_state=8, d_conv=4, expand=2, prefix="b0")
    x = Tensor("x", (1, 10, 32), 1)
    assert isinstance(blk(x), Tensor)

    model = Mamba(vocab_size=100, d_model=32, n_layer=1, d_state=8)
    assert isinstance(model, Mamba)
    ids = Tensor("ids", (1, 10), 7)
    assert isinstance(model(ids), Tensor)
    assert isinstance(mamba_130m(), Mamba)
