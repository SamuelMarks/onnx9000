"""Unit tests for RWKV model in `onnx9000.core.models.rwkv`."""

from onnx9000.core.ir import Tensor, Variable
from onnx9000.core.models.rwkv import (
    RWKV,
    RWKVBlock,
    RWKVChannelMix,
    RWKVTimeMix,
    get_param,
    rwkv_v4,
)


def test_rwkv_model() -> None:
    """Verify RWKVTimeMix, RWKVChannelMix, RWKVBlock, RWKV, and rwkv_v4."""
    assert isinstance(get_param("w", [8]), Variable)

    tm = RWKVTimeMix(dim=32, prefix="tm")
    cm = RWKVChannelMix(dim=32, prefix="cm")
    blk = RWKVBlock(dim=32, prefix="b0")
    x = Tensor("x", (1, 8, 32), 1)
    assert isinstance(tm(x), Tensor)
    assert isinstance(cm(x), Tensor)
    assert isinstance(blk(x), Tensor)

    model = RWKV(vocab_size=100, dim=32, depth=1)
    assert isinstance(model, RWKV)
    ids = Tensor("ids", (1, 8), 7)
    assert isinstance(model(ids), Tensor)
    assert isinstance(rwkv_v4(), RWKV)
