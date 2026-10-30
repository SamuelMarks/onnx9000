"""Unit tests for Mixtral Sparse MoE model in `onnx9000.core.models.mixtral`."""

from onnx9000.core.ir import Tensor, Variable
from onnx9000.core.models.mixtral import Mixtral, MixtralBlock, SparseMoE, get_param, mixtral_8x7b


def test_mixtral_model() -> None:
    """Verify SparseMoE, MixtralBlock, Mixtral, and mixtral_8x7b."""
    assert isinstance(get_param("w", [8]), Variable)

    moe = SparseMoE(num_experts=2, top_k=1, dim=32, ffn_dim=64, prefix="moe")
    x = Tensor("x", (1, 8, 32), 1)
    pos = Tensor("pos", (8,), 7)
    assert isinstance(moe(x), Tensor)

    blk = MixtralBlock(
        dim=32, num_heads=4, num_kv_heads=2, ffn_dim=64, num_experts=2, top_k=1, prefix="b0"
    )
    assert isinstance(blk(x, pos), Tensor)

    model = Mixtral(
        vocab_size=100, dim=32, num_heads=4, num_kv_heads=2, depth=1, ffn_dim=64, num_experts=2
    )
    assert isinstance(model, Mixtral)
    ids = Tensor("ids", (1, 8), 7)
    assert isinstance(model(ids, pos), Tensor)
    assert isinstance(mixtral_8x7b(), Mixtral)
