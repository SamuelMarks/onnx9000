"""Unit tests for LLaMA and Mistral models in `onnx9000.core.models.llama`."""

from onnx9000.core.ir import Tensor, Variable
from onnx9000.core.models.llama import LLaMA, LLaMABlock, SwiGLU, get_param, llama_7b, mistral_7b


def test_llama_and_mistral() -> None:
    """Verify SwiGLU, LLaMABlock, LLaMA, llama_7b, and mistral_7b."""
    assert isinstance(get_param("w", [8]), Variable)

    mlp = SwiGLU(hidden_dim=64, ffn_dim=128, prefix="mlp")
    x = Tensor("x", (1, 8, 64), 1)
    pos = Tensor("pos", (8,), 7)
    assert isinstance(mlp(x), Tensor)

    blk = LLaMABlock(dim=64, num_heads=4, num_kv_heads=2, ffn_dim=128, prefix="b0")
    assert isinstance(blk(x, pos), Tensor)

    model = LLaMA(vocab_size=100, dim=64, num_heads=4, num_kv_heads=2, depth=1, ffn_dim=128)
    ids = Tensor("ids", (1, 8), 7)
    assert isinstance(model(ids, pos), Tensor)

    assert isinstance(llama_7b(), LLaMA)
    assert isinstance(mistral_7b(), LLaMA)
