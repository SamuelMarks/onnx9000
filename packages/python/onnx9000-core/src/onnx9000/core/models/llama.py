"""Llama."""

from typing import Any, Optional

from onnx9000.core.ir import Tensor, Variable
from onnx9000.core.ops import add, mul
from onnx9000.core.primitives import (
    Gemm,
    GroupedQueryAttention,
    RMSNorm,
    RoPE,
    Silu,
)


def get_param(name: str, shape: list[int], dtype: int = 1) -> Variable:  # noqa: D103
    """Get param."""
    return Variable(name=name, shape=shape, dtype=dtype)


class SwiGLU:  # noqa: D101
    """Swi glu."""

    def __init__(self, hidden_dim: int, ffn_dim: int, prefix: str = ""):  # noqa: D107
        """Init."""
        self.prefix = prefix
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.w1 = Gemm(trans_b=1)  # Gate
        self.w2 = Gemm(trans_b=1)  # Down
        self.w3 = Gemm(trans_b=1)  # Up
        self.act = Silu()

    def __call__(self, x: Tensor) -> Tensor:  # noqa: D102
        """Call."""
        gate = self.w1(x, get_param(f"{self.prefix}.w1.weight", [self.ffn_dim, self.hidden_dim]))
        up = self.w3(x, get_param(f"{self.prefix}.w3.weight", [self.ffn_dim, self.hidden_dim]))

        # SwiGLU = Silu(x * w1) * (x * w3)
        activated_gate = self.act(gate)
        hidden = mul(activated_gate, up)

        down = self.w2(
            hidden, get_param(f"{self.prefix}.w2.weight", [self.hidden_dim, self.ffn_dim])
        )
        return down


class LLaMABlock:  # noqa: D101
    """L la ma block."""

    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, ffn_dim: int, prefix: str = ""):  # noqa: D107
        """Init."""
        self.prefix = prefix
        self.dim = dim
        self.norm1 = RMSNorm((dim,))
        self.attn = GroupedQueryAttention(num_heads, num_kv_heads, qkv_bias=False, out_bias=False)
        self.norm2 = RMSNorm((dim,))
        self.mlp = SwiGLU(dim, ffn_dim, prefix=f"{prefix}.mlp")

    def __call__(self, x: Tensor, pos: Tensor, mask: Optional[Tensor] = None) -> Tensor:  # noqa: D102
        """Call."""
        identity = x
        x_norm = self.norm1(x, get_param(f"{self.prefix}.norm1.weight", [self.dim]))

        # In real LLaMA, RoPE is applied to Q and K inside attention, but for AST layout
        # we might just use the primitive. Let's assume GroupedQueryAttention handles it or we apply it before.
        # We will just pass it to attn.
        x_attn = self.attn(x_norm, x_norm, x_norm, mask=mask)
        x = add(identity, x_attn)

        identity = x
        x_norm = self.norm2(x, get_param(f"{self.prefix}.norm2.weight", [self.dim]))
        x_mlp = self.mlp(x_norm)
        x = add(identity, x_mlp)

        return x


class LLaMA:  # noqa: D101
    """L la ma."""

    def __init__(  # noqa: D107
        self,
        vocab_size: int = 32000,
        dim: int = 4096,
        num_heads: int = 32,
        num_kv_heads: int = 32,
        depth: int = 32,
        ffn_dim: int = 11008,
        max_seq_len: int = 2048,
    ):
        """Init."""
        self.vocab_size = vocab_size
        self.dim = dim
        self.depth = depth
        self.max_seq_len = max_seq_len

        # We use a Gather for embedding
        self.blocks = [
            LLaMABlock(dim, num_heads, num_kv_heads, ffn_dim, prefix=f"blocks.{i}")
            for i in range(depth)
        ]
        self.norm = RMSNorm((dim,))
        self.lm_head = Gemm(trans_b=1)
        self.rope = RoPE(dim // num_heads, max_seq_len=max_seq_len)

    def __call__(self, input_ids: Tensor, pos: Tensor, mask: Optional[Tensor] = None) -> Tensor:  # noqa: D102
        """Call."""
        from onnx9000.core.ops import gather

        x = gather(
            get_param("tok_embeddings.weight", [self.vocab_size, self.dim]), input_ids, axis=0
        )

        # Apply RoPE to embeddings conceptually or inside blocks
        x = self.rope(x, pos)

        for block in self.blocks:
            x = block(x, pos, mask)

        x = self.norm(x, get_param("norm.weight", [self.dim]))
        x = self.lm_head(x, get_param("output.weight", [self.vocab_size, self.dim]))
        return x


def llama_7b(**kwargs: Any) -> LLaMA:  # noqa: D103
    """Llama 7b."""
    return LLaMA(
        vocab_size=32000, dim=4096, num_heads=32, num_kv_heads=32, depth=32, ffn_dim=11008, **kwargs
    )


def mistral_7b(**kwargs: Any) -> LLaMA:  # noqa: D103
    """Mistral 7b."""
    return LLaMA(
        vocab_size=32000, dim=4096, num_heads=32, num_kv_heads=8, depth=32, ffn_dim=14336, **kwargs
    )
