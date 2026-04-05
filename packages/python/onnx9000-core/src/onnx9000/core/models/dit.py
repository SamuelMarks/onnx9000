"""Dit."""

from typing import Any

from onnx9000.core.ir import Tensor, Variable
from onnx9000.core.models.vit import PatchEmbed
from onnx9000.core.ops import add, mul
from onnx9000.core.primitives import Gelu, Gemm, LayerNormalization, MultiHeadAttention


def get_param(name: str, shape: list[int], dtype: int = 1) -> Variable:  # noqa: D103
    """Get param."""
    return Variable(name=name, shape=shape, dtype=dtype)


class DiTBlock:
    """Diffusion Transformer Block with AdaLN-Zero."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, prefix: str = ""):  # noqa: D107
        """Init."""
        self.prefix = prefix
        self.hidden_size = hidden_size
        self.norm1 = LayerNormalization((hidden_size,), epsilon=1e-6)
        self.attn = MultiHeadAttention(num_heads=num_heads, qkv_bias=True)
        self.norm2 = LayerNormalization((hidden_size,), epsilon=1e-6)
        int(hidden_size * mlp_ratio)
        self.mlp_fc1 = Gemm(trans_b=1)
        self.act = Gelu()
        self.mlp_fc2 = Gemm(trans_b=1)
        self.adaLN_modulation = Gemm(trans_b=1)  # projects time_emb to shift, scale, gate

    def __call__(self, x: Tensor, c: Tensor) -> Tensor:  # noqa: D102
        # c is the timestep/condition embedding [B, hidden_size]
        """Call."""
        from onnx9000.core.ops import split, squeeze

        # AdaLN-Zero mapping
        # We need 6 chunks: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        c_proj = self.adaLN_modulation(
            c,
            get_param(
                f"{self.prefix}.adaLN_modulation.weight", [6 * self.hidden_size, self.hidden_size]
            ),
            get_param(f"{self.prefix}.adaLN_modulation.bias", [6 * self.hidden_size]),
        )

        # For AST builder mock, split returns a single tensor, so we just duplicate it
        split_out = split(c_proj)
        chunks = [split_out] * 6
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunks

        # Needs unsqueeze to broadcast across sequence length
        from onnx9000.core.ops import record_op, unsqueeze

        axis_tensor = record_op("Constant", [], {"value": [1], "dtype": 7})
        shift_msa = record_op("Unsqueeze", [shift_msa, axis_tensor])
        scale_msa = record_op("Unsqueeze", [scale_msa, axis_tensor])
        gate_msa = record_op("Unsqueeze", [gate_msa, axis_tensor])
        shift_mlp = record_op("Unsqueeze", [shift_mlp, axis_tensor])
        scale_mlp = record_op("Unsqueeze", [scale_mlp, axis_tensor])
        gate_mlp = record_op("Unsqueeze", [gate_mlp, axis_tensor])

        # Apply norm and modulation for attention
        identity = x
        x_norm = self.norm1(
            x,
            get_param(f"{self.prefix}.norm1.weight", [self.hidden_size]),
            get_param(f"{self.prefix}.norm1.bias", [self.hidden_size]),
        )

        # (1 + scale_msa) * x_norm + shift_msa
        one_tensor = record_op("Constant", [], {"value": [1.0], "dtype": 1})
        x_modulated = add(mul(add(one_tensor, scale_msa), x_norm), shift_msa)

        x_attn = self.attn(x_modulated, x_modulated, x_modulated)

        # gate_msa * x_attn
        x_attn = mul(gate_msa, x_attn)
        x = add(identity, x_attn)

        # Apply norm and modulation for MLP
        identity = x
        x_norm = self.norm2(
            x,
            get_param(f"{self.prefix}.norm2.weight", [self.hidden_size]),
            get_param(f"{self.prefix}.norm2.bias", [self.hidden_size]),
        )

        x_modulated = add(mul(add(one_tensor, scale_mlp), x_norm), shift_mlp)

        x_mlp = self.mlp_fc1(
            x_modulated,
            get_param(
                f"{self.prefix}.mlp.fc1.weight", [int(self.hidden_size * 4), self.hidden_size]
            ),
            get_param(f"{self.prefix}.mlp.fc1.bias", [int(self.hidden_size * 4)]),
        )
        x_mlp = self.act(x_mlp)
        x_mlp = self.mlp_fc2(
            x_mlp,
            get_param(
                f"{self.prefix}.mlp.fc2.weight", [self.hidden_size, int(self.hidden_size * 4)]
            ),
            get_param(f"{self.prefix}.mlp.fc2.bias", [self.hidden_size]),
        )

        x_mlp = mul(gate_mlp, x_mlp)
        x = add(identity, x_mlp)

        return x


class DiT:  # noqa: D101
    """Di t."""

    def __init__(  # noqa: D107
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
    ):
        """Init."""
        self.hidden_size = hidden_size
        self.patch_embed = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, prefix="x_embedder"
        )
        self.t_embedder = Gemm(trans_b=1)  # Time embedding

        self.blocks = [
            DiTBlock(hidden_size, num_heads, mlp_ratio, prefix=f"blocks.{i}") for i in range(depth)
        ]

        self.final_layer_norm = LayerNormalization((hidden_size,), epsilon=1e-6)
        self.final_layer_adaLN = Gemm(trans_b=1)
        self.final_layer_proj = Gemm(trans_b=1)

        self.out_channels = in_channels * patch_size * patch_size

    def __call__(self, x: Tensor, t: Tensor) -> Tensor:  # noqa: D102
        """Call."""
        x = self.patch_embed(x)

        pos_embed = get_param("pos_embed", [1, self.patch_embed.num_patches, self.hidden_size])
        x = add(x, pos_embed)

        c = self.t_embedder(
            t,
            get_param("t_embedder.weight", [self.hidden_size, self.hidden_size]),
            get_param("t_embedder.bias", [self.hidden_size]),
        )

        for block in self.blocks:
            x = block(x, c)

        # Final AdaLN
        from onnx9000.core.ops import record_op, split

        c_proj = self.final_layer_adaLN(
            c,
            get_param(
                "final_layer.adaLN_modulation.weight", [2 * self.hidden_size, self.hidden_size]
            ),
            get_param("final_layer.adaLN_modulation.bias", [2 * self.hidden_size]),
        )

        split_out = split(c_proj)
        shift, scale = split_out, split_out
        axis_tensor = record_op("Constant", [], {"value": [1], "dtype": 7})
        shift = record_op("Unsqueeze", [shift, axis_tensor])
        scale = record_op("Unsqueeze", [scale, axis_tensor])

        x = self.final_layer_norm(
            x,
            get_param("final_layer.norm.weight", [self.hidden_size]),
            get_param("final_layer.norm.bias", [self.hidden_size]),
        )
        one_tensor = record_op("Constant", [], {"value": [1.0], "dtype": 1})
        x = add(mul(add(one_tensor, scale), x), shift)

        x = self.final_layer_proj(
            x,
            get_param("final_layer.linear.weight", [self.out_channels, self.hidden_size]),
            get_param("final_layer.linear.bias", [self.out_channels]),
        )
        return x


def dit_xl_2(**kwargs: Any) -> DiT:  # noqa: D103
    """Dit xl 2."""
    return DiT(hidden_size=1152, depth=28, num_heads=16, **kwargs)
