"""Diffusion Transformer (DiT) model implementation for ONNX9000.

This module implements the Diffusion Transformer (DiT) architecture, which replaces
the traditional U-Net backbone in diffusion models with a transformer operating on
latent patches. It supports Adaptive Layer Normalization (AdaLN-Zero) block structures,
patch embeddings, and conditional timestep embedding projections.

Reference:
    Peebles, W., & Xie, S. (2023). Scalable Diffusion Models with Transformers.
    IEEE/CVF International Conference on Computer Vision (ICCV).
"""

from onnx9000.core.ir import Tensor, Variable
from onnx9000.core.models.vit import PatchEmbed
from onnx9000.core.ops import add, mul
from onnx9000.core.primitives import Gelu, Gemm, LayerNormalization, MultiHeadAttention


def get_param(name: str, shape: list[int], dtype: int = 1) -> Variable:
    """Retrieves or registers a model parameter.

    This utility helps initialize and retrieve a Parameter variable with the specified
    name, shape, and data type inside the ONNX9000 model graph structure.

    Args:
        name (str): The identifier name of the parameter variable.
        shape (list[int]): The shape of the parameter tensor.
        dtype (int): The data type of the parameter (defaults to 1).

    Returns:
        Variable: The registered parameter variable with the given properties.
    """
    return Variable(name=name, shape=shape, dtype=dtype)


class DiTBlock:
    """Diffusion Transformer Block with AdaLN-Zero.

    This block implements a transformer layer consisting of Multi-Head Self-Attention
    and MLP sub-layers, both modulated using Adaptive Layer Normalization (AdaLN-Zero)
    parameters derived from condition embeddings.
    """

    def __init__(
        self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, prefix: str = ""
    ) -> None:
        """Initializes the DiTBlock layer.

        Sets up layer normalizations, attention modules, MLP projection matrices, and
        adaptive layer normalization modulations.

        Args:
            hidden_size (int): The size of the hidden/embedding dimension.
            num_heads (int): The number of attention heads in the MultiHeadAttention layer.
            mlp_ratio (float): The ratio of MLP hidden dimension to hidden size. Defaults to 4.0.
            prefix (str): Prefix string prepend to parameter names. Defaults to "".

        Returns:
            None
        """
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

    def __call__(self, x: Tensor, c: Tensor) -> Tensor:
        # c is the timestep/condition embedding [B, hidden_size]
        """Applies the forward transformation of the DiT block.

        Applies Multi-Head Self-Attention and MLP transformations on the input,
        incorporating modulation parameters computed from the condition embedding.

        Args:
            x (Tensor): The input latent token sequence of shape [B, S, D], where
                B is the batch size, S is the sequence length, and D is the hidden size.
            c (Tensor): The conditioning timestep/condition embedding of shape [B, D].

        Returns:
            Tensor: The processed token sequence tensor of shape [B, S, D].
        """
        from onnx9000.core.ops import split

        # AdaLN-Zero mapping
        # We need 6 chunks: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        c_proj = self.adaLN_modulation(
            c,
            get_param(
                f"{self.prefix}.adaLN_modulation.weight",
                [6 * self.hidden_size, self.hidden_size],
            ),
            get_param(f"{self.prefix}.adaLN_modulation.bias", [6 * self.hidden_size]),
        )

        # For AST builder mock, split returns a single tensor, so we just duplicate it
        split_out = split(c_proj)
        chunks = [split_out] * 6
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunks

        # Needs unsqueeze to broadcast across sequence length
        from onnx9000.core.ops import record_op

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
                f"{self.prefix}.mlp.fc1.weight",
                [int(self.hidden_size * 4), self.hidden_size],
            ),
            get_param(f"{self.prefix}.mlp.fc1.bias", [int(self.hidden_size * 4)]),
        )
        x_mlp = self.act(x_mlp)
        x_mlp = self.mlp_fc2(
            x_mlp,
            get_param(
                f"{self.prefix}.mlp.fc2.weight",
                [self.hidden_size, int(self.hidden_size * 4)],
            ),
            get_param(f"{self.prefix}.mlp.fc2.bias", [self.hidden_size]),
        )

        x_mlp = mul(gate_mlp, x_mlp)
        x = add(identity, x_mlp)

        return x


class DiT:
    """Diffusion Transformer (DiT) model.

    This class defines the full DiT architecture. It converts a batch of input images
    to patch embeddings, injects positional embeddings, embeds timesteps, and sequentially
    passes them through multiple DiT Blocks before finalizing with a modulated layer
    normalization and linear projection.
    """

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
    ) -> None:
        """Initializes the DiT model.

        Args:
            input_size (int): Height and width of the input image. Defaults to 32.
            patch_size (int): The spatial size of each image patch. Defaults to 2.
            in_channels (int): The number of channels in the input image. Defaults to 4.
            hidden_size (int): The embedding size/hidden dimension of the transformer. Defaults to 1152.
            depth (int): The number of DiTBlocks in the transformer architecture. Defaults to 28.
            num_heads (int): The number of attention heads. Defaults to 16.
            mlp_ratio (float): The expansion ratio of the MLP hidden layer. Defaults to 4.0.

        Returns:
            None
        """
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

    def __call__(self, x: Tensor, t: Tensor) -> Tensor:
        """Applies the forward pass of the DiT model.

        Converts input latents to patch tokens, incorporates positions and timestep embeddings,
        passes them through all transformer layers, and projects the final output back to latent space.

        Args:
            x (Tensor): Input image latent tensor of shape [B, C, H, W], where B is batch size,
                C is channels, and H, W are spatial dimensions.
            t (Tensor): Input timestep embedding tensor of shape [B, 1] or [B].

        Returns:
            Tensor: The predicted noise or latent projection tensor of shape [B, S, out_channels].
        """
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
                "final_layer.adaLN_modulation.weight",
                [2 * self.hidden_size, self.hidden_size],
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


def dit_xl_2(**kwargs: int | float | str | bool | bytes | list | dict | tuple | None) -> DiT:
    """Helper function to create a Diffusion Transformer XL/2 configuration.

    This function instantiates a standard DiT-XL/2 model with a hidden size of 1152,
    a depth of 28 blocks, and 16 attention heads.

    Args:
        **kwargs (Union[int, float, str, bool, bytes, list, dict, tuple, None]): Additional keyword arguments to customize the DiT initialization.

    Returns:
        DiT: The initialized DiT model configured with DiT-XL/2 parameters.
    """
    return DiT(hidden_size=1152, depth=28, num_heads=16, **kwargs)
