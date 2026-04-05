"""Mae."""

from typing import Any

from onnx9000.core.ir import Tensor, Variable
from onnx9000.core.models.vit import Block, PatchEmbed
from onnx9000.core.ops import add, concat, flatten, gather, reshape, scatter_nd
from onnx9000.core.primitives import ConvND, Gemm, LayerNormalization


def get_param(name: str, shape: list[int], dtype: int = 1) -> Variable:  # noqa: D103
    """Get param."""
    return Variable(name=name, shape=shape, dtype=dtype)


class MaskedAutoencoderViT:
    """Masked Autoencoder with VisionTransformer backbone."""

    def __init__(  # noqa: D107
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_layer: Any = LayerNormalization,
    ):
        """Init."""
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim

        # Encoder
        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_chans, embed_dim, prefix="patch_embed"
        )
        self.blocks = [
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, prefix=f"blocks.{i}")
            for i in range(depth)
        ]
        self.norm = norm_layer((embed_dim,))

        # Decoder
        self.decoder_embed = Gemm(trans_b=1)
        self.decoder_blocks = [
            Block(
                decoder_embed_dim,
                decoder_num_heads,
                mlp_ratio,
                qkv_bias=True,
                prefix=f"decoder_blocks.{i}",
            )
            for i in range(decoder_depth)
        ]
        self.decoder_norm = norm_layer((decoder_embed_dim,))
        self.decoder_pred = Gemm(trans_b=1)

    def forward_encoder(self, x: Tensor, mask_indices: Tensor) -> tuple[Tensor, Tensor]:  # noqa: D102
        # Embed patches
        """Forward encoder."""
        x = self.patch_embed(x)

        # Add pos embed w/o cls token
        pos_embed = get_param("pos_embed", [1, self.num_patches + 1, self.embed_dim])

        from onnx9000.core.ops import constant
        from onnx9000.core.ops import slice as slice_op

        starts = constant([1], dtype=7)
        ends = constant([self.num_patches + 1], dtype=7)
        axes = constant([1], dtype=7)
        pos_embed_no_cls = slice_op(pos_embed, starts, ends, axes)

        x = add(x, pos_embed_no_cls)

        # Gather kept patches (we simulate mask by gathering indices)
        x = gather(x, mask_indices, axis=1)

        # Append cls token
        cls_token = get_param("cls_token", [1, 1, self.embed_dim])
        cls_pos_embed = slice_op(
            pos_embed, constant([0], dtype=7), constant([1], dtype=7), constant([1], dtype=7)
        )
        cls_token = add(cls_token, cls_pos_embed)

        x = concat([cls_token, x], axis=1)

        # Apply blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(
            x, get_param("norm.weight", [self.embed_dim]), get_param("norm.bias", [self.embed_dim])
        )

        return x, mask_indices

    def forward_decoder(self, x: Tensor, mask_indices: Tensor) -> Tensor:  # noqa: D102
        # Embed tokens
        """Forward decoder."""
        x = self.decoder_embed(
            x,
            get_param("decoder_embed.weight", [self.decoder_embed_dim, self.embed_dim]),
            get_param("decoder_embed.bias", [self.decoder_embed_dim]),
        )

        # Scatter unmasked tokens back into full sequence
        # ScatterND requires data, indices, updates
        # To reconstruct the full sequence, we start with mask tokens
        mask_token = get_param("mask_token", [1, 1, self.decoder_embed_dim])
        # Expand mask_token to full size
        from onnx9000.core.ops import constant, tile

        full_mask = tile(mask_token, constant([1, self.num_patches, 1], dtype=7))

        # x is [B, 1+kept_len, C], we drop the cls token for scattering
        starts = constant([1], dtype=7)
        ends = constant([self.num_patches + 1], dtype=7)  # just large enough
        axes = constant([1], dtype=7)
        from onnx9000.core.ops import slice as slice_op

        x_no_cls = slice_op(x, starts, ends, axes)

        # Unmask via ScatterND
        x_full = scatter_nd(full_mask, mask_indices, x_no_cls)

        # Add cls token back
        x_cls = slice_op(x, constant([0], dtype=7), constant([1], dtype=7), constant([1], dtype=7))
        x = concat([x_cls, x_full], axis=1)

        # Add pos embed
        decoder_pos_embed = get_param(
            "decoder_pos_embed", [1, self.num_patches + 1, self.decoder_embed_dim]
        )
        x = add(x, decoder_pos_embed)

        # Apply blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(
            x,
            get_param("decoder_norm.weight", [self.decoder_embed_dim]),
            get_param("decoder_norm.bias", [self.decoder_embed_dim]),
        )

        # Predict pixels
        x = self.decoder_pred(
            x,
            get_param("decoder_pred.weight", [self.patch_size**2 * 3, self.decoder_embed_dim]),
            get_param("decoder_pred.bias", [self.patch_size**2 * 3]),
        )

        # Drop cls token
        x = slice_op(
            x,
            constant([1], dtype=7),
            constant([self.num_patches + 1], dtype=7),
            constant([1], dtype=7),
        )
        return x

    def __call__(self, x: Tensor, mask_indices: Tensor) -> Tensor:  # noqa: D102
        """Call."""
        latent, mask_indices = self.forward_encoder(x, mask_indices)
        pred = self.forward_decoder(latent, mask_indices)
        return pred


def mae_vit_base_patch16(**kwargs: Any) -> MaskedAutoencoderViT:  # noqa: D103
    """Mae vit base patch16."""
    return MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        **kwargs,
    )
