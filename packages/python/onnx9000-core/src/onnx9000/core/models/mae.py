"""Masked Autoencoder (MAE) with a Vision Transformer (ViT) backbone.

This module implements a Masked Autoencoder (MAE) architecture using a Vision Transformer
backbone. It includes the encoder, decoder, and specialized utility functions for param
initialization, patching, masking, and feature reconstruction for self-supervised learning
on vision tasks.
"""

from onnx9000.core.ir import Tensor, Variable
from onnx9000.core.models.vit import Block, PatchEmbed
from onnx9000.core.ops import add, concat, gather, scatter_nd
from onnx9000.core.primitives import Gemm, LayerNormalization


def get_param(name: str, shape: list[int], dtype: int = 1) -> Variable:
    """Get or create a parameter Variable with the specified shape and data type.

    Args:
        name (str): The unique identifier name for the parameter.
        shape (list[int]): The shape/dimensions of the parameter tensor.
        dtype (int, optional): The ONNX data type code of the parameter. Defaults to 1 (float32).

    Returns:
        Variable: An initialized Variable object representing the parameter.

    """
    return Variable(name=name, shape=shape, dtype=dtype)


class MaskedAutoencoderViT:
    """Masked Autoencoder with a Vision Transformer (ViT) backbone.

    This class implements the full Masked Autoencoder (MAE) model. It supports patch
    embedding of input images, masking a subset of patches, encoding the unmasked patches,
    and reconstructing the original patches using a lightweight decoder.

    Attributes:
        img_size (int): The height and width of the input square image.
        patch_size (int): The height and width of each square patch.
        num_patches (int): The total number of patches in an image.
        embed_dim (int): The channel/embedding dimension of the encoder transformer.
        decoder_embed_dim (int): The channel/embedding dimension of the decoder transformer.
        patch_embed (PatchEmbed): The module that embeds image patches.
        blocks (list[Block]): The sequence of transformer blocks for the encoder.
        norm (LayerNormalization): The normalization layer for the encoder.
        decoder_embed (Gemm): The projection layer from encoder dimension to decoder dimension.
        decoder_blocks (list[Block]): The sequence of transformer blocks for the decoder.
        decoder_norm (LayerNormalization): The normalization layer for the decoder.
        decoder_pred (Gemm): The projection layer from decoder dimension to the original pixel space.

    """

    def __init__(
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
        norm_layer: int
        | float
        | str
        | bool
        | bytes
        | list
        | dict
        | tuple
        | None = LayerNormalization,
    ) -> None:
        """Initializes the Masked Autoencoder with Vision Transformer (MAE-ViT) model.

        Args:
            img_size (int): Height and width of the input image. Defaults to 224.
            patch_size (int): Size of each square patch. Defaults to 16.
            in_chans (int): Number of input image channels. Defaults to 3.
            embed_dim (int): Embedding dimension of the encoder. Defaults to 1024.
            depth (int): Depth (number of layers) of the encoder transformer. Defaults to 24.
            num_heads (int): Number of attention heads in the encoder. Defaults to 16.
            decoder_embed_dim (int): Embedding dimension of the decoder. Defaults to 512.
            decoder_depth (int): Depth (number of layers) of the decoder. Defaults to 8.
            decoder_num_heads (int): Number of attention heads in the decoder. Defaults to 16.
            mlp_ratio (float): Ratio of MLP hidden dim to embedding dim. Defaults to 4.0.
            norm_layer (Union[int, float, str, bool, bytes, list, dict, tuple, None]): Normalization layer class. Defaults to LayerNormalization.

        """
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

    def forward_encoder(self, x: Tensor, mask_indices: Tensor) -> tuple[Tensor, Tensor]:
        """Runs the encoder forward pass of the MAE.

        This method projects the input image to patches, adds the positional embeddings,
        gathers only the unmasked patches according to the mask indices, prepends the
        class (CLS) token with its positional embedding, and runs the sequence through the
        encoder transformer blocks and normalization layer.

        Args:
            x (Tensor): Input image tensor of shape [B, C, H, W].
            mask_indices (Tensor): Indices of the unmasked/kept patches to keep, of shape [B, kept_len].

        Returns:
            tuple[Tensor, Tensor]: A tuple containing:
                - Tensor: The encoded latent representations of shape [B, 1 + kept_len, embed_dim].
                - Tensor: The original mask indices propagated forward.

        """
        # Embed patches
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
            pos_embed,
            constant([0], dtype=7),
            constant([1], dtype=7),
            constant([1], dtype=7),
        )
        cls_token = add(cls_token, cls_pos_embed)

        x = concat([cls_token, x], axis=1)

        # Apply blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(
            x,
            get_param("norm.weight", [self.embed_dim]),
            get_param("norm.bias", [self.embed_dim]),
        )

        return x, mask_indices

    def forward_decoder(self, x: Tensor, mask_indices: Tensor) -> Tensor:
        """Runs the decoder forward pass of the MAE.

        This method projects the latent representation back to the decoder embedding dimension,
        scatters the kept patches and the placeholder mask tokens back to their original sequence positions
        (unmasking the sequence), appends the class token, adds the decoder positional embeddings,
        passes the full sequence through the decoder transformer blocks, and finally projects the features
        to predict the original pixel values of the patches.

        Args:
            x (Tensor): Latent representation tensor from the encoder of shape [B, 1 + kept_len, embed_dim].
            mask_indices (Tensor): Indices of the kept/unmasked patches of shape [B, kept_len] used to scatter the tokens.

        Returns:
            Tensor: Predicted patch pixel values of shape [B, num_patches, patch_size**2 * 3].

        """
        # Embed tokens
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

    def __call__(self, x: Tensor, mask_indices: Tensor) -> Tensor:
        """Executes the full forward pass of the MAE (encoding followed by decoding).

        Args:
            x (Tensor): Input image tensor of shape [B, C, H, W].
            mask_indices (Tensor): Indices of the kept/unmasked patches of shape [B, kept_len].

        Returns:
            Tensor: Reconstructed patch pixel predictions of shape [B, num_patches, patch_size**2 * 3].

        """
        latent, mask_indices = self.forward_encoder(x, mask_indices)
        pred = self.forward_decoder(latent, mask_indices)
        return pred


def mae_vit_base_patch16(
    **kwargs: int | float | str | bool | bytes | list | dict | tuple | None,
) -> MaskedAutoencoderViT:
    """Instantiates a standard MAE ViT-Base model with a patch size of 16.

    This helper function configures a standard Masked Autoencoder with a ViT-Base encoder
    (12 layers, embedding dimension of 768, 12 attention heads) and a corresponding
    standard MAE decoder (8 layers, embedding dimension of 512, 16 attention heads).

    Args:
        **kwargs (Union[int, float, str, bool, bytes, list, dict, tuple, None]): Additional keyword arguments to pass to the MaskedAutoencoderViT constructor.

    Returns:
        MaskedAutoencoderViT: An instance of MaskedAutoencoderViT configured with base hyperparameters.

    """
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
