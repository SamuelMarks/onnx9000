"""PyTorch-compatible neural network, activation, convolution, pooling, and normalization operations."""

from onnx9000.core.ir import AttributeValue, Tensor


def adaptive_avg_pool1d(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply adaptive_avg_pool1d to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for adaptive_avg_pool1d.
    """
    from onnx9000.core.ops.torch_auto import adaptive_avg_pool1d as core_adaptive_avg_pool1d

    return core_adaptive_avg_pool1d(input_tensor, *args, **kwargs)


def adaptive_max_pool1d(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply adaptive_max_pool1d to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for adaptive_max_pool1d.
    """
    from onnx9000.core.ops.torch_auto import adaptive_max_pool1d as core_adaptive_max_pool1d

    return core_adaptive_max_pool1d(input_tensor, *args, **kwargs)


def affine_grid_generator(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply affine_grid_generator to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for affine_grid_generator.
    """
    from onnx9000.core.ops.torch_auto import affine_grid_generator as core_affine_grid_generator

    return core_affine_grid_generator(input_tensor, *args, **kwargs)


def alpha_dropout(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply alpha_dropout to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for alpha_dropout.
    """
    from onnx9000.core.ops.torch_auto import alpha_dropout as core_alpha_dropout

    return core_alpha_dropout(input_tensor, *args, **kwargs)


def alpha_dropout_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply alpha_dropout_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for alpha_dropout_.
    """
    from onnx9000.core.ops.torch_auto import alpha_dropout_ as core_alpha_dropout_

    return core_alpha_dropout_(input_tensor, *args, **kwargs)


def avg_pool1d(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply avg_pool1d to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for avg_pool1d.
    """
    from onnx9000.core.ops.torch_auto import avg_pool1d as core_avg_pool1d

    return core_avg_pool1d(input_tensor, *args, **kwargs)


def batch_norm(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply batch_norm to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for batch_norm.
    """
    from onnx9000.core.ops.torch_auto import batch_norm as core_batch_norm

    return core_batch_norm(input_tensor, *args, **kwargs)


def batch_norm_backward_elemt(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply batch_norm_backward_elemt to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for batch_norm_backward_elemt.
    """
    from onnx9000.core.ops.torch_auto import (
        batch_norm_backward_elemt as core_batch_norm_backward_elemt,
    )

    return core_batch_norm_backward_elemt(input_tensor, *args, **kwargs)


def batch_norm_backward_reduce(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply batch_norm_backward_reduce to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for batch_norm_backward_reduce.
    """
    from onnx9000.core.ops.torch_auto import (
        batch_norm_backward_reduce as core_batch_norm_backward_reduce,
    )

    return core_batch_norm_backward_reduce(input_tensor, *args, **kwargs)


def batch_norm_elemt(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply batch_norm_elemt to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for batch_norm_elemt.
    """
    from onnx9000.core.ops.torch_auto import batch_norm_elemt as core_batch_norm_elemt

    return core_batch_norm_elemt(input_tensor, *args, **kwargs)


def batch_norm_gather_stats(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply batch_norm_gather_stats to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for batch_norm_gather_stats.
    """
    from onnx9000.core.ops.torch_auto import batch_norm_gather_stats as core_batch_norm_gather_stats

    return core_batch_norm_gather_stats(input_tensor, *args, **kwargs)


def batch_norm_gather_stats_with_counts(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply batch_norm_gather_stats_with_counts to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for batch_norm_gather_stats_with_counts.
    """
    from onnx9000.core.ops.torch_auto import (
        batch_norm_gather_stats_with_counts as core_batch_norm_gather_stats_with_counts,
    )

    return core_batch_norm_gather_stats_with_counts(input_tensor, *args, **kwargs)


def batch_norm_stats(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply batch_norm_stats to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for batch_norm_stats.
    """
    from onnx9000.core.ops.torch_auto import batch_norm_stats as core_batch_norm_stats

    return core_batch_norm_stats(input_tensor, *args, **kwargs)


def batch_norm_update_stats(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply batch_norm_update_stats to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for batch_norm_update_stats.
    """
    from onnx9000.core.ops.torch_auto import batch_norm_update_stats as core_batch_norm_update_stats

    return core_batch_norm_update_stats(input_tensor, *args, **kwargs)


def bilinear(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply bilinear to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for bilinear.
    """
    from onnx9000.core.ops.torch_auto import bilinear as core_bilinear

    return core_bilinear(input_tensor, *args, **kwargs)


def binary_cross_entropy_with_logits(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply binary_cross_entropy_with_logits to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for binary_cross_entropy_with_logits.
    """
    from onnx9000.core.ops.torch_auto import (
        binary_cross_entropy_with_logits as core_binary_cross_entropy_with_logits,
    )

    return core_binary_cross_entropy_with_logits(input_tensor, *args, **kwargs)


def celu_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply celu_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for celu_.
    """
    from onnx9000.core.ops.torch_auto import celu_ as core_celu_

    return core_celu_(input_tensor, *args, **kwargs)


def channel_shuffle(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply channel_shuffle to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for channel_shuffle.
    """
    from onnx9000.core.ops.torch_auto import channel_shuffle as core_channel_shuffle

    return core_channel_shuffle(input_tensor, *args, **kwargs)


def choose_qparams_optimized(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply choose_qparams_optimized to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for choose_qparams_optimized.
    """
    from onnx9000.core.ops.torch_auto import (
        choose_qparams_optimized as core_choose_qparams_optimized,
    )

    return core_choose_qparams_optimized(input_tensor, *args, **kwargs)


def constant_pad_nd(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply constant_pad_nd to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for constant_pad_nd.
    """
    from onnx9000.core.ops.torch_auto import constant_pad_nd as core_constant_pad_nd

    return core_constant_pad_nd(input_tensor, *args, **kwargs)


def conv1d(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply conv1d to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for conv1d.
    """
    from onnx9000.core.ops.torch_auto import conv1d as core_conv1d

    return core_conv1d(input_tensor, *args, **kwargs)


def conv2d(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply conv2d to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for conv2d.
    """
    from onnx9000.core.ops.torch_auto import conv2d as core_conv2d

    return core_conv2d(input_tensor, *args, **kwargs)


def conv3d(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply conv3d to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for conv3d.
    """
    from onnx9000.core.ops.torch_auto import conv3d as core_conv3d

    return core_conv3d(input_tensor, *args, **kwargs)


def conv_tbc(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply conv_tbc to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for conv_tbc.
    """
    from onnx9000.core.ops.torch_auto import conv_tbc as core_conv_tbc

    return core_conv_tbc(input_tensor, *args, **kwargs)


def conv_transpose1d(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply conv_transpose1d to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for conv_transpose1d.
    """
    from onnx9000.core.ops.torch_auto import conv_transpose1d as core_conv_transpose1d

    return core_conv_transpose1d(input_tensor, *args, **kwargs)


def conv_transpose2d(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply conv_transpose2d to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for conv_transpose2d.
    """
    from onnx9000.core.ops.torch_auto import conv_transpose2d as core_conv_transpose2d

    return core_conv_transpose2d(input_tensor, *args, **kwargs)


def conv_transpose3d(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply conv_transpose3d to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for conv_transpose3d.
    """
    from onnx9000.core.ops.torch_auto import conv_transpose3d as core_conv_transpose3d

    return core_conv_transpose3d(input_tensor, *args, **kwargs)


def convolution(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply convolution to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for convolution.
    """
    from onnx9000.core.ops.torch_auto import convolution as core_convolution

    return core_convolution(input_tensor, *args, **kwargs)


def cosine_embedding_loss(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply cosine_embedding_loss to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for cosine_embedding_loss.
    """
    from onnx9000.core.ops.torch_auto import cosine_embedding_loss as core_cosine_embedding_loss

    return core_cosine_embedding_loss(input_tensor, *args, **kwargs)


def cosine_similarity(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply cosine_similarity to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for cosine_similarity.
    """
    from onnx9000.core.ops.torch_auto import cosine_similarity as core_cosine_similarity

    return core_cosine_similarity(input_tensor, *args, **kwargs)


def ctc_loss(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply ctc_loss to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for ctc_loss.
    """
    from onnx9000.core.ops.torch_auto import ctc_loss as core_ctc_loss

    return core_ctc_loss(input_tensor, *args, **kwargs)


def cudnn_affine_grid_generator(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply cudnn_affine_grid_generator to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for cudnn_affine_grid_generator.
    """
    from onnx9000.core.ops.torch_auto import (
        cudnn_affine_grid_generator as core_cudnn_affine_grid_generator,
    )

    return core_cudnn_affine_grid_generator(input_tensor, *args, **kwargs)


def cudnn_batch_norm(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply cudnn_batch_norm to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for cudnn_batch_norm.
    """
    from onnx9000.core.ops.torch_auto import cudnn_batch_norm as core_cudnn_batch_norm

    return core_cudnn_batch_norm(input_tensor, *args, **kwargs)


def cudnn_convolution(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply cudnn_convolution to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for cudnn_convolution.
    """
    from onnx9000.core.ops.torch_auto import cudnn_convolution as core_cudnn_convolution

    return core_cudnn_convolution(input_tensor, *args, **kwargs)


def cudnn_convolution_add_relu(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply cudnn_convolution_add_relu to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for cudnn_convolution_add_relu.
    """
    from onnx9000.core.ops.torch_auto import (
        cudnn_convolution_add_relu as core_cudnn_convolution_add_relu,
    )

    return core_cudnn_convolution_add_relu(input_tensor, *args, **kwargs)


def cudnn_convolution_relu(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply cudnn_convolution_relu to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for cudnn_convolution_relu.
    """
    from onnx9000.core.ops.torch_auto import cudnn_convolution_relu as core_cudnn_convolution_relu

    return core_cudnn_convolution_relu(input_tensor, *args, **kwargs)


def cudnn_convolution_transpose(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply cudnn_convolution_transpose to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for cudnn_convolution_transpose.
    """
    from onnx9000.core.ops.torch_auto import (
        cudnn_convolution_transpose as core_cudnn_convolution_transpose,
    )

    return core_cudnn_convolution_transpose(input_tensor, *args, **kwargs)


def cudnn_grid_sampler(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply cudnn_grid_sampler to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for cudnn_grid_sampler.
    """
    from onnx9000.core.ops.torch_auto import cudnn_grid_sampler as core_cudnn_grid_sampler

    return core_cudnn_grid_sampler(input_tensor, *args, **kwargs)


def dequantize(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply dequantize to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for dequantize.
    """
    from onnx9000.core.ops.torch_auto import dequantize as core_dequantize

    return core_dequantize(input_tensor, *args, **kwargs)


def dropout_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply dropout_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for dropout_.
    """
    from onnx9000.core.ops.torch_auto import dropout_ as core_dropout_

    return core_dropout_(input_tensor, *args, **kwargs)


def embedding(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply embedding to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for embedding.
    """
    from onnx9000.core.ops.torch_auto import embedding as core_embedding

    return core_embedding(input_tensor, *args, **kwargs)


def embedding_bag(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply embedding_bag to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for embedding_bag.
    """
    from onnx9000.core.ops.torch_auto import embedding_bag as core_embedding_bag

    return core_embedding_bag(input_tensor, *args, **kwargs)


def embedding_renorm_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply embedding_renorm_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for embedding_renorm_.
    """
    from onnx9000.core.ops.torch_auto import embedding_renorm_ as core_embedding_renorm_

    return core_embedding_renorm_(input_tensor, *args, **kwargs)


def fake_quantize_per_channel_affine(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply fake_quantize_per_channel_affine to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for fake_quantize_per_channel_affine.
    """
    from onnx9000.core.ops.torch_auto import (
        fake_quantize_per_channel_affine as core_fake_quantize_per_channel_affine,
    )

    return core_fake_quantize_per_channel_affine(input_tensor, *args, **kwargs)


def fake_quantize_per_tensor_affine(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply fake_quantize_per_tensor_affine to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for fake_quantize_per_tensor_affine.
    """
    from onnx9000.core.ops.torch_auto import (
        fake_quantize_per_tensor_affine as core_fake_quantize_per_tensor_affine,
    )

    return core_fake_quantize_per_tensor_affine(input_tensor, *args, **kwargs)


def fbgemm_linear_fp16_weight(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply fbgemm_linear_fp16_weight to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for fbgemm_linear_fp16_weight.
    """
    from onnx9000.core.ops.torch_auto import (
        fbgemm_linear_fp16_weight as core_fbgemm_linear_fp16_weight,
    )

    return core_fbgemm_linear_fp16_weight(input_tensor, *args, **kwargs)


def fbgemm_linear_fp16_weight_fp32_activation(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply fbgemm_linear_fp16_weight_fp32_activation to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for fbgemm_linear_fp16_weight_fp32_activation.
    """
    from onnx9000.core.ops.torch_auto import (
        fbgemm_linear_fp16_weight_fp32_activation as core_fbgemm_linear_fp16_weight_fp32_activation,
    )

    return core_fbgemm_linear_fp16_weight_fp32_activation(input_tensor, *args, **kwargs)


def fbgemm_linear_int8_weight(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply fbgemm_linear_int8_weight to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for fbgemm_linear_int8_weight.
    """
    from onnx9000.core.ops.torch_auto import (
        fbgemm_linear_int8_weight as core_fbgemm_linear_int8_weight,
    )

    return core_fbgemm_linear_int8_weight(input_tensor, *args, **kwargs)


def fbgemm_linear_int8_weight_fp32_activation(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply fbgemm_linear_int8_weight_fp32_activation to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for fbgemm_linear_int8_weight_fp32_activation.
    """
    from onnx9000.core.ops.torch_auto import (
        fbgemm_linear_int8_weight_fp32_activation as core_fbgemm_linear_int8_weight_fp32_activation,
    )

    return core_fbgemm_linear_int8_weight_fp32_activation(input_tensor, *args, **kwargs)


def fbgemm_linear_quantize_weight(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply fbgemm_linear_quantize_weight to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for fbgemm_linear_quantize_weight.
    """
    from onnx9000.core.ops.torch_auto import (
        fbgemm_linear_quantize_weight as core_fbgemm_linear_quantize_weight,
    )

    return core_fbgemm_linear_quantize_weight(input_tensor, *args, **kwargs)


def fbgemm_pack_gemm_matrix_fp16(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply fbgemm_pack_gemm_matrix_fp16 to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for fbgemm_pack_gemm_matrix_fp16.
    """
    from onnx9000.core.ops.torch_auto import (
        fbgemm_pack_gemm_matrix_fp16 as core_fbgemm_pack_gemm_matrix_fp16,
    )

    return core_fbgemm_pack_gemm_matrix_fp16(input_tensor, *args, **kwargs)


def fbgemm_pack_quantized_matrix(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply fbgemm_pack_quantized_matrix to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for fbgemm_pack_quantized_matrix.
    """
    from onnx9000.core.ops.torch_auto import (
        fbgemm_pack_quantized_matrix as core_fbgemm_pack_quantized_matrix,
    )

    return core_fbgemm_pack_quantized_matrix(input_tensor, *args, **kwargs)


def feature_alpha_dropout(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply feature_alpha_dropout to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for feature_alpha_dropout.
    """
    from onnx9000.core.ops.torch_auto import feature_alpha_dropout as core_feature_alpha_dropout

    return core_feature_alpha_dropout(input_tensor, *args, **kwargs)


def feature_alpha_dropout_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply feature_alpha_dropout_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for feature_alpha_dropout_.
    """
    from onnx9000.core.ops.torch_auto import feature_alpha_dropout_ as core_feature_alpha_dropout_

    return core_feature_alpha_dropout_(input_tensor, *args, **kwargs)


def feature_dropout(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply feature_dropout to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for feature_dropout.
    """
    from onnx9000.core.ops.torch_auto import feature_dropout as core_feature_dropout

    return core_feature_dropout(input_tensor, *args, **kwargs)


def feature_dropout_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply feature_dropout_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for feature_dropout_.
    """
    from onnx9000.core.ops.torch_auto import feature_dropout_ as core_feature_dropout_

    return core_feature_dropout_(input_tensor, *args, **kwargs)


def fused_moving_avg_obs_fake_quant(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply fused_moving_avg_obs_fake_quant to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for fused_moving_avg_obs_fake_quant.
    """
    from onnx9000.core.ops.torch_auto import (
        fused_moving_avg_obs_fake_quant as core_fused_moving_avg_obs_fake_quant,
    )

    return core_fused_moving_avg_obs_fake_quant(input_tensor, *args, **kwargs)


def grid_sampler(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply grid_sampler to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for grid_sampler.
    """
    from onnx9000.core.ops.torch_auto import grid_sampler as core_grid_sampler

    return core_grid_sampler(input_tensor, *args, **kwargs)


def grid_sampler_2d(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply grid_sampler_2d to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for grid_sampler_2d.
    """
    from onnx9000.core.ops.torch_auto import grid_sampler_2d as core_grid_sampler_2d

    return core_grid_sampler_2d(input_tensor, *args, **kwargs)


def grid_sampler_3d(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply grid_sampler_3d to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for grid_sampler_3d.
    """
    from onnx9000.core.ops.torch_auto import grid_sampler_3d as core_grid_sampler_3d

    return core_grid_sampler_3d(input_tensor, *args, **kwargs)


def group_norm(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply group_norm to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for group_norm.
    """
    from onnx9000.core.ops.torch_auto import group_norm as core_group_norm

    return core_group_norm(input_tensor, *args, **kwargs)


def gru_cell(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply gru_cell to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for gru_cell.
    """
    from onnx9000.core.ops.torch_auto import gru_cell as core_gru_cell

    return core_gru_cell(input_tensor, *args, **kwargs)


def hardshrink(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply hardshrink to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for hardshrink.
    """
    from onnx9000.core.ops.torch_auto import hardshrink as core_hardshrink

    return core_hardshrink(input_tensor, *args, **kwargs)


def hinge_embedding_loss(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply hinge_embedding_loss to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for hinge_embedding_loss.
    """
    from onnx9000.core.ops.torch_auto import hinge_embedding_loss as core_hinge_embedding_loss

    return core_hinge_embedding_loss(input_tensor, *args, **kwargs)


def instance_norm(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply instance_norm to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for instance_norm.
    """
    from onnx9000.core.ops.torch_auto import instance_norm as core_instance_norm

    return core_instance_norm(input_tensor, *args, **kwargs)


def int_repr(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply int_repr to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for int_repr.
    """
    from onnx9000.core.ops.torch_auto import int_repr as core_int_repr

    return core_int_repr(input_tensor, *args, **kwargs)


def kl_div(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply kl_div to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for kl_div.
    """
    from onnx9000.core.ops.torch_auto import kl_div as core_kl_div

    return core_kl_div(input_tensor, *args, **kwargs)


def layer_norm(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply layer_norm to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for layer_norm.
    """
    from onnx9000.core.ops.torch_auto import layer_norm as core_layer_norm

    return core_layer_norm(input_tensor, *args, **kwargs)


def lstm_cell(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply lstm_cell to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for lstm_cell.
    """
    from onnx9000.core.ops.torch_auto import lstm_cell as core_lstm_cell

    return core_lstm_cell(input_tensor, *args, **kwargs)


def margin_ranking_loss(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply margin_ranking_loss to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for margin_ranking_loss.
    """
    from onnx9000.core.ops.torch_auto import margin_ranking_loss as core_margin_ranking_loss

    return core_margin_ranking_loss(input_tensor, *args, **kwargs)


def max_pool1d(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply max_pool1d to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for max_pool1d.
    """
    from onnx9000.core.ops.torch_auto import max_pool1d as core_max_pool1d

    return core_max_pool1d(input_tensor, *args, **kwargs)


def max_pool1d_with_indices(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply max_pool1d_with_indices to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for max_pool1d_with_indices.
    """
    from onnx9000.core.ops.torch_auto import max_pool1d_with_indices as core_max_pool1d_with_indices

    return core_max_pool1d_with_indices(input_tensor, *args, **kwargs)


def max_pool2d(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply max_pool2d to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for max_pool2d.
    """
    from onnx9000.core.ops.torch_auto import max_pool2d as core_max_pool2d

    return core_max_pool2d(input_tensor, *args, **kwargs)


def max_pool3d(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply max_pool3d to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for max_pool3d.
    """
    from onnx9000.core.ops.torch_auto import max_pool3d as core_max_pool3d

    return core_max_pool3d(input_tensor, *args, **kwargs)


def miopen_batch_norm(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply miopen_batch_norm to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for miopen_batch_norm.
    """
    from onnx9000.core.ops.torch_auto import miopen_batch_norm as core_miopen_batch_norm

    return core_miopen_batch_norm(input_tensor, *args, **kwargs)


def miopen_convolution(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply miopen_convolution to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for miopen_convolution.
    """
    from onnx9000.core.ops.torch_auto import miopen_convolution as core_miopen_convolution

    return core_miopen_convolution(input_tensor, *args, **kwargs)


def miopen_convolution_add_relu(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply miopen_convolution_add_relu to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for miopen_convolution_add_relu.
    """
    from onnx9000.core.ops.torch_auto import (
        miopen_convolution_add_relu as core_miopen_convolution_add_relu,
    )

    return core_miopen_convolution_add_relu(input_tensor, *args, **kwargs)


def miopen_convolution_relu(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply miopen_convolution_relu to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for miopen_convolution_relu.
    """
    from onnx9000.core.ops.torch_auto import miopen_convolution_relu as core_miopen_convolution_relu

    return core_miopen_convolution_relu(input_tensor, *args, **kwargs)


def miopen_convolution_transpose(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply miopen_convolution_transpose to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for miopen_convolution_transpose.
    """
    from onnx9000.core.ops.torch_auto import (
        miopen_convolution_transpose as core_miopen_convolution_transpose,
    )

    return core_miopen_convolution_transpose(input_tensor, *args, **kwargs)


def miopen_ctc_loss(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply miopen_ctc_loss to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for miopen_ctc_loss.
    """
    from onnx9000.core.ops.torch_auto import miopen_ctc_loss as core_miopen_ctc_loss

    return core_miopen_ctc_loss(input_tensor, *args, **kwargs)


def miopen_depthwise_convolution(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply miopen_depthwise_convolution to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for miopen_depthwise_convolution.
    """
    from onnx9000.core.ops.torch_auto import (
        miopen_depthwise_convolution as core_miopen_depthwise_convolution,
    )

    return core_miopen_depthwise_convolution(input_tensor, *args, **kwargs)


def miopen_rnn(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply miopen_rnn to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for miopen_rnn.
    """
    from onnx9000.core.ops.torch_auto import miopen_rnn as core_miopen_rnn

    return core_miopen_rnn(input_tensor, *args, **kwargs)


def mkldnn_adaptive_avg_pool2d(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply mkldnn_adaptive_avg_pool2d to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for mkldnn_adaptive_avg_pool2d.
    """
    from onnx9000.core.ops.torch_auto import (
        mkldnn_adaptive_avg_pool2d as core_mkldnn_adaptive_avg_pool2d,
    )

    return core_mkldnn_adaptive_avg_pool2d(input_tensor, *args, **kwargs)


def mkldnn_convolution(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply mkldnn_convolution to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for mkldnn_convolution.
    """
    from onnx9000.core.ops.torch_auto import mkldnn_convolution as core_mkldnn_convolution

    return core_mkldnn_convolution(input_tensor, *args, **kwargs)


def mkldnn_linear_backward_weights(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply mkldnn_linear_backward_weights to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for mkldnn_linear_backward_weights.
    """
    from onnx9000.core.ops.torch_auto import (
        mkldnn_linear_backward_weights as core_mkldnn_linear_backward_weights,
    )

    return core_mkldnn_linear_backward_weights(input_tensor, *args, **kwargs)


def mkldnn_max_pool2d(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply mkldnn_max_pool2d to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for mkldnn_max_pool2d.
    """
    from onnx9000.core.ops.torch_auto import mkldnn_max_pool2d as core_mkldnn_max_pool2d

    return core_mkldnn_max_pool2d(input_tensor, *args, **kwargs)


def mkldnn_max_pool3d(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply mkldnn_max_pool3d to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for mkldnn_max_pool3d.
    """
    from onnx9000.core.ops.torch_auto import mkldnn_max_pool3d as core_mkldnn_max_pool3d

    return core_mkldnn_max_pool3d(input_tensor, *args, **kwargs)


def mkldnn_rnn_layer(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply mkldnn_rnn_layer to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for mkldnn_rnn_layer.
    """
    from onnx9000.core.ops.torch_auto import mkldnn_rnn_layer as core_mkldnn_rnn_layer

    return core_mkldnn_rnn_layer(input_tensor, *args, **kwargs)


def native_batch_norm(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply native_batch_norm to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for native_batch_norm.
    """
    from onnx9000.core.ops.torch_auto import native_batch_norm as core_native_batch_norm

    return core_native_batch_norm(input_tensor, *args, **kwargs)


def native_channel_shuffle(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply native_channel_shuffle to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for native_channel_shuffle.
    """
    from onnx9000.core.ops.torch_auto import native_channel_shuffle as core_native_channel_shuffle

    return core_native_channel_shuffle(input_tensor, *args, **kwargs)


def native_dropout(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply native_dropout to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for native_dropout.
    """
    from onnx9000.core.ops.torch_auto import native_dropout as core_native_dropout

    return core_native_dropout(input_tensor, *args, **kwargs)


def native_group_norm(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply native_group_norm to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for native_group_norm.
    """
    from onnx9000.core.ops.torch_auto import native_group_norm as core_native_group_norm

    return core_native_group_norm(input_tensor, *args, **kwargs)


def native_layer_norm(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply native_layer_norm to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for native_layer_norm.
    """
    from onnx9000.core.ops.torch_auto import native_layer_norm as core_native_layer_norm

    return core_native_layer_norm(input_tensor, *args, **kwargs)


def native_norm(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply native_norm to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for native_norm.
    """
    from onnx9000.core.ops.torch_auto import native_norm as core_native_norm

    return core_native_norm(input_tensor, *args, **kwargs)


def pairwise_distance(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply pairwise_distance to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for pairwise_distance.
    """
    from onnx9000.core.ops.torch_auto import pairwise_distance as core_pairwise_distance

    return core_pairwise_distance(input_tensor, *args, **kwargs)


def pdist(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply pdist to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for pdist.
    """
    from onnx9000.core.ops.torch_auto import pdist as core_pdist

    return core_pdist(input_tensor, *args, **kwargs)


def pixel_shuffle(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply pixel_shuffle to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for pixel_shuffle.
    """
    from onnx9000.core.ops.torch_auto import pixel_shuffle as core_pixel_shuffle

    return core_pixel_shuffle(input_tensor, *args, **kwargs)


def pixel_unshuffle(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply pixel_unshuffle to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for pixel_unshuffle.
    """
    from onnx9000.core.ops.torch_auto import pixel_unshuffle as core_pixel_unshuffle

    return core_pixel_unshuffle(input_tensor, *args, **kwargs)


def poisson_nll_loss(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply poisson_nll_loss to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for poisson_nll_loss.
    """
    from onnx9000.core.ops.torch_auto import poisson_nll_loss as core_poisson_nll_loss

    return core_poisson_nll_loss(input_tensor, *args, **kwargs)


def q_per_channel_axis(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply q_per_channel_axis to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for q_per_channel_axis.
    """
    from onnx9000.core.ops.torch_auto import q_per_channel_axis as core_q_per_channel_axis

    return core_q_per_channel_axis(input_tensor, *args, **kwargs)


def q_per_channel_scales(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply q_per_channel_scales to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for q_per_channel_scales.
    """
    from onnx9000.core.ops.torch_auto import q_per_channel_scales as core_q_per_channel_scales

    return core_q_per_channel_scales(input_tensor, *args, **kwargs)


def q_per_channel_zero_points(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply q_per_channel_zero_points to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for q_per_channel_zero_points.
    """
    from onnx9000.core.ops.torch_auto import (
        q_per_channel_zero_points as core_q_per_channel_zero_points,
    )

    return core_q_per_channel_zero_points(input_tensor, *args, **kwargs)


def q_scale(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply q_scale to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for q_scale.
    """
    from onnx9000.core.ops.torch_auto import q_scale as core_q_scale

    return core_q_scale(input_tensor, *args, **kwargs)


def q_zero_point(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply q_zero_point to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for q_zero_point.
    """
    from onnx9000.core.ops.torch_auto import q_zero_point as core_q_zero_point

    return core_q_zero_point(input_tensor, *args, **kwargs)


def quantize_per_channel(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply quantize_per_channel to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for quantize_per_channel.
    """
    from onnx9000.core.ops.torch_auto import quantize_per_channel as core_quantize_per_channel

    return core_quantize_per_channel(input_tensor, *args, **kwargs)


def quantize_per_tensor(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply quantize_per_tensor to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for quantize_per_tensor.
    """
    from onnx9000.core.ops.torch_auto import quantize_per_tensor as core_quantize_per_tensor

    return core_quantize_per_tensor(input_tensor, *args, **kwargs)


def quantize_per_tensor_dynamic(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply quantize_per_tensor_dynamic to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for quantize_per_tensor_dynamic.
    """
    from onnx9000.core.ops.torch_auto import (
        quantize_per_tensor_dynamic as core_quantize_per_tensor_dynamic,
    )

    return core_quantize_per_tensor_dynamic(input_tensor, *args, **kwargs)


def quantized_batch_norm(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply quantized_batch_norm to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for quantized_batch_norm.
    """
    from onnx9000.core.ops.torch_auto import quantized_batch_norm as core_quantized_batch_norm

    return core_quantized_batch_norm(input_tensor, *args, **kwargs)


def quantized_gru_cell(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply quantized_gru_cell to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for quantized_gru_cell.
    """
    from onnx9000.core.ops.torch_auto import quantized_gru_cell as core_quantized_gru_cell

    return core_quantized_gru_cell(input_tensor, *args, **kwargs)


def quantized_lstm_cell(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply quantized_lstm_cell to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for quantized_lstm_cell.
    """
    from onnx9000.core.ops.torch_auto import quantized_lstm_cell as core_quantized_lstm_cell

    return core_quantized_lstm_cell(input_tensor, *args, **kwargs)


def quantized_max_pool1d(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply quantized_max_pool1d to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for quantized_max_pool1d.
    """
    from onnx9000.core.ops.torch_auto import quantized_max_pool1d as core_quantized_max_pool1d

    return core_quantized_max_pool1d(input_tensor, *args, **kwargs)


def quantized_max_pool2d(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply quantized_max_pool2d to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for quantized_max_pool2d.
    """
    from onnx9000.core.ops.torch_auto import quantized_max_pool2d as core_quantized_max_pool2d

    return core_quantized_max_pool2d(input_tensor, *args, **kwargs)


def quantized_max_pool3d(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply quantized_max_pool3d to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for quantized_max_pool3d.
    """
    from onnx9000.core.ops.torch_auto import quantized_max_pool3d as core_quantized_max_pool3d

    return core_quantized_max_pool3d(input_tensor, *args, **kwargs)


def quantized_rnn_relu_cell(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply quantized_rnn_relu_cell to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for quantized_rnn_relu_cell.
    """
    from onnx9000.core.ops.torch_auto import quantized_rnn_relu_cell as core_quantized_rnn_relu_cell

    return core_quantized_rnn_relu_cell(input_tensor, *args, **kwargs)


def quantized_rnn_tanh_cell(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply quantized_rnn_tanh_cell to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for quantized_rnn_tanh_cell.
    """
    from onnx9000.core.ops.torch_auto import quantized_rnn_tanh_cell as core_quantized_rnn_tanh_cell

    return core_quantized_rnn_tanh_cell(input_tensor, *args, **kwargs)


def relu_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply relu_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for relu_.
    """
    from onnx9000.core.ops.torch_auto import relu_ as core_relu_

    return core_relu_(input_tensor, *args, **kwargs)


def rms_norm(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply rms_norm to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for rms_norm.
    """
    from onnx9000.core.ops.torch_auto import rms_norm as core_rms_norm

    return core_rms_norm(input_tensor, *args, **kwargs)


def rnn_relu(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply rnn_relu to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for rnn_relu.
    """
    from onnx9000.core.ops.torch_auto import rnn_relu as core_rnn_relu

    return core_rnn_relu(input_tensor, *args, **kwargs)


def rnn_relu_cell(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply rnn_relu_cell to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for rnn_relu_cell.
    """
    from onnx9000.core.ops.torch_auto import rnn_relu_cell as core_rnn_relu_cell

    return core_rnn_relu_cell(input_tensor, *args, **kwargs)


def rnn_tanh(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply rnn_tanh to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for rnn_tanh.
    """
    from onnx9000.core.ops.torch_auto import rnn_tanh as core_rnn_tanh

    return core_rnn_tanh(input_tensor, *args, **kwargs)


def rnn_tanh_cell(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply rnn_tanh_cell to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for rnn_tanh_cell.
    """
    from onnx9000.core.ops.torch_auto import rnn_tanh_cell as core_rnn_tanh_cell

    return core_rnn_tanh_cell(input_tensor, *args, **kwargs)


def rrelu(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply rrelu to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for rrelu.
    """
    from onnx9000.core.ops.torch_auto import rrelu as core_rrelu

    return core_rrelu(input_tensor, *args, **kwargs)


def rrelu_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply rrelu_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for rrelu_.
    """
    from onnx9000.core.ops.torch_auto import rrelu_ as core_rrelu_

    return core_rrelu_(input_tensor, *args, **kwargs)


def selu_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply selu_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for selu_.
    """
    from onnx9000.core.ops.torch_auto import selu_ as core_selu_

    return core_selu_(input_tensor, *args, **kwargs)


def sigmoid_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply sigmoid_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for sigmoid_.
    """
    from onnx9000.core.ops.torch_auto import sigmoid_ as core_sigmoid_

    return core_sigmoid_(input_tensor, *args, **kwargs)


def tanh_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply tanh_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for tanh_.
    """
    from onnx9000.core.ops.torch_auto import tanh_ as core_tanh_

    return core_tanh_(input_tensor, *args, **kwargs)


def threshold(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply threshold to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for threshold.
    """
    from onnx9000.core.ops.torch_auto import threshold as core_threshold

    return core_threshold(input_tensor, *args, **kwargs)


def threshold_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply threshold_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for threshold_.
    """
    from onnx9000.core.ops.torch_auto import threshold_ as core_threshold_

    return core_threshold_(input_tensor, *args, **kwargs)


def triplet_margin_loss(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply triplet_margin_loss to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for triplet_margin_loss.
    """
    from onnx9000.core.ops.torch_auto import triplet_margin_loss as core_triplet_margin_loss

    return core_triplet_margin_loss(input_tensor, *args, **kwargs)
