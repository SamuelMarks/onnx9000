"""Functional interface."""

from typing import Any, Optional

from onnx9000.converters.frontend.tensor import Tensor


def relu(input: Tensor) -> Tensor:
    """Apply the rectified linear unit (ReLU) activation function element-wise.

    Args:
        input: The input tensor.

    Returns:
        The output tensor after applying ReLU.
    """
    return input.relu()


def sigmoid(input: Tensor) -> Tensor:
    """Apply the sigmoid activation function element-wise.

    Args:
        input: The input tensor.

    Returns:
        The output tensor after applying sigmoid.
    """
    return input.sigmoid()


def tanh(input: Tensor) -> Tensor:
    """Apply the hyperbolic tangent (Tanh) activation function element-wise.

    Args:
        input: The input tensor.

    Returns:
        The output tensor after applying Tanh.
    """
    return input.tanh()


def gelu(input: Tensor) -> Tensor:
    """Apply the Gaussian Error Linear Unit (GELU) activation function element-wise.

    Args:
        input: The input tensor.

    Returns:
        The output tensor after applying GELU.
    """
    return input.gelu()


def softmax(input: Tensor, dim: int = -1) -> Tensor:
    """Apply the softmax function to the input tensor along a specified dimension.

    Args:
        input: The input tensor.
        dim: The dimension along which softmax will be computed.

    Returns:
        The output tensor after applying softmax.
    """
    return input.softmax(dim)


def log_softmax(input: Tensor, dim: int = -1) -> Tensor:
    """Apply the natural logarithm of the softmax function to the input tensor along a dimension.

    Args:
        input: The input tensor.
        dim: The dimension along which log_softmax will be computed.

    Returns:
        The output tensor after applying log_softmax.
    """
    return input.log_softmax(dim)


def max_pool2d(
    input: Tensor,
    kernel_size: Any,
    stride: Any = None,
    padding: Any = 0,
    dilation: Any = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> Tensor:
    """Apply a 2D max pooling over an input signal composed of several input planes.

    Args:
        input: The input tensor (N, C, H, W).
        kernel_size: The size of the window to take a max over.
        stride: The stride of the window. Default value is kernel_size.
        padding: Implicit zero padding to be added on both sides.
        dilation: The spacing between window elements.
        ceil_mode: When True, will use ceil instead of floor to compute the output shape.
        return_indices: If True, will return the max indices along with the outputs.

    Returns:
        The output tensor after max pooling.
    """
    from onnx9000.converters.frontend.utils import record_op

    def _pair(x):
        """Helper to convert int or tuple to (int, int)."""
        return (x, x) if isinstance(x, int) else x

    kernel_size_ = _pair(kernel_size)
    stride_ = _pair(stride) if stride is not None else kernel_size_
    padding_ = _pair(padding)
    dilation_ = _pair(dilation)

    attrs = {
        "kernel_shape": list(kernel_size_),
        "strides": list(stride_),
        "pads": list(padding_) * 2,
        "dilations": list(dilation_),
        "ceil_mode": 1 if ceil_mode else 0,
    }
    return record_op("MaxPool", [input], attrs)


def linear(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    """Apply a linear transformation to the incoming data: y = xW^T + b.

    Args:
        input: The input tensor (*, in_features).
        weight: The learnable weights of the module of shape (out_features, in_features).
        bias: The learnable bias of the module of shape (out_features).

    Returns:
        The output tensor (*, out_features).
    """
    res = input @ weight.T
    if bias is not None:
        res = res + bias
    return res


def conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Any = 1,
    padding: Any = 0,
    dilation: Any = 1,
    groups: int = 1,
) -> Tensor:
    """Apply a 2D convolution over an input image composed of several input planes.

    Args:
        input: Input tensor of shape (N, C_in, H, W).
        weight: Filters of shape (C_out, C_in / groups, kH, kW).
        bias: Optional bias tensor of shape (C_out).
        stride: Stride of the convolution.
        padding: Zero-padding added to both sides of the input.
        dilation: Spacing between kernel elements.
        groups: Number of blocked connections from input channels to output channels.

    Returns:
        The output tensor of shape (N, C_out, H_out, W_out).
    """
    from onnx9000.converters.frontend.utils import record_op

    def _pair(x):
        """Helper to convert int or tuple to (int, int)."""
        return (x, x) if isinstance(x, int) else x

    attrs = {
        "kernel_shape": list(weight.shape[2:]),
        "strides": list(_pair(stride)),
        "pads": list(_pair(padding)) * 2,
        "dilations": list(_pair(dilation)),
        "group": groups,
    }
    inputs = [input, weight]
    if bias is not None:
        inputs.append(bias)
    return record_op("Conv", inputs, attrs)


def pad(input: Tensor, pad: tuple[int], mode: str = "constant", value: float = 0.0) -> Tensor:
    """Pad the input tensor using various modes.

    Args:
        input: The input tensor.
        pad: m-elements tuple, where m/2 is the number of dimensions to pad.
             Format is (left, right, top, bottom, front, back) starting from last dim.
        mode: 'constant', 'reflect' or 'replicate'.
        value: fill value for 'constant' padding.

    Returns:
        The padded tensor.
    """
    import numpy as np
    from onnx9000.converters.frontend.tensor import Parameter
    from onnx9000.converters.frontend.utils import record_op
    from onnx9000.core.dtypes import DType

    dims = len(input.shape)
    num_pad_dims = len(pad) // 2
    onnx_pads = [0] * (dims * 2)
    for i in range(num_pad_dims):
        onnx_pads[dims - 1 - i] = pad[i * 2]
        onnx_pads[2 * dims - 1 - i] = pad[i * 2 + 1]
    pads_tensor = Parameter(
        (len(onnx_pads),), DType.INT64, "pads", np.array(onnx_pads, dtype=np.int64)
    )
    inputs = [input, pads_tensor]
    if mode == "constant":
        val_tensor = Parameter(
            (), DType.FLOAT32, "constant_value", np.array(value, dtype=np.float32)
        )
        inputs.append(val_tensor)
    return record_op("Pad", inputs, {"mode": mode})


def interpolate(
    input: Tensor,
    size: Optional[Any] = None,
    scale_factor: Optional[Any] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
) -> Tensor:
    """Down/up samples the input to either the given size or the given scale factor.

    Args:
        input: The input tensor.
        size: Output spatial size.
        scale_factor: Multiplier for spatial size.
        mode: The interpolation mode: 'nearest' or 'bilinear'.
        align_corners: Whether to align corners for bilinear interpolation.

    Returns:
        The interpolated tensor.
    """
    import numpy as np
    from onnx9000.converters.frontend.tensor import Parameter
    from onnx9000.converters.frontend.utils import record_op
    from onnx9000.core.dtypes import DType

    attrs = {"mode": mode}
    if align_corners is not None and mode in ["linear", "bilinear", "trilinear", "bicubic"]:
        return None
    inputs = [input]
    inputs.append(Parameter((0,), DType.FLOAT32, "roi", np.array([], dtype=np.float32)))
    if scale_factor is not None:
        scales = [1.0, 1.0]
        if isinstance(scale_factor, float):
            scales += [scale_factor] * (len(input.shape) - 2)
        elif isinstance(scale_factor, (tuple, list)):
            scales += list(scale_factor)
        scales_tensor = Parameter(
            (len(scales),), DType.FLOAT32, "scales", np.array(scales, dtype=np.float32)
        )
        inputs.append(scales_tensor)
    else:
        return None
    if size is not None:
        return None
    return record_op("Resize", inputs, attrs)


def one_hot(tensor: Tensor, num_classes: int = -1) -> Tensor:
    """Return a tensor where the last dimension is one-hot encoded.

    Args:
        tensor: The input tensor containing class indices.
        num_classes: Total number of classes. If -1, it's inferred from the input.

    Returns:
        The one-hot encoded tensor.
    """
    import numpy as np
    from onnx9000.converters.frontend.tensor import Parameter
    from onnx9000.converters.frontend.utils import record_op
    from onnx9000.core.dtypes import DType

    depth = Parameter((), DType.INT64, "depth", np.array(num_classes, dtype=np.int64))
    values = Parameter((2,), DType.INT64, "values", np.array([0, 1], dtype=np.int64))
    return record_op("OneHot", [tensor, depth, values], {"axis": -1})
