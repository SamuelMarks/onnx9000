"""Convolution layers."""

from typing import Any, Optional, Union
from onnx9000.core.dtypes import DType
from onnx9000.frontend.frontend.nn.module import Module
from onnx9000.frontend.frontend.tensor import Parameter, Tensor


def _pair(x: Union[int, tuple[int, int]]) -> tuple[int, int]:
    """Implements the _pair method."""
    if isinstance(x, int):
        return (x, x)
    return x


def _single(x: Union[int, tuple[int]]) -> tuple[int]:
    """Implements the _single method."""
    if isinstance(x, int):
        return (x,)
    return x


def _triple(x: Union[int, tuple[int, int, int]]) -> tuple[int, int, int]:
    """Implements the _triple method."""
    if isinstance(x, int):
        return (x, x, x)
    return x


class _ConvNd(Module):
    """Class _ConvNd implementation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Any,
        stride: Any,
        padding: Any,
        dilation: Any,
        groups: int,
        bias: bool,
        padding_mode: str,
        dtype: DType,
    ) -> None:
        """Implements the __init__ method."""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.weight: Optional[Parameter] = None
        self.bias: Optional[Parameter] = None

    def forward(self, input: Tensor) -> Tensor:
        """Implements the forward method."""
        from onnx9000.frontend.frontend.utils import record_op

        attrs = {
            "kernel_shape": list(self.kernel_size),
            "strides": list(self.stride),
            "pads": list(self.padding) * 2,
            "dilations": list(self.dilation),
            "group": self.groups,
        }
        inputs = [input, self.weight]
        if self.bias is not None:
            inputs.append(self.bias)
        return record_op("Conv", inputs, attrs)


class Conv1d(_ConvNd):
    """Conv1d layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple[int]],
        stride: Union[int, tuple[int]] = 1,
        padding: Union[int, tuple[int]] = 0,
        dilation: Union[int, tuple[int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        dtype: DType = DType.FLOAT32,
    ) -> None:
        """Implements the __init__ method."""
        kernel_size_ = _single(kernel_size)
        stride_ = _single(stride)
        padding_ = _single(padding)
        dilation_ = _single(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            groups,
            bias,
            padding_mode,
            dtype,
        )
        self.weight = Parameter(
            (out_channels, in_channels // groups, *kernel_size_), dtype, "weight"
        )
        if bias:
            self.bias = Parameter((out_channels,), dtype, "bias")
        else:
            self.register_parameter("bias", None)


class Conv2d(_ConvNd):
    """Conv2d layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple[int, int]],
        stride: Union[int, tuple[int, int]] = 1,
        padding: Union[int, tuple[int, int]] = 0,
        dilation: Union[int, tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        dtype: DType = DType.FLOAT32,
    ) -> None:
        """Implements the __init__ method."""
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            groups,
            bias,
            padding_mode,
            dtype,
        )
        self.weight = Parameter(
            (out_channels, in_channels // groups, *kernel_size_), dtype, "weight"
        )
        if bias:
            self.bias = Parameter((out_channels,), dtype, "bias")
        else:
            self.register_parameter("bias", None)


class Conv3d(_ConvNd):
    """Conv3d layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple[int, int, int]],
        stride: Union[int, tuple[int, int, int]] = 1,
        padding: Union[int, tuple[int, int, int]] = 0,
        dilation: Union[int, tuple[int, int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        dtype: DType = DType.FLOAT32,
    ) -> None:
        """Implements the __init__ method."""
        kernel_size_ = _triple(kernel_size)
        stride_ = _triple(stride)
        padding_ = _triple(padding)
        dilation_ = _triple(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            groups,
            bias,
            padding_mode,
            dtype,
        )
        self.weight = Parameter(
            (out_channels, in_channels // groups, *kernel_size_), dtype, "weight"
        )
        if bias:
            self.bias = Parameter((out_channels,), dtype, "bias")
        else:
            self.register_parameter("bias", None)


class _ConvTransposeNd(Module):
    """Class _ConvTransposeNd implementation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Any,
        stride: Any,
        padding: Any,
        output_padding: Any,
        groups: int,
        bias: bool,
        dilation: Any,
        padding_mode: str,
        dtype: DType,
    ) -> None:
        """Implements the __init__ method."""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.dilation = dilation
        self.padding_mode = padding_mode
        self.weight: Optional[Parameter] = None
        self.bias: Optional[Parameter] = None

    def forward(self, input: Tensor) -> Tensor:
        """Implements the forward method."""
        from onnx9000.frontend.frontend.utils import record_op

        attrs = {
            "kernel_shape": list(self.kernel_size),
            "strides": list(self.stride),
            "pads": list(self.padding) * 2,
            "dilations": list(self.dilation),
            "group": self.groups,
        }
        if sum(self.output_padding) > 0:
            attrs["output_padding"] = list(self.output_padding)
        inputs = [input, self.weight]
        if self.bias is not None:
            inputs.append(self.bias)
        return record_op("ConvTranspose", inputs, attrs)


class ConvTranspose1d(_ConvTransposeNd):
    """ConvTranspose1d layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple[int]],
        stride: Union[int, tuple[int]] = 1,
        padding: Union[int, tuple[int]] = 0,
        output_padding: Union[int, tuple[int]] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: Union[int, tuple[int]] = 1,
        padding_mode: str = "zeros",
        dtype: DType = DType.FLOAT32,
    ) -> None:
        """Implements the __init__ method."""
        kernel_size_ = _single(kernel_size)
        stride_ = _single(stride)
        padding_ = _single(padding)
        output_padding_ = _single(output_padding)
        dilation_ = _single(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            output_padding_,
            groups,
            bias,
            dilation_,
            padding_mode,
            dtype,
        )
        self.weight = Parameter(
            (in_channels, out_channels // groups, *kernel_size_), dtype, "weight"
        )
        if bias:
            self.bias = Parameter((out_channels,), dtype, "bias")
        else:
            self.register_parameter("bias", None)


class ConvTranspose2d(_ConvTransposeNd):
    """ConvTranspose2d layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple[int, int]],
        stride: Union[int, tuple[int, int]] = 1,
        padding: Union[int, tuple[int, int]] = 0,
        output_padding: Union[int, tuple[int, int]] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: Union[int, tuple[int, int]] = 1,
        padding_mode: str = "zeros",
        dtype: DType = DType.FLOAT32,
    ) -> None:
        """Implements the __init__ method."""
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = _pair(padding)
        output_padding_ = _pair(output_padding)
        dilation_ = _pair(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            output_padding_,
            groups,
            bias,
            dilation_,
            padding_mode,
            dtype,
        )
        self.weight = Parameter(
            (in_channels, out_channels // groups, *kernel_size_), dtype, "weight"
        )
        if bias:
            self.bias = Parameter((out_channels,), dtype, "bias")
        else:
            self.register_parameter("bias", None)
