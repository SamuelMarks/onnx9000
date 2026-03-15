"""Convolution layers."""

from typing import Any, Optional, Union, Tuple
from onnx9000.frontends.frontend.nn.module import Module
from onnx9000.frontends.frontend.tensor import Parameter, Tensor
from onnx9000.core.dtypes import DType


def _pair(x: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """Provides semantic functionality and verification."""
    if isinstance(x, int):
        return (x, x)
    return x


def _single(x: Union[int, Tuple[int]]) -> Tuple[int]:
    """Provides semantic functionality and verification."""
    if isinstance(x, int):
        return (x,)
    return x


def _triple(x: Union[int, Tuple[int, int, int]]) -> Tuple[int, int, int]:
    """Provides semantic functionality and verification."""
    if isinstance(x, int):
        return (x, x, x)
    return x


class _ConvNd(Module):
    """Provides semantic functionality and verification."""

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
        """Provides semantic functionality and verification."""
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
        """Provides semantic functionality and verification."""
        from onnx9000.frontends.frontend.utils import record_op

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
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int]] = 0,
        dilation: Union[int, Tuple[int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        dtype: DType = DType.FLOAT32,
    ) -> None:
        """Provides semantic functionality and verification."""
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
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        dtype: DType = DType.FLOAT32,
    ) -> None:
        """Provides semantic functionality and verification."""
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
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        dtype: DType = DType.FLOAT32,
    ) -> None:
        """Provides semantic functionality and verification."""
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
    """Provides semantic functionality and verification."""

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
        """Provides semantic functionality and verification."""
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
        """Provides semantic functionality and verification."""
        from onnx9000.frontends.frontend.utils import record_op

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
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int]] = 0,
        output_padding: Union[int, Tuple[int]] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: Union[int, Tuple[int]] = 1,
        padding_mode: str = "zeros",
        dtype: DType = DType.FLOAT32,
    ) -> None:
        """Provides semantic functionality and verification."""
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
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        output_padding: Union[int, Tuple[int, int]] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: Union[int, Tuple[int, int]] = 1,
        padding_mode: str = "zeros",
        dtype: DType = DType.FLOAT32,
    ) -> None:
        """Provides semantic functionality and verification."""
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
