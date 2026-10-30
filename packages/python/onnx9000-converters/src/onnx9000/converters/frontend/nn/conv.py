"""Convolution layers."""

from onnx9000.converters.frontend.nn.module import Module
from onnx9000.converters.frontend.tensor import Parameter, Tensor
from onnx9000.core.dtypes import DType


def _pair(x: int | tuple[int, int]) -> tuple[int, int]:
    """Execute pair operation.

    Args:
        x: Parameter specifying x (int | tuple[int, int]).

    Returns:
        Result of pair (tuple[int, int]).
    """
    if isinstance(x, int):
        return (x, x)
    return x


def _single(x: int | tuple[int]) -> tuple[int]:
    """Execute single operation.

    Args:
        x: Parameter specifying x (int | tuple[int]).

    Returns:
        Result of single (tuple[int]).
    """
    if isinstance(x, int):
        return (x,)
    return x


def _triple(x: int | tuple[int, int, int]) -> tuple[int, int, int]:
    """Execute triple operation.

    Args:
        x: Parameter specifying x (int | tuple[int, int, int]).

    Returns:
        Result of triple (tuple[int, int, int]).
    """
    if isinstance(x, int):
        return (x, x, x)
    return x


class _ConvNd(Module):
    """Class _ConvNd implementation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | float | str | bool | bytes | list | dict | tuple | None,
        stride: int | float | str | bool | bytes | list | dict | tuple | None,
        padding: int | float | str | bool | bytes | list | dict | tuple | None,
        dilation: int | float | str | bool | bytes | list | dict | tuple | None,
        groups: int,
        bias: bool,
        padding_mode: str,
        dtype: DType,
    ) -> None:
        """Initialize _ConvNd with configured parameters and state.

        Args:
            in_channels: Parameter specifying in channels (int).
            out_channels: Parameter specifying out channels (int).
            kernel_size: Parameter specifying kernel size (Union[int, float, str, bool, bytes, list, dict, tuple, None]).
            stride: Parameter specifying stride (Union[int, float, str, bool, bytes, list, dict, tuple, None]).
            padding: Parameter specifying padding (Union[int, float, str, bool, bytes, list, dict, tuple, None]).
            dilation: Parameter specifying dilation (Union[int, float, str, bool, bytes, list, dict, tuple, None]).
            groups: Parameter specifying groups (int).
            bias: Parameter specifying bias (bool).
            padding_mode: Parameter specifying padding mode (str).
            dtype: Parameter specifying dtype (DType).
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.weight: Parameter | None = None
        self.bias: Parameter | None = None

    def forward(self, input: Tensor) -> Tensor:
        """Execute forward on _ConvNd.

        Args:
            input: Parameter specifying input (Tensor).

        Returns:
            Result of forward (Tensor).
        """
        from onnx9000.converters.frontend.utils import record_op

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
        kernel_size: int | tuple[int],
        stride: int | tuple[int] = 1,
        padding: int | tuple[int] = 0,
        dilation: int | tuple[int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        dtype: DType = DType.FLOAT32,
    ) -> None:
        """Initialize Conv1d with configured parameters and state.

        Args:
            in_channels: Parameter specifying in channels (int).
            out_channels: Parameter specifying out channels (int).
            kernel_size: Parameter specifying kernel size (int | tuple[int]).
            stride: Parameter specifying stride (int | tuple[int]).
            padding: Parameter specifying padding (int | tuple[int]).
            dilation: Parameter specifying dilation (int | tuple[int]).
            groups: Parameter specifying groups (int).
            bias: Parameter specifying bias (bool).
            padding_mode: Parameter specifying padding mode (str).
            dtype: Parameter specifying dtype (DType).
        """
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
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        dtype: DType = DType.FLOAT32,
    ) -> None:
        """Initialize Conv2d with configured parameters and state.

        Args:
            in_channels: Parameter specifying in channels (int).
            out_channels: Parameter specifying out channels (int).
            kernel_size: Parameter specifying kernel size (int | tuple[int, int]).
            stride: Parameter specifying stride (int | tuple[int, int]).
            padding: Parameter specifying padding (int | tuple[int, int]).
            dilation: Parameter specifying dilation (int | tuple[int, int]).
            groups: Parameter specifying groups (int).
            bias: Parameter specifying bias (bool).
            padding_mode: Parameter specifying padding mode (str).
            dtype: Parameter specifying dtype (DType).
        """
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
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        dtype: DType = DType.FLOAT32,
    ) -> None:
        """Initialize Conv3d with configured parameters and state.

        Args:
            in_channels: Parameter specifying in channels (int).
            out_channels: Parameter specifying out channels (int).
            kernel_size: Parameter specifying kernel size (int | tuple[int, int, int]).
            stride: Parameter specifying stride (int | tuple[int, int, int]).
            padding: Parameter specifying padding (int | tuple[int, int, int]).
            dilation: Parameter specifying dilation (int | tuple[int, int, int]).
            groups: Parameter specifying groups (int).
            bias: Parameter specifying bias (bool).
            padding_mode: Parameter specifying padding mode (str).
            dtype: Parameter specifying dtype (DType).
        """
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
        kernel_size: int | float | str | bool | bytes | list | dict | tuple | None,
        stride: int | float | str | bool | bytes | list | dict | tuple | None,
        padding: int | float | str | bool | bytes | list | dict | tuple | None,
        output_padding: int | float | str | bool | bytes | list | dict | tuple | None,
        groups: int,
        bias: bool,
        dilation: int | float | str | bool | bytes | list | dict | tuple | None,
        padding_mode: str,
        dtype: DType,
    ) -> None:
        """Initialize _ConvTransposeNd with configured parameters and state.

        Args:
            in_channels: Parameter specifying in channels (int).
            out_channels: Parameter specifying out channels (int).
            kernel_size: Parameter specifying kernel size (Union[int, float, str, bool, bytes, list, dict, tuple, None]).
            stride: Parameter specifying stride (Union[int, float, str, bool, bytes, list, dict, tuple, None]).
            padding: Parameter specifying padding (Union[int, float, str, bool, bytes, list, dict, tuple, None]).
            output_padding: Parameter specifying output padding (Union[int, float, str, bool, bytes, list, dict, tuple, None]).
            groups: Parameter specifying groups (int).
            bias: Parameter specifying bias (bool).
            dilation: Parameter specifying dilation (Union[int, float, str, bool, bytes, list, dict, tuple, None]).
            padding_mode: Parameter specifying padding mode (str).
            dtype: Parameter specifying dtype (DType).
        """
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
        self.weight: Parameter | None = None
        self.bias: Parameter | None = None

    def forward(self, input: Tensor) -> Tensor:
        """Execute forward on _ConvTransposeNd.

        Args:
            input: Parameter specifying input (Tensor).

        Returns:
            Result of forward (Tensor).
        """
        from onnx9000.converters.frontend.utils import record_op

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
        kernel_size: int | tuple[int],
        stride: int | tuple[int] = 1,
        padding: int | tuple[int] = 0,
        output_padding: int | tuple[int] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int | tuple[int] = 1,
        padding_mode: str = "zeros",
        dtype: DType = DType.FLOAT32,
    ) -> None:
        """Initialize ConvTranspose1d with configured parameters and state.

        Args:
            in_channels: Parameter specifying in channels (int).
            out_channels: Parameter specifying out channels (int).
            kernel_size: Parameter specifying kernel size (int | tuple[int]).
            stride: Parameter specifying stride (int | tuple[int]).
            padding: Parameter specifying padding (int | tuple[int]).
            output_padding: Parameter specifying output padding (int | tuple[int]).
            groups: Parameter specifying groups (int).
            bias: Parameter specifying bias (bool).
            dilation: Parameter specifying dilation (int | tuple[int]).
            padding_mode: Parameter specifying padding mode (str).
            dtype: Parameter specifying dtype (DType).
        """
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
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        output_padding: int | tuple[int, int] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int | tuple[int, int] = 1,
        padding_mode: str = "zeros",
        dtype: DType = DType.FLOAT32,
    ) -> None:
        """Initialize ConvTranspose2d with configured parameters and state.

        Args:
            in_channels: Parameter specifying in channels (int).
            out_channels: Parameter specifying out channels (int).
            kernel_size: Parameter specifying kernel size (int | tuple[int, int]).
            stride: Parameter specifying stride (int | tuple[int, int]).
            padding: Parameter specifying padding (int | tuple[int, int]).
            output_padding: Parameter specifying output padding (int | tuple[int, int]).
            groups: Parameter specifying groups (int).
            bias: Parameter specifying bias (bool).
            dilation: Parameter specifying dilation (int | tuple[int, int]).
            padding_mode: Parameter specifying padding mode (str).
            dtype: Parameter specifying dtype (DType).
        """
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
