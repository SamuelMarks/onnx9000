"""Pooling layers."""

from onnx9000.converters.frontend.nn.module import Module
from onnx9000.converters.frontend.tensor import Tensor


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


class _MaxPoolNd(Module):
    """Class _MaxPoolNd implementation."""

    def __init__(
        self,
        kernel_size: int | float | str | bool | bytes | list | dict | tuple | None,
        stride: int | float | str | bool | bytes | list | dict | tuple | None,
        padding: int | float | str | bool | bytes | list | dict | tuple | None,
        dilation: int | float | str | bool | bytes | list | dict | tuple | None,
        ceil_mode: bool,
    ) -> None:
        """Initialize _MaxPoolNd with configured parameters and state.

        Args:
            kernel_size: Parameter specifying kernel size (Union[int, float, str, bool, bytes, list, dict, tuple, None]).
            stride: Parameter specifying stride (Union[int, float, str, bool, bytes, list, dict, tuple, None]).
            padding: Parameter specifying padding (Union[int, float, str, bool, bytes, list, dict, tuple, None]).
            dilation: Parameter specifying dilation (Union[int, float, str, bool, bytes, list, dict, tuple, None]).
            ceil_mode: Parameter specifying ceil mode (bool).
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode

    def forward(self, input: Tensor) -> Tensor:
        """Execute forward on _MaxPoolNd.

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
            "ceil_mode": 1 if self.ceil_mode else 0,
        }
        return record_op("MaxPool", [input], attrs)


class MaxPool1d(_MaxPoolNd):
    """MaxPool1d."""

    def __init__(
        self,
        kernel_size: int | tuple[int],
        stride: int | tuple[int] | None = None,
        padding: int | tuple[int] = 0,
        dilation: int | tuple[int] = 1,
        ceil_mode: bool = False,
    ) -> None:
        """Initialize MaxPool1d with configured parameters and state.

        Args:
            kernel_size: Parameter specifying kernel size (int | tuple[int]).
            stride: Parameter specifying stride (int | tuple[int]).
            padding: Parameter specifying padding (int | tuple[int]).
            dilation: Parameter specifying dilation (int | tuple[int]).
            ceil_mode: Parameter specifying ceil mode (bool).
        """
        kernel_size_ = _single(kernel_size)
        stride_ = _single(stride) if stride is not None else kernel_size_
        padding_ = _single(padding)
        dilation_ = _single(dilation)
        super().__init__(kernel_size_, stride_, padding_, dilation_, ceil_mode)


class MaxPool2d(_MaxPoolNd):
    """MaxPool2d."""

    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        ceil_mode: bool = False,
    ) -> None:
        """Initialize MaxPool2d with configured parameters and state.

        Args:
            kernel_size: Parameter specifying kernel size (int | tuple[int, int]).
            stride: Parameter specifying stride (int | tuple[int, int]).
            padding: Parameter specifying padding (int | tuple[int, int]).
            dilation: Parameter specifying dilation (int | tuple[int, int]).
            ceil_mode: Parameter specifying ceil mode (bool).
        """
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride) if stride is not None else kernel_size_
        padding_ = _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(kernel_size_, stride_, padding_, dilation_, ceil_mode)


class _AvgPoolNd(Module):
    """Class _AvgPoolNd implementation."""

    def __init__(
        self,
        kernel_size: int | float | str | bool | bytes | list | dict | tuple | None,
        stride: int | float | str | bool | bytes | list | dict | tuple | None,
        padding: int | float | str | bool | bytes | list | dict | tuple | None,
        ceil_mode: bool,
        count_include_pad: bool,
    ) -> None:
        """Initialize _AvgPoolNd with configured parameters and state.

        Args:
            kernel_size: Parameter specifying kernel size (Union[int, float, str, bool, bytes, list, dict, tuple, None]).
            stride: Parameter specifying stride (Union[int, float, str, bool, bytes, list, dict, tuple, None]).
            padding: Parameter specifying padding (Union[int, float, str, bool, bytes, list, dict, tuple, None]).
            ceil_mode: Parameter specifying ceil mode (bool).
            count_include_pad: Parameter specifying count include pad (bool).
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, input: Tensor) -> Tensor:
        """Execute forward on _AvgPoolNd.

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
            "ceil_mode": 1 if self.ceil_mode else 0,
            "count_include_pad": 1 if self.count_include_pad else 0,
        }
        return record_op("AveragePool", [input], attrs)


class AvgPool1d(_AvgPoolNd):
    """AvgPool1d."""

    def __init__(
        self,
        kernel_size: int | tuple[int],
        stride: int | tuple[int] | None = None,
        padding: int | tuple[int] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ) -> None:
        """Initialize AvgPool1d with configured parameters and state.

        Args:
            kernel_size: Parameter specifying kernel size (int | tuple[int]).
            stride: Parameter specifying stride (int | tuple[int]).
            padding: Parameter specifying padding (int | tuple[int]).
            ceil_mode: Parameter specifying ceil mode (bool).
            count_include_pad: Parameter specifying count include pad (bool).
        """
        kernel_size_ = _single(kernel_size)
        stride_ = _single(stride) if stride is not None else kernel_size_
        padding_ = _single(padding)
        super().__init__(kernel_size_, stride_, padding_, ceil_mode, count_include_pad)


class AvgPool2d(_AvgPoolNd):
    """AvgPool2d."""

    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ) -> None:
        """Initialize AvgPool2d with configured parameters and state.

        Args:
            kernel_size: Parameter specifying kernel size (int | tuple[int, int]).
            stride: Parameter specifying stride (int | tuple[int, int]).
            padding: Parameter specifying padding (int | tuple[int, int]).
            ceil_mode: Parameter specifying ceil mode (bool).
            count_include_pad: Parameter specifying count include pad (bool).
        """
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride) if stride is not None else kernel_size_
        padding_ = _pair(padding)
        super().__init__(kernel_size_, stride_, padding_, ceil_mode, count_include_pad)


class _AdaptiveAvgPoolNd(Module):
    """Class _AdaptiveAvgPoolNd implementation."""

    def __init__(
        self, output_size: int | float | str | bool | bytes | list | dict | tuple | None
    ) -> None:
        """Initialize _AdaptiveAvgPoolNd with configured parameters and state.

        Args:
            output_size: Parameter specifying output size (Union[int, float, str, bool, bytes, list, dict, tuple, None]).
        """
        super().__init__()
        self.output_size = output_size

    def forward(self, input: Tensor) -> Tensor:
        """Execute forward on _AdaptiveAvgPoolNd.

        Args:
            input: Parameter specifying input (Tensor).

        Returns:
            Result of forward (Tensor).
        """
        from onnx9000.converters.frontend.utils import record_op

        return record_op("GlobalAveragePool", [input])


class AdaptiveAvgPool2d(_AdaptiveAvgPoolNd):
    """AdaptiveAvgPool2d."""

    def __init__(self, output_size: int | tuple[int, int]) -> None:
        """Initialize AdaptiveAvgPool2d with configured parameters and state.

        Args:
            output_size: Parameter specifying output size (int | tuple[int, int]).
        """
        super().__init__(output_size)
