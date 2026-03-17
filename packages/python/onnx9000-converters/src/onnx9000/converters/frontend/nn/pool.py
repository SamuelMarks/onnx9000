"""Pooling layers."""

from typing import Any, Union
from onnx9000.converters.frontend.nn.module import Module
from onnx9000.converters.frontend.tensor import Tensor


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


class _MaxPoolNd(Module):
    """Class _MaxPoolNd implementation."""

    def __init__(
        self, kernel_size: Any, stride: Any, padding: Any, dilation: Any, ceil_mode: bool
    ) -> None:
        """Implements the __init__ method."""
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode

    def forward(self, input: Tensor) -> Tensor:
        """Implements the forward method."""
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
        kernel_size: Union[int, tuple[int]],
        stride: Union[int, tuple[int]] = None,
        padding: Union[int, tuple[int]] = 0,
        dilation: Union[int, tuple[int]] = 1,
        ceil_mode: bool = False,
    ) -> None:
        """Implements the __init__ method."""
        kernel_size_ = _single(kernel_size)
        stride_ = _single(stride) if stride is not None else kernel_size_
        padding_ = _single(padding)
        dilation_ = _single(dilation)
        super().__init__(kernel_size_, stride_, padding_, dilation_, ceil_mode)


class MaxPool2d(_MaxPoolNd):
    """MaxPool2d."""

    def __init__(
        self,
        kernel_size: Union[int, tuple[int, int]],
        stride: Union[int, tuple[int, int]] = None,
        padding: Union[int, tuple[int, int]] = 0,
        dilation: Union[int, tuple[int, int]] = 1,
        ceil_mode: bool = False,
    ) -> None:
        """Implements the __init__ method."""
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride) if stride is not None else kernel_size_
        padding_ = _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(kernel_size_, stride_, padding_, dilation_, ceil_mode)


class _AvgPoolNd(Module):
    """Class _AvgPoolNd implementation."""

    def __init__(
        self, kernel_size: Any, stride: Any, padding: Any, ceil_mode: bool, count_include_pad: bool
    ) -> None:
        """Implements the __init__ method."""
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, input: Tensor) -> Tensor:
        """Implements the forward method."""
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
        kernel_size: Union[int, tuple[int]],
        stride: Union[int, tuple[int]] = None,
        padding: Union[int, tuple[int]] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ) -> None:
        """Implements the __init__ method."""
        kernel_size_ = _single(kernel_size)
        stride_ = _single(stride) if stride is not None else kernel_size_
        padding_ = _single(padding)
        super().__init__(kernel_size_, stride_, padding_, ceil_mode, count_include_pad)


class AvgPool2d(_AvgPoolNd):
    """AvgPool2d."""

    def __init__(
        self,
        kernel_size: Union[int, tuple[int, int]],
        stride: Union[int, tuple[int, int]] = None,
        padding: Union[int, tuple[int, int]] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ) -> None:
        """Implements the __init__ method."""
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride) if stride is not None else kernel_size_
        padding_ = _pair(padding)
        super().__init__(kernel_size_, stride_, padding_, ceil_mode, count_include_pad)


class _AdaptiveAvgPoolNd(Module):
    """Class _AdaptiveAvgPoolNd implementation."""

    def __init__(self, output_size: Any) -> None:
        """Implements the __init__ method."""
        super().__init__()
        self.output_size = output_size

    def forward(self, input: Tensor) -> Tensor:
        """Implements the forward method."""
        from onnx9000.converters.frontend.utils import record_op

        return record_op("GlobalAveragePool", [input])


class AdaptiveAvgPool2d(_AdaptiveAvgPoolNd):
    """AdaptiveAvgPool2d"""

    def __init__(self, output_size: Union[int, tuple[int, int]]) -> None:
        """Implements the __init__ method."""
        super().__init__(output_size)
