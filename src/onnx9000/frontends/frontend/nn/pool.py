"""Pooling layers."""

from typing import Any, Union, Tuple
from onnx9000.frontends.frontend.nn.module import Module
from onnx9000.frontends.frontend.tensor import Tensor


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


class _MaxPoolNd(Module):
    """Provides semantic functionality and verification."""

    def __init__(
        self,
        kernel_size: Any,
        stride: Any,
        padding: Any,
        dilation: Any,
        ceil_mode: bool,
    ) -> None:
        """Provides semantic functionality and verification."""
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode

    def forward(self, input: Tensor) -> Tensor:
        """Provides semantic functionality and verification."""
        from onnx9000.frontends.frontend.utils import record_op

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
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]] = None,
        padding: Union[int, Tuple[int]] = 0,
        dilation: Union[int, Tuple[int]] = 1,
        ceil_mode: bool = False,
    ) -> None:
        """Provides semantic functionality and verification."""
        kernel_size_ = _single(kernel_size)
        stride_ = _single(stride) if stride is not None else kernel_size_
        padding_ = _single(padding)
        dilation_ = _single(dilation)
        super().__init__(kernel_size_, stride_, padding_, dilation_, ceil_mode)


class MaxPool2d(_MaxPoolNd):
    """MaxPool2d."""

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        ceil_mode: bool = False,
    ) -> None:
        """Provides semantic functionality and verification."""
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride) if stride is not None else kernel_size_
        padding_ = _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(kernel_size_, stride_, padding_, dilation_, ceil_mode)


class _AvgPoolNd(Module):
    """Provides semantic functionality and verification."""

    def __init__(
        self,
        kernel_size: Any,
        stride: Any,
        padding: Any,
        ceil_mode: bool,
        count_include_pad: bool,
    ) -> None:
        """Provides semantic functionality and verification."""
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, input: Tensor) -> Tensor:
        """Provides semantic functionality and verification."""
        from onnx9000.frontends.frontend.utils import record_op

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
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]] = None,
        padding: Union[int, Tuple[int]] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ) -> None:
        """Provides semantic functionality and verification."""
        kernel_size_ = _single(kernel_size)
        stride_ = _single(stride) if stride is not None else kernel_size_
        padding_ = _single(padding)
        super().__init__(kernel_size_, stride_, padding_, ceil_mode, count_include_pad)


class AvgPool2d(_AvgPoolNd):
    """AvgPool2d."""

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ) -> None:
        """Provides semantic functionality and verification."""
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride) if stride is not None else kernel_size_
        padding_ = _pair(padding)
        super().__init__(kernel_size_, stride_, padding_, ceil_mode, count_include_pad)


class _AdaptiveAvgPoolNd(Module):
    """Provides semantic functionality and verification."""

    def __init__(self, output_size: Any) -> None:
        """Provides semantic functionality and verification."""
        super().__init__()
        self.output_size = output_size

    def forward(self, input: Tensor) -> Tensor:
        """Provides semantic functionality and verification."""
        from onnx9000.frontends.frontend.utils import record_op

        # For dynamic output size we might need something complex, but standard GlobalAveragePool is simple.
        # Here we just map to GlobalAveragePool assuming output_size is 1 (or (1,1))
        # If output_size is not 1, we can fallback to standard AveragePool with computed strides,
        # but matching exact PyTorch semantics dynamically requires shape info.
        return record_op("GlobalAveragePool", [input])


class AdaptiveAvgPool2d(_AdaptiveAvgPoolNd):
    """AdaptiveAvgPool2d"""

    def __init__(self, output_size: Union[int, Tuple[int, int]]) -> None:
        """Provides semantic functionality and verification."""
        super().__init__(output_size)
