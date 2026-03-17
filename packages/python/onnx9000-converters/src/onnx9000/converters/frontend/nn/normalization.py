"""Normalization layers."""

from typing import Union

from onnx9000.converters.frontend.nn.module import Module
from onnx9000.converters.frontend.tensor import Parameter, Tensor
from onnx9000.core.dtypes import DType


class _BatchNormNd(Module):
    """Class _BatchNormNd implementation."""

    def __init__(
        self,
        num_features: int,
        eps: float,
        momentum: float,
        affine: bool,
        track_running_stats: bool,
        dtype: DType,
    ) -> None:
        """Implements the __init__ method."""
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter((num_features,), dtype, "weight")
            self.bias = Parameter((num_features,), dtype, "bias")
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.running_mean = Tensor((num_features,), dtype, "running_mean")
            self.running_var = Tensor((num_features,), dtype, "running_var")
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)

    def forward(self, input: Tensor) -> Tensor:
        """Implements the forward method."""
        from onnx9000.converters.frontend.utils import record_op

        attrs = {
            "epsilon": self.eps,
            "momentum": self.momentum,
            "training_mode": 1 if self.training else 0,
        }
        scale = self.weight
        B = self.bias
        mean = self.running_mean
        var = self.running_var
        if scale is None:
            scale = record_op("Constant", [], {"value_float": 1.0})
        if B is None:
            B = record_op("Constant", [], {"value_float": 0.0})
        if mean is None:
            mean = record_op("Constant", [], {"value_float": 0.0})
        if var is None:
            var = record_op("Constant", [], {"value_float": 1.0})
        res = record_op("BatchNormalization", [input, scale, B, mean, var], attrs)
        if isinstance(res, list):
            return res[0]
        return res


class BatchNorm1d(_BatchNormNd):
    """BatchNorm1d layer."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        dtype: DType = DType.FLOAT32,
    ) -> None:
        """Implements the __init__ method."""
        super().__init__(num_features, eps, momentum, affine, track_running_stats, dtype)


class BatchNorm2d(_BatchNormNd):
    """BatchNorm2d layer."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        dtype: DType = DType.FLOAT32,
    ) -> None:
        """Implements the __init__ method."""
        super().__init__(num_features, eps, momentum, affine, track_running_stats, dtype)


class BatchNorm3d(_BatchNormNd):
    """BatchNorm3d layer."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        dtype: DType = DType.FLOAT32,
    ) -> None:
        """Implements the __init__ method."""
        super().__init__(num_features, eps, momentum, affine, track_running_stats, dtype)


class LayerNorm(Module):
    """LayerNorm layer."""

    def __init__(
        self,
        normalized_shape: Union[int, tuple[int]],
        eps: float = 1e-05,
        elementwise_affine: bool = True,
        dtype: DType = DType.FLOAT32,
    ) -> None:
        """Implements the __init__ method."""
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(normalized_shape, dtype, "weight")
            self.bias = Parameter(normalized_shape, dtype, "bias")

    def forward(self, input: Tensor) -> Tensor:
        """Implements the forward method."""
        from onnx9000.converters.frontend.utils import record_op

        attrs = {"axis": -len(self.normalized_shape), "epsilon": self.eps}
        inputs = [input]
        if self.elementwise_affine:
            inputs.append(self.weight)
            inputs.append(self.bias)
        return record_op("LayerNormalization", inputs, attrs)


class GroupNorm(Module):
    """GroupNorm layer."""

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-05,
        affine: bool = True,
        dtype: DType = DType.FLOAT32,
    ) -> None:
        """Implements the __init__ method."""
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = Parameter((num_channels,), dtype, "weight")
            self.bias = Parameter((num_channels,), dtype, "bias")

    def forward(self, input: Tensor) -> Tensor:
        """Implements the forward method."""
        from onnx9000.converters.frontend.utils import record_op

        attrs = {"epsilon": self.eps, "num_groups": self.num_groups}
        inputs = [input]
        if self.affine:
            inputs.append(self.weight)
            inputs.append(self.bias)
        return record_op("GroupNormalization", inputs, attrs)


class _InstanceNormNd(Module):
    """Class _InstanceNormNd implementation."""

    def __init__(
        self,
        num_features: int,
        eps: float,
        momentum: float,
        affine: bool,
        track_running_stats: bool,
        dtype: DType,
    ) -> None:
        """Implements the __init__ method."""
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        if self.affine:
            self.register_parameter("weight", Parameter((num_features,), dtype, "weight"))
            self.register_parameter("bias", Parameter((num_features,), dtype, "bias"))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, input: Tensor) -> Tensor:
        """Implements the forward method."""
        from onnx9000.converters.frontend.utils import record_op

        attrs = {"epsilon": self.eps}
        inputs = [input]
        if self.affine:
            inputs.append(self.weight)
            inputs.append(self.bias)
        return record_op("InstanceNormalization", inputs, attrs)


class InstanceNorm1d(_InstanceNormNd):
    """InstanceNorm1d layer."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        dtype: DType = DType.FLOAT32,
    ) -> None:
        """Implements the __init__ method."""
        super().__init__(num_features, eps, momentum, affine, track_running_stats, dtype)


class InstanceNorm2d(_InstanceNormNd):
    """InstanceNorm2d layer."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        dtype: DType = DType.FLOAT32,
    ) -> None:
        """Implements the __init__ method."""
        super().__init__(num_features, eps, momentum, affine, track_running_stats, dtype)


class InstanceNorm3d(_InstanceNormNd):
    """InstanceNorm3d layer."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        dtype: DType = DType.FLOAT32,
    ) -> None:
        """Implements the __init__ method."""
        super().__init__(num_features, eps, momentum, affine, track_running_stats, dtype)
