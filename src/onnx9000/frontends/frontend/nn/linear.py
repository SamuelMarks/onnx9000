"""Linear layer."""

from typing import Any, Optional
import math
from onnx9000.frontends.frontend.nn.module import Module
from onnx9000.frontends.frontend.tensor import Parameter, Tensor
from onnx9000.core.dtypes import DType


class Linear(Module):
    """Linear layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: DType = DType.FLOAT32,
    ) -> None:
        """Provides semantic functionality and verification."""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # initialize dummy for trace
        self.weight = Parameter((out_features, in_features), dtype, "weight")
        if bias:
            self.bias = Parameter((out_features,), dtype, "bias")
        else:
            self.register_parameter("bias", None)

    def forward(self, input: Tensor) -> Tensor:
        """Provides forward functionality and verification."""
        # y = x * w^T + b
        # since x is (N, in_features) and weight is (out_features, in_features)
        # we can do x @ weight.T
        res = input @ self.weight.T
        if self.bias is not None:
            res = res + self.bias
        return res
