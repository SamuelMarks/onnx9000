"""Linear layer."""

from onnx9000.core.dtypes import DType
from onnx9000.frontend.frontend.nn.module import Module
from onnx9000.frontend.frontend.tensor import Parameter, Tensor


class Linear(Module):
    """Linear layer."""

    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, dtype: DType = DType.FLOAT32
    ) -> None:
        """Implements the __init__ method."""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter((out_features, in_features), dtype, "weight")
        if bias:
            self.bias = Parameter((out_features,), dtype, "bias")
        else:
            self.register_parameter("bias", None)

    def forward(self, input: Tensor) -> Tensor:
        """Implements the forward method."""
        res = input @ self.weight.T
        if self.bias is not None:
            res = res + self.bias
        return res
