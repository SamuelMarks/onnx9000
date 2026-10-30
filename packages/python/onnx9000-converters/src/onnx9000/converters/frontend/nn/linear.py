"""Linear layer."""

from onnx9000.converters.frontend.nn.module import Module
from onnx9000.converters.frontend.tensor import Parameter, Tensor
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
        """Initialize Linear with configured parameters and state.

        Args:
            in_features: Parameter specifying in features (int).
            out_features: Parameter specifying out features (int).
            bias: Parameter specifying bias (bool).
            dtype: Parameter specifying dtype (DType).
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter((out_features, in_features), dtype, "weight")
        if bias:
            self.bias = Parameter((out_features,), dtype, "bias")
        else:
            self.register_parameter("bias", None)

    def forward(self, input: Tensor) -> Tensor:
        """Execute forward on Linear.

        Args:
            input: Parameter specifying input (Tensor).

        Returns:
            Result of forward (Tensor).
        """
        res = input @ self.weight.T
        if self.bias is not None:
            res = res + self.bias
        return res
