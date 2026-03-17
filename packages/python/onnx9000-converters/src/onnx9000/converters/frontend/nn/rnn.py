"""RNN layers."""

from typing import Optional

from onnx9000.converters.frontend.nn.module import Module
from onnx9000.converters.frontend.tensor import Parameter, Tensor
from onnx9000.core.dtypes import DType


class RNNBase(Module):
    """Class RNNBase implementation."""

    def __init__(
        self,
        mode: str,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        bias: bool,
        batch_first: bool,
        dropout: float,
        bidirectional: bool,
        dtype: DType,
    ) -> None:
        """Implements the __init__ method."""
        super().__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.weight_ih_l0 = Parameter(
            (self.num_directions * hidden_size, input_size), dtype, "weight_ih_l0"
        )
        self.weight_hh_l0 = Parameter(
            (self.num_directions * hidden_size, hidden_size), dtype, "weight_hh_l0"
        )
        if bias:
            self.bias_ih_l0 = Parameter((self.num_directions * hidden_size,), dtype, "bias_ih_l0")
            self.bias_hh_l0 = Parameter((self.num_directions * hidden_size,), dtype, "bias_hh_l0")

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        """Implements the forward method."""
        from onnx9000.converters.frontend.utils import record_op

        attrs = {
            "hidden_size": self.hidden_size,
            "direction": "bidirectional" if self.bidirectional else "forward",
        }
        inputs = [input, self.weight_ih_l0, self.weight_hh_l0]
        if self.bias:
            inputs.append(self.bias_ih_l0)
        if hx is not None:
            inputs.append(hx)
        op_name = self.mode
        res = record_op(op_name, inputs, attrs)
        if isinstance(res, list) and len(res) >= 2:
            return (res[0], res[1])
        return (res, res)


class RNN(RNNBase):
    """RNN layer."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: str = "tanh",
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        dtype: DType = DType.FLOAT32,
    ) -> None:
        """Implements the __init__ method."""
        super().__init__(
            "RNN",
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout,
            bidirectional,
            dtype,
        )


class LSTM(RNNBase):
    """LSTM layer."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        dtype: DType = DType.FLOAT32,
    ) -> None:
        """Implements the __init__ method."""
        super().__init__(
            "LSTM",
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout,
            bidirectional,
            dtype,
        )

    def forward(
        self, input: Tensor, hx: Optional[tuple[Tensor, Tensor]] = None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Implements the forward method."""
        from onnx9000.converters.frontend.utils import record_op

        attrs = {
            "hidden_size": self.hidden_size,
            "direction": "bidirectional" if self.bidirectional else "forward",
        }
        inputs = [input, self.weight_ih_l0, self.weight_hh_l0]
        if self.bias:
            inputs.append(self.bias_ih_l0)
        else:
            inputs.append(None)
        if hx is not None:
            inputs.extend([hx[0], hx[1]])
        res = record_op("LSTM", inputs, attrs)
        if isinstance(res, list) and len(res) >= 2:
            return (res[0], (res[1], res[2] if len(res) > 2 else res[1]))
        return (res, (res, res))


class GRU(RNNBase):
    """GRU layer."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        dtype: DType = DType.FLOAT32,
    ) -> None:
        """Implements the __init__ method."""
        super().__init__(
            "GRU",
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout,
            bidirectional,
            dtype,
        )

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        """Implements the forward method."""
        from onnx9000.converters.frontend.utils import record_op

        attrs = {
            "hidden_size": self.hidden_size,
            "direction": "bidirectional" if self.bidirectional else "forward",
        }
        inputs = [input, self.weight_ih_l0, self.weight_hh_l0]
        if self.bias:
            inputs.append(self.bias_ih_l0)
        if hx is not None:
            inputs.append(hx)
        res = record_op("GRU", inputs, attrs)
        if isinstance(res, list) and len(res) >= 2:
            return (res[0], res[1])
        return (res, res)
