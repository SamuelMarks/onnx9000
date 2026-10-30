"""Identity layer."""

from onnx9000.converters.frontend.nn.module import Module
from onnx9000.converters.frontend.tensor import Tensor


class Identity(Module):
    """A placeholder identity operator that is argument-insensitive."""

    def __init__(
        self,
        *args: int | float | str | bool | bytes | list | dict | tuple | None,
        **kwargs: int | float | str | bool | bytes | list | dict | tuple | None,
    ) -> None:
        """Initialize Identity with configured parameters and state.

        Args:
            *args: Variadic positional arguments for args.
            **kwargs: Variadic keyword arguments for kwargs.
        """
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        """Execute forward on Identity.

        Args:
            input: Parameter specifying input (Tensor).

        Returns:
            Result of forward (Tensor).
        """
        from onnx9000.converters.frontend.utils import record_op

        return record_op("Identity", [input])
