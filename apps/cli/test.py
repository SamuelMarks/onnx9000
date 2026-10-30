"""Module docstring."""

from onnx9000.converters.frontend.nn.module import Module


class MyModel(Module):
    """My model."""

    def __init__(self) -> None:
        """Initialize MyModel with configured parameters and state."""
        super().__init__()

    def forward(
        self, x: int | float | str | bool | bytes | list | dict | tuple | None
    ) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Forward."""
        return x
