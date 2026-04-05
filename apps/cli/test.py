"""Module docstring."""

from onnx9000.converters.frontend.nn.module import Module


class MyModel(Module):
    """My model."""

    def __init__(self):
        """Init."""
        super().__init__()

    def forward(self, x):
        """Forward."""
        return x
