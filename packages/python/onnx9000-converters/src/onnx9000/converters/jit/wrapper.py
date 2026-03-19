"""Module providing core logic and structural definitions."""

from typing import Any, Union

import numpy as np
from onnx9000.core.ir import Graph


class CompiledModel:
    """Python wrapper around the dynamically loaded C++ model class.

    Handles the conversion of inputs to numpy arrays and calling the C++ forward method.
    """

    def __init__(self, cpp_model: Any, graph: Graph) -> None:
        """Implement the __init__ method or operation."""
        self._cpp_model = cpp_model
        self.graph = graph

    def __call__(self, *args: Any, **kwargs: Any) -> Union[np.ndarray, tuple[np.ndarray]]:
        """Implement the __call__ method or operation."""
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Union[np.ndarray, tuple[np.ndarray]]:
        """Implement the forward method or operation."""
        input_arrays: list[np.ndarray] = []
        if len(args) != len(self.graph.inputs):
            raise ValueError(f"Expected {len(self.graph.inputs)} inputs, got {len(args)}")
        for arg in args:
            if not isinstance(arg, np.ndarray):
                arg = np.asarray(arg)
            input_arrays.append(arg)
        out = self._cpp_model.forward(*input_arrays)
        if not isinstance(out, tuple):
            return (out,)
        return out
