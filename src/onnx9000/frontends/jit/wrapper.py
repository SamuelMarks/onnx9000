"""Module providing core logic and structural definitions."""

from typing import Any, Union

import numpy as np

from onnx9000.core.ir import Graph


class CompiledModel:
    """
    Python wrapper around the dynamically loaded C++ model class.
    Handles the conversion of inputs to numpy arrays and calling the C++ forward method.
    """

    def __init__(self, cpp_model: Any, graph: Graph):
        """Provides   init   functionality and verification."""

        self._cpp_model = cpp_model
        self.graph = graph

    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> Union[np.ndarray, tuple[np.ndarray]]:
        """Provides   call   functionality and verification."""

        return self.forward(*args, **kwargs)

    def forward(
        self, *args: Any, **kwargs: Any
    ) -> Union[np.ndarray, tuple[np.ndarray]]:
        # Map args/kwargs to expected graph inputs

        """Provides forward functionality and verification."""

        input_arrays: list[np.ndarray] = []

        # Simple list-based argument mapping for now
        if len(args) != len(self.graph.inputs):
            raise ValueError(
                f"Expected {len(self.graph.inputs)} inputs, got {len(args)}"
            )

        for arg in args:
            if not isinstance(arg, np.ndarray):
                arg = np.asarray(arg)
            input_arrays.append(arg)

        # The C++ forward_py expects arguments in the exact order of graph.inputs
        out = self._cpp_model.forward(*input_arrays)
        if not isinstance(out, tuple):
            return (out,)
        return out
