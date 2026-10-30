"""Implementation of wrapper (converters / jit / wrapper.py) providing CompiledModel."""

import numpy as np
from onnx9000.core.exceptions import ConversionError
from onnx9000.core.ir import Graph


class CompiledModel:
    """Python wrapper around the dynamically loaded C++ model class.

    Handles the conversion of inputs to numpy arrays and calling the C++ forward method.
    """

    def __init__(
        self, cpp_model: int | float | str | bool | bytes | list | dict | tuple | None, graph: Graph
    ) -> None:
        """Initialize CompiledModel with configured parameters and state.

        Args:
            cpp_model: Parameter specifying cpp model (Union[int, float, str, bool, bytes, list, dict, tuple, None]).
            graph: Parameter specifying graph (Graph).
        """
        self._cpp_model = cpp_model
        self.graph = graph

    def __call__(
        self,
        *args: int | float | str | bool | bytes | list | dict | tuple | None,
        **kwargs: int | float | str | bool | bytes | list | dict | tuple | None,
    ) -> np.ndarray | tuple[np.ndarray]:
        """Invoke CompiledModel with the provided inputs and keyword arguments.

        Args:
            *args: Variadic positional arguments for args.
            **kwargs: Variadic keyword arguments for kwargs.

        Returns:
            Result of call (np.ndarray | tuple[np.ndarray]).
        """
        return self.forward(*args, **kwargs)

    def forward(
        self,
        *args: int | float | str | bool | bytes | list | dict | tuple | None,
        **kwargs: int | float | str | bool | bytes | list | dict | tuple | None,
    ) -> np.ndarray | tuple[np.ndarray]:
        """Execute forward on CompiledModel.

        Args:
            *args: Variadic positional arguments for args.
            **kwargs: Variadic keyword arguments for kwargs.

        Returns:
            Result of forward (np.ndarray | tuple[np.ndarray]).

        Raises:
            ValueError: If an invalid state or argument is encountered during execution.
        """
        input_arrays: list[np.ndarray] = []
        if len(args) != len(self.graph.inputs):
            raise ConversionError(f"Expected {len(self.graph.inputs)} inputs, got {len(args)}")
        for arg in args:
            if not isinstance(arg, np.ndarray):
                arg = np.asarray(arg)
            input_arrays.append(arg)
        out = self._cpp_model.forward(*input_arrays)
        if not isinstance(out, tuple):
            return (out,)
        return out
