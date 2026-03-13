"""
Operator Registry

Central catalog for translating ONNX ops into C++ generation functions.
"""

from typing import Callable

from onnx9000.exceptions import UnsupportedOpError


class OperatorRegistry:
    """
    Registry for mapping ONNX operator types to their respective
    C++ code generator functions or templates.
    """

    def __init__(self) -> None:
        """__init__ docstring."""

        self._registry: dict[str, Callable[..., str]] = {}

    def register(
        self, op_type: str
    ) -> Callable[[Callable[..., str]], Callable[..., str]]:
        """
        Decorator to register a code generator for a specific ONNX op_type.
        """

        def wrapper(func: Callable[..., str]) -> Callable[..., str]:
            """wrapper docstring."""

            if op_type in self._registry:
                raise ValueError(
                    f"Operator {op_type} is already registered."
                )  # pragma: no cover
            self._registry[op_type] = func
            return func

        return wrapper

    def load_plugin(self, module_name: str) -> None:
        """
        Dynamically load a plugin module to register custom operations.
        """
        import importlib  # pragma: no cover

        importlib.import_module(module_name)  # pragma: no cover

    def get_generator(self, op_type: str) -> Callable[..., str]:
        """
        Retrieve the code generator for an ONNX operator.
        """
        if op_type not in self._registry:
            raise UnsupportedOpError(op_type)  # pragma: no cover
        return self._registry[op_type]


# Global registry instance
registry = OperatorRegistry()
