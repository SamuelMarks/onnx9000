"""
Operator Registry

Central catalog for translating ONNX ops into C++ generation functions.
"""

from typing import Callable

from onnx9000.core.exceptions import UnsupportedOpError


class OperatorRegistry:
    """
    Registry for mapping ONNX operator types to their respective
    C++ code generator functions or templates.
    """

    def __init__(self) -> None:
        """Provides   init   functionality and verification."""

        self._registry: dict[str, Callable[..., str]] = {}
        self._domains = ["", "ai.onnx.ml", "ai.onnx.preview.training"]

    def register(
        self, op_type: str, domain: str = ""
    ) -> Callable[[Callable[..., str]], Callable[..., str]]:
        """
        Decorator to register a code generator for a specific ONNX op_type.
        """

        def wrapper(func: Callable[..., str]) -> Callable[..., str]:
            """Provides wrapper functionality and verification."""

            if domain not in self._domains:
                raise ValueError(f"Domain '{domain}' is not registered.")

            full_op_type = f"{domain}.{op_type}" if domain else op_type
            if full_op_type in self._registry:
                raise ValueError(f"Operator {full_op_type} is already registered.")
            self._registry[full_op_type] = func
            return func

        return wrapper

    def load_plugin(self, module_name: str) -> None:
        """
        Dynamically load a plugin module to register custom operations.
        """
        import importlib

        importlib.import_module(module_name)

    def get_generator(self, op_type: str, domain: str = "") -> Callable[..., str]:
        """
        Retrieve the code generator for an ONNX operator.
        """
        full_op_type = f"{domain}.{op_type}" if domain else op_type
        if full_op_type not in self._registry:
            raise UnsupportedOpError(full_op_type)
        return self._registry[full_op_type]


# Global registry instance
registry = OperatorRegistry()
