"""
Operator Registry

Central catalog for registering ONNX operators across the ecosystem.
"""

from typing import Any, Callable

from onnx9000.core.exceptions import UnsupportedOpError


class OperatorRegistry:
    """
    Registry for mapping ONNX operator types to their respective
    implementations, parsers, or execution kernels.
    """

    def __init__(self) -> None:
        """Initialize the operator registry."""
        self._registry: dict[str, Any] = {}
        self._domains: list[str] = [
            "",
            "ai.onnx",
            "ai.onnx.ml",
            "ai.onnx.preview.training",
            "ai.onnx.contrib",
        ]

    def register_op(self, op_type: str, domain: str = "") -> Callable[[Any], Any]:
        """
        Decorator to register a class or function for a specific ONNX op_type.
        """

        def wrapper(impl: Any) -> Any:
            """Wrap the implementation and register it."""
            if domain not in self._domains:
                self._domains.append(domain)
            full_op_type = f"{domain}.{op_type}" if domain else op_type
            if full_op_type in self._registry:
                pass  # Already registered, allow overwrite to support reloading
            self._registry[full_op_type] = impl
            return impl

        return wrapper

    def get_op(self, op_type: str, domain: str = "") -> Any:
        """
        Retrieve the registered implementation for an ONNX operator.
        """
        full_op_type = f"{domain}.{op_type}" if domain else op_type
        if full_op_type not in self._registry:
            raise UnsupportedOpError(full_op_type)
        return self._registry[full_op_type]


global_registry = OperatorRegistry()


def register_op(op_type: str, domain: str = "") -> Callable[[Any], Any]:
    """Exposed decorator for registering ops."""
    return global_registry.register_op(op_type, domain)
