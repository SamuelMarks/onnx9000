"""Operator Registry.

Central catalog for registering ONNX operators across the ecosystem.
"""

from typing import Any, Callable, Optional

from onnx9000.core.exceptions import UnsupportedOpError


class OperatorRegistry:
    """Registry for mapping ONNX operator types to their respective.

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

    def register_op(
        self, domain: str, op_type: str, provider: Optional[str] = None
    ) -> Callable[[Any], Any]:
        """Decorate to register a class or function for a specific ONNX op_type."""

        def wrapper(impl: Any) -> Any:
            """Wrap the implementation and register it."""
            if domain not in self._domains:
                self._domains.append(domain)
            key = (domain, op_type, provider)
            self._registry[key] = impl
            return impl

        return wrapper

    def get_op(self, domain: str, op_type: str, provider: Optional[str] = None) -> Any:
        """Retrieve the registered implementation for an ONNX operator."""
        key = (domain, op_type, provider)
        if key not in self._registry:
            # Fallback to no provider if specific one not found
            fallback_key = (domain, op_type, None)
            if fallback_key in self._registry:
                return self._registry[fallback_key]
            raise UnsupportedOpError(f"{domain}.{op_type} (provider={provider})")
        return self._registry[key]

    def get_all_registered(self, provider: Optional[str] = None) -> dict[str, Any]:
        """Return all registered operators for a given provider."""
        return {
            f"{k[0]}.{k[1]}" if k[0] else k[1]: v
            for k, v in self._registry.items()
            if k[2] == provider
        }


global_registry = OperatorRegistry()


def register_op(domain: str, op_type: str, provider: Optional[str] = None) -> Callable[[Any], Any]:
    """Exposed decorator for registering ops."""
    return global_registry.register_op(domain, op_type, provider=provider)
