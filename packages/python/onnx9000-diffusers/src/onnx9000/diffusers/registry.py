"""Module docstring."""

from typing import Any, Callable


def register_op(domain: str, op_name: str) -> Callable:
    """Decorator to register an operator to the core dispatcher."""

    def decorator(cls: Any) -> Any:
        cls._domain = domain
        cls._op_name = op_name
        return cls

    return decorator
