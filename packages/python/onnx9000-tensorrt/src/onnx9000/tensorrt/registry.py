"""Registry."""

import logging
from collections.abc import Callable

from onnx9000.core.exceptions import BackendExecutionError

logger = logging.getLogger("onnx9000.tensorrt.registry")

_TRT_OP_REGISTRY: dict[tuple[str, str], Callable] = {}


def register_op(
    domain: str, op_name: str
) -> int | float | str | bool | bytes | list | dict | tuple | None:
    """Decorate to register an ONNX operator to TensorRT translator function."""

    def decorator(func: Callable) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Execute decorator."""
        key = (domain, op_name)
        if key in _TRT_OP_REGISTRY:
            logger.warning(f"Overwriting previously registered TRT op for {domain}::{op_name}")
        _TRT_OP_REGISTRY[key] = func
        return func

    return decorator


def get_op_translator(domain: str, op_name: str) -> Callable:
    """Execute get_op_translator."""
    key = (domain, op_name)
    if key not in _TRT_OP_REGISTRY:
        raise BackendExecutionError(f"No TensorRT translation registered for {domain}::{op_name}")
    return _TRT_OP_REGISTRY[key]
