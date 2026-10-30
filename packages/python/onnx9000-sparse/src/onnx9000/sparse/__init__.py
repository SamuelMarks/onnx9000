"""Sparse tensor format conversions and SparseML recipe modifiers for ONNX9000."""

from onnx9000.core.sparse import (
    get_byte_size,
    get_struct_fmt,
    pack_data,
    pack_sparse_int8,
    unpack_data,
)
from onnx9000.optimizer.sparse.modifier import ConstantPruningModifier, Modifier


def run() -> str:
    """Execute the default sparse module status check.

    Returns:
        Sparse status confirmation string.
    """
    return "[sparse] processed"


__all__ = [
    "ConstantPruningModifier",
    "Modifier",
    "get_byte_size",
    "get_struct_fmt",
    "pack_data",
    "pack_sparse_int8",
    "run",
    "unpack_data",
]
