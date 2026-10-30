"""Comprehensive unit tests for onnx9000.sparse and onnx9000_sparse."""

import onnx9000_sparse
from onnx9000.core.dtypes import DType
from onnx9000.sparse import (
    ConstantPruningModifier,
    get_byte_size,
    pack_data,
    pack_sparse_int8,
    run,
    unpack_data,
)


def test_sparse_packing_and_modifiers() -> None:
    """Verify run(), pack_data(), unpack_data(), pack_sparse_int8(), and ConstantPruningModifier."""
    assert run() == "[sparse] processed"
    assert onnx9000_sparse.run() == "[sparse] processed"
    assert get_byte_size(DType.FLOAT32) == 4

    raw = pack_data([1.0, 2.0], DType.FLOAT32)
    assert unpack_data(raw, DType.FLOAT32) == [1.0, 2.0]
    assert len(pack_sparse_int8([1, 2], [0, 3])) == 3

    mod = ConstantPruningModifier(params=["weight"])
    assert mod.params == ["weight"]
