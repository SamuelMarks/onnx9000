"""Unit tests for onnx9000_graphsurgeon re-exports."""

from onnx9000_graphsurgeon import GraphSurgeon, cleanup, fold_constants, run, toposort


def test_graphsurgeon_reexports() -> None:
    """Verify all symbols are re-exported from onnx9000_graphsurgeon."""
    assert run() == "[graphsurgeon] processed"
    assert GraphSurgeon is not None
    assert callable(cleanup)
    assert callable(fold_constants)
    assert callable(toposort)
