"""Test executor wrapper."""

from onnx9000.backends.apple.executor import Dispatcher
from onnx9000.core.ir import Graph


def test_dummy_run() -> None:
    """Dummy run."""
    d = Dispatcher(Graph("test"))
    assert d is not None
