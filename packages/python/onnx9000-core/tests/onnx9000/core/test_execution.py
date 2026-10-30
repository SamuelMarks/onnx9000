"""Unit tests for execution provider abstractions in `onnx9000.core.execution`."""

from onnx9000.core.execution import (
    Environment,
    ExecutionContext,
    ExecutionProvider,
    RunOptions,
    SessionOptions,
)
from onnx9000.core.ir import Graph


class DummyProvider(ExecutionProvider):
    """Concrete ExecutionProvider subclass for testing."""

    def get_supported_nodes(self, graph: Graph) -> list[str]:
        """Return all node names."""
        return [n.name for n in graph.nodes]

    def allocate_tensors(self, tensors: list[object]) -> None:
        """No-op allocation."""
        return None

    def execute(
        self, graph: Graph, context: ExecutionContext, inputs: dict[str, object]
    ) -> dict[str, object]:
        """Return inputs as outputs."""
        return inputs


def test_execution_classes() -> None:
    """Verify SessionOptions, RunOptions, Environment, ExecutionContext, and ExecutionProvider."""
    so = SessionOptions()
    assert so.execution_mode == "SEQUENTIAL"

    ro = RunOptions()
    assert ro.terminate is False

    env = Environment()
    assert env is Environment()
    assert Environment.get_device() == "CPU"

    ctx = ExecutionContext(options=so)
    assert ctx.device_id == 0

    ep = DummyProvider({"device_id": 1})
    assert ep.device_id == 1
    g = Graph("exec_g")
    assert ep.get_supported_nodes(g) == []
    assert ep.execute(g, ctx, {"x": 1}) == {"x": 1}
