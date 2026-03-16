from onnx9000.core.ir import Graph
from onnx9000.optimizer.simplifier.passes.base import GraphPass


def test_pass_base() -> None:

    class DummyPass(GraphPass):
        def run(self, graph: Graph) -> bool:
            super().run(graph)
            return False

    p = DummyPass()
    p.run(Graph("test"))
