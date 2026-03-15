from onnx9000.optimize.simplifier.passes.base import GraphPass
from onnx9000.core.ir import Graph


def test_pass_base():

    class DummyPass(GraphPass):
        def run(self, graph: Graph) -> bool:
            super().run(graph)
            return False

    p = DummyPass()
    p.run(Graph("test"))
