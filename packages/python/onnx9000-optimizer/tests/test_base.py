from onnx9000.optimizer.base import PassContext, Pass
from onnx9000.core.ir import Graph


def test_pass_context():
    ctx = PassContext("test")
    ctx.log_change("change")
    assert ctx.modifications == ["change"]


def test_pass_base():
    class MockPass(Pass):
        def run(self, graph):
            return super().run(graph)

    p = MockPass("test_pass")
    g = Graph("g")
    ctx = p.run(g)
    assert ctx.pass_name == "test_pass"
    assert repr(p) == "OptimizationPass(test_pass)"
