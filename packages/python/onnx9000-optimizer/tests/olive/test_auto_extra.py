"""Extra tests for AutoOptimizer."""

from onnx9000.core.ir import Graph
from onnx9000.optimizer.olive.auto import AutoOptimizer
from onnx9000.optimizer.olive.model import OliveModel
from onnx9000.optimizer.olive.passes import Pass
from onnx9000.optimizer.olive.target import Target


class SkipPass(Pass):
    """A pass that should be skipped."""

    def __init__(self) -> None:
        super().__init__("SkipPass")

    def run(self, model, context):
        return model


class CustomOptimizer(AutoOptimizer):
    def _check_hardware_limits(self, p: Pass) -> bool:
        return p.name != "SkipPass"


def test_skip_pass() -> None:
    g = Graph("test")
    model = OliveModel(g)
    passes = [SkipPass()]
    opt = CustomOptimizer(Target.CPU, passes)
    res = opt.optimize(model)
    assert res is model
