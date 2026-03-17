"""Tests for AutoOptimizer and Evaluator."""

from typing import NoReturn

from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.optimizer.olive.auto import AutoOptimizer
from onnx9000.optimizer.olive.evaluator import Evaluator
from onnx9000.optimizer.olive.model import OliveModel
from onnx9000.optimizer.olive.passes import Pass
from onnx9000.optimizer.olive.target import Target


class DummyPass(Pass):
    """A dummy pass."""

    def __init__(self) -> None:
        """Init."""
        super().__init__("DummyPass")

    def run(self, model, context):
        """Run."""
        return model


class FailingPass(Pass):
    """A failing pass."""

    def __init__(self) -> None:
        """Init."""
        super().__init__("FailingPass")

    def run(self, model, context) -> NoReturn:
        """Run."""
        raise RuntimeError("Fail")


def test_target() -> None:
    """Test Target enum."""
    assert Target.CPU.name == "CPU"
    assert Target.WebGPU.name == "WebGPU"
    assert Target.WASM_SIMD.name == "WASM_SIMD"
    assert Target.Accelerate.name == "Accelerate"
    assert Target.CoreML.name == "CoreML"
    assert Target.TensorRT.name == "TensorRT"


def test_evaluator() -> None:
    """Test Evaluator."""
    g = Graph("test")
    g.add_node(Node("Dummy", [], [], {}, "n1"))
    g.add_node(Node("Dummy2", [], [], {}, "n2"))
    from onnx9000.core.ir import DynamicDim

    g.add_tensor(Tensor("t1", "FLOAT32", [DynamicDim("N")], [1.0]))
    model = OliveModel(g)
    assert Evaluator.evaluate_flops(model) == 2000
    assert Evaluator.evaluate_memory(model) == 1024
    assert Evaluator.evaluate_accuracy(model, model) == 1.0
    assert Evaluator.track_latency() == 0.0
    report = Evaluator.generate_report(model, model)
    assert report["flops_before"] == 2000
    assert report["flops_after"] == 2000
    assert report["memory_before"] == 1024
    assert report["memory_after"] == 1024
    assert "netron_link" in report


def test_auto_optimizer() -> None:
    """Test AutoOptimizer."""
    g = Graph("test")
    model = OliveModel(g)
    passes = [DummyPass(), FailingPass()]
    opt = AutoOptimizer(Target.WebGPU, passes, {"test": 1})
    assert opt.target == Target.WebGPU
    assert opt.config == {"test": 1}
    res = opt.optimize(model)
    assert isinstance(res, OliveModel)
