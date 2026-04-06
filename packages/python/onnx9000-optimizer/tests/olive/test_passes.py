"""Tests for Olive optimizer components."""

from onnx9000.core.ir import Graph
from onnx9000.optimizer.olive.context import PassContext
from onnx9000.optimizer.olive.model import OliveModel
from onnx9000.optimizer.olive.passes import (
    DynamicQuantizationPass,
    GraphFusionPass,
    LayoutConversionPass,
    MixedPrecisionPass,
    OrtPerfTuningPass,
    OrtTransformerOptimizationPass,
    PruningPass,
    QuantizationPass,
    StaticQuantizationPass,
    WeightOnlyQuantizationPass,
)


def test_olive_model() -> None:
    """Test OliveModel."""
    graph = Graph("test")
    model = OliveModel(graph)
    assert model.graph is graph
    assert model.metadata == {}
    model2 = OliveModel(graph, metadata={"foo": "bar"})
    assert model2.metadata == {"foo": "bar"}


def test_pass_context() -> None:
    """Test PassContext."""
    ctx = PassContext()
    assert ctx.state == {}
    ctx2 = PassContext(state={"foo": "bar"})
    assert ctx2.state == {"foo": "bar"}


from onnx9000.core.ir import Node


def test_olive_passes() -> None:
    """Test all passes."""
    graph = Graph("test")
    graph.nodes.append(Node("MatMul", ["A", "B"], ["C"]))
    model = OliveModel(graph)
    ctx = PassContext()
    passes = [
        QuantizationPass(),
        DynamicQuantizationPass(),
        StaticQuantizationPass(),
        WeightOnlyQuantizationPass(),
        PruningPass(),
        GraphFusionPass(),
        MixedPrecisionPass(),
        LayoutConversionPass(config={"target_layout": "NHWC"}),
        LayoutConversionPass(config={"target_layout": "NCHW"}),
        OrtPerfTuningPass(),
        OrtTransformerOptimizationPass(),
    ]
    for p in passes:
        assert isinstance(p.name, str)
        assert isinstance(p.config, dict)
        res = p.run(model, ctx)
        assert res is model


from onnx9000.optimizer.olive.passes import (
    ConstantFoldingPass,
    ExtractSymbolicShapesPass,
    StripIdentityPass,
    StripUnusedInitializersPass,
)


def test_extra_passes() -> None:
    """Tests the extra passes functionality."""
    graph = Graph("test")
    graph.nodes.append(Node("Identity", ["A"], ["B"]))
    graph.nodes.append(Node("Add", ["B", "C"], ["D"]))
    graph.initializers = ["C", "unused_init"]
    from onnx9000.core.ir import Tensor
    from onnx9000.core.dtypes import DType

    graph.tensors["unused_init"] = Tensor(
        "unused_init", (1,), DType.FLOAT32, data=b"", is_initializer=True
    )
    graph.tensors["C"] = Tensor("C", (1,), DType.FLOAT32, data=b"", is_initializer=True)
    graph.tensors["normal_tensor"] = Tensor(
        "normal_tensor", (1,), DType.FLOAT32, data=None, is_initializer=False
    )

    from onnx9000.core.ir import ValueInfo

    graph.outputs.append(ValueInfo("D", DType.FLOAT32, (1,)))

    model = OliveModel(graph)
    ctx = PassContext()
    passes = [
        ConstantFoldingPass(),
        StripIdentityPass(),
        StripUnusedInitializersPass(),
        ExtractSymbolicShapesPass(),
    ]
    for p in passes:
        assert isinstance(p.name, str)
        assert isinstance(p.config, dict)
        res = p.run(model, ctx)
        assert res is model
