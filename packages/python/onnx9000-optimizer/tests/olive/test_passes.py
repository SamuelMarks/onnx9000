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


def test_olive_passes() -> None:
    """Test all passes."""
    graph = Graph("test")
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
        LayoutConversionPass(),
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
