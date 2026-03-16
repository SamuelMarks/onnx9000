"""Olive optimizer components."""

from onnx9000.optimizer.olive.auto import AutoOptimizer
from onnx9000.optimizer.olive.context import PassContext
from onnx9000.optimizer.olive.evaluator import Evaluator
from onnx9000.optimizer.olive.model import OliveModel
from onnx9000.optimizer.olive.passes import (
    DynamicQuantizationPass,
    GraphFusionPass,
    LayoutConversionPass,
    MixedPrecisionPass,
    OrtPerfTuningPass,
    OrtTransformerOptimizationPass,
    Pass,
    PruningPass,
    QuantizationPass,
    StaticQuantizationPass,
    WeightOnlyQuantizationPass,
)
from onnx9000.optimizer.olive.target import Target

__all__ = [
    "OliveModel",
    "PassContext",
    "Pass",
    "QuantizationPass",
    "DynamicQuantizationPass",
    "StaticQuantizationPass",
    "WeightOnlyQuantizationPass",
    "PruningPass",
    "GraphFusionPass",
    "MixedPrecisionPass",
    "LayoutConversionPass",
    "OrtPerfTuningPass",
    "OrtTransformerOptimizationPass",
    "AutoOptimizer",
    "Evaluator",
    "Target",
]
__all__.extend(
    [
        "ConstantFoldingPass",
        "StripIdentityPass",
        "StripUnusedInitializersPass",
        "ExtractSymbolicShapesPass",
    ]
)
