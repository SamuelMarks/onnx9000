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
    "AutoOptimizer",
    "DynamicQuantizationPass",
    "Evaluator",
    "GraphFusionPass",
    "LayoutConversionPass",
    "MixedPrecisionPass",
    "OliveModel",
    "OrtPerfTuningPass",
    "OrtTransformerOptimizationPass",
    "Pass",
    "PassContext",
    "PruningPass",
    "QuantizationPass",
    "StaticQuantizationPass",
    "Target",
    "WeightOnlyQuantizationPass",
]
__all__.extend(
    [
        "ConstantFoldingPass",
        "ExtractSymbolicShapesPass",
        "StripIdentityPass",
        "StripUnusedInitializersPass",
    ]
)
