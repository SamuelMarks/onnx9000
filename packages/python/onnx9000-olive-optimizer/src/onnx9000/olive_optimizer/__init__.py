"""Olive hardware-aware quantization, pruning, and fusion pipeline for ONNX9000."""

from onnx9000.optimizer.olive import (
    AutoOptimizer,
    DynamicQuantizationPass,
    Evaluator,
    GraphFusionPass,
    LayoutConversionPass,
    MixedPrecisionPass,
    OliveModel,
    OrtPerfTuningPass,
    OrtTransformerOptimizationPass,
    Pass,
    PassContext,
    PruningPass,
    QuantizationPass,
    StaticQuantizationPass,
    Target,
    WeightOnlyQuantizationPass,
)


class OliveOptimizer:
    """High-level Olive optimization workflow controller."""

    def optimize_model(self, model: OliveModel, target: Target = Target.CPU) -> OliveModel:
        """Run automatic hardware-aware Olive optimization passes on `model`.

        Args:
            model: Input `OliveModel` wrapping an ONNX9000 `Graph`.
            target: Target hardware deployment profile.

        Returns:
            Optimized `OliveModel`.
        """
        auto = AutoOptimizer(passes=[GraphFusionPass()], target=target)
        return auto.optimize(model)

    def process(self, arg: str) -> str:
        """Process an Olive optimization target descriptor.

        Args:
            arg: Target descriptor string.

        Returns:
            Formatted Olive Optimizer status string.
        """
        return f"Olive Optimizer processed {arg}"


__all__ = [
    "AutoOptimizer",
    "DynamicQuantizationPass",
    "Evaluator",
    "GraphFusionPass",
    "LayoutConversionPass",
    "MixedPrecisionPass",
    "OliveModel",
    "OliveOptimizer",
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
