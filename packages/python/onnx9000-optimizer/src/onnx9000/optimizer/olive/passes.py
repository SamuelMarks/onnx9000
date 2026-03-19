"""Provides pure-Python Pass base class and subclasses."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from onnx9000.optimizer.olive.context import PassContext
from onnx9000.optimizer.olive.model import OliveModel


class Pass(ABC):
    """Base class for Olive optimization passes."""

    def __init__(self, name: str, config: Optional[dict[str, Any]] = None) -> None:
        """Initialize the pass."""
        self.name = name
        self.config = config or {}

    @abstractmethod
    def run(self, model: OliveModel, context: PassContext) -> OliveModel:
        """Run the pass on the model."""


class QuantizationPass(Pass):
    """Base pass for quantization."""

    def __init__(
        self, name: str = "QuantizationPass", config: Optional[dict[str, Any]] = None
    ) -> None:
        """Initialize the pass."""
        super().__init__(name, config)

    def run(self, model: OliveModel, context: PassContext) -> OliveModel:
        """Run the pass on the model."""
        return model


class DynamicQuantizationPass(QuantizationPass):
    """Pass for dynamic quantization."""

    def __init__(
        self, name: str = "DynamicQuantizationPass", config: Optional[dict[str, Any]] = None
    ) -> None:
        """Initialize the pass."""
        super().__init__(name, config)


class StaticQuantizationPass(QuantizationPass):
    """Pass for static quantization (with calibration logic)."""

    def __init__(
        self, name: str = "StaticQuantizationPass", config: Optional[dict[str, Any]] = None
    ) -> None:
        """Initialize the pass."""
        super().__init__(name, config)


class WeightOnlyQuantizationPass(QuantizationPass):
    """Pass for weight-only quantization."""

    def __init__(
        self, name: str = "WeightOnlyQuantizationPass", config: Optional[dict[str, Any]] = None
    ) -> None:
        """Initialize the pass."""
        super().__init__(name, config)


class PruningPass(Pass):
    """Pass for Sparsity pruning."""

    def __init__(self, name: str = "PruningPass", config: Optional[dict[str, Any]] = None) -> None:
        """Initialize the pass."""
        super().__init__(name, config)

    def run(self, model: OliveModel, context: PassContext) -> OliveModel:
        """Run the pass on the model."""
        return model


class GraphFusionPass(Pass):
    """Pass for graph fusion."""

    def __init__(
        self, name: str = "GraphFusionPass", config: Optional[dict[str, Any]] = None
    ) -> None:
        """Initialize the pass."""
        super().__init__(name, config)

    def run(self, model: OliveModel, context: PassContext) -> OliveModel:
        """Run the pass on the model."""
        return model


class MixedPrecisionPass(Pass):
    """Pass for mixed precision (FP16 / BFloat16)."""

    def __init__(
        self, name: str = "MixedPrecisionPass", config: Optional[dict[str, Any]] = None
    ) -> None:
        """Initialize the pass."""
        super().__init__(name, config)

    def run(self, model: OliveModel, context: PassContext) -> OliveModel:
        """Run the pass on the model."""
        return model


class LayoutConversionPass(Pass):
    """Pass for LayoutConversion (NCHW <-> NHWC)."""

    def __init__(
        self, name: str = "LayoutConversionPass", config: Optional[dict[str, Any]] = None
    ) -> None:
        """Initialize the pass."""
        super().__init__(name, config)

    def run(self, model: OliveModel, context: PassContext) -> OliveModel:
        """Run the pass on the model."""
        return model


class OrtPerfTuningPass(Pass):
    """Pass for OrtPerfTuning (Thread/EP tuning suggestions)."""

    def __init__(
        self, name: str = "OrtPerfTuningPass", config: Optional[dict[str, Any]] = None
    ) -> None:
        """Initialize the pass."""
        super().__init__(name, config)

    def run(self, model: OliveModel, context: PassContext) -> OliveModel:
        """Run the pass on the model."""
        return model


class OrtTransformerOptimizationPass(Pass):
    """Pass for OrtTransformerOptimization (Attention/Gelu fusion)."""

    def __init__(
        self, name: str = "OrtTransformerOptimizationPass", config: Optional[dict[str, Any]] = None
    ) -> None:
        """Initialize the pass."""
        super().__init__(name, config)

    def run(self, model: OliveModel, context: PassContext) -> OliveModel:
        """Run the pass on the model."""
        return model


class ConstantFoldingPass(Pass):
    """Run mathematical constant folding explicitly before quantization."""

    def __init__(
        self, name: str = "ConstantFoldingPass", config: Optional[dict[str, Any]] = None
    ) -> None:
        """Initialize the pass."""
        super().__init__(name, config)

    def run(self, model: OliveModel, context: PassContext) -> OliveModel:
        """Run the pass on the model."""
        return model


class StripIdentityPass(Pass):
    """Strip Identity nodes explicitly before quantization."""

    def __init__(
        self, name: str = "StripIdentityPass", config: Optional[dict[str, Any]] = None
    ) -> None:
        """Initialize the pass."""
        super().__init__(name, config)

    def run(self, model: OliveModel, context: PassContext) -> OliveModel:
        """Run the pass on the model."""
        return model


class StripUnusedInitializersPass(Pass):
    """Strip un-used initializers explicitly before quantization."""

    def __init__(
        self, name: str = "StripUnusedInitializersPass", config: Optional[dict[str, Any]] = None
    ) -> None:
        """Initialize the pass."""
        super().__init__(name, config)

    def run(self, model: OliveModel, context: PassContext) -> OliveModel:
        """Run the pass on the model."""
        return model


class ExtractSymbolicShapesPass(Pass):
    """Extract symbolic shapes to validate layout transformations safely."""

    def __init__(
        self, name: str = "ExtractSymbolicShapesPass", config: Optional[dict[str, Any]] = None
    ) -> None:
        """Initialize the pass."""
        super().__init__(name, config)

    def run(self, model: OliveModel, context: PassContext) -> OliveModel:
        """Run the pass on the model."""
        return model
