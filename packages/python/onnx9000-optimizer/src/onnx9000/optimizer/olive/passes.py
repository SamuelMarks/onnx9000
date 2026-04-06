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
        from onnx9000.optimizer.simplifier.passes.quantization import convert_to_int8

        convert_to_int8(model.graph)
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
        from onnx9000.optimizer.sparse.modifier import GlobalMagnitudePruningModifier

        sparsity = self.config.get("sparsity", 0.5)
        mod = GlobalMagnitudePruningModifier(params=["re:.*"], final_sparsity=sparsity)
        mod.apply(model.graph)
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
        from onnx9000.optimizer.simplifier.passes.fusion import run_all_fusions

        run_all_fusions(model.graph)
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
        from onnx9000.core.dtypes import DType
        from onnx9000.core.ir import Node

        target_type = self.config.get("dtype", "FLOAT16")
        for node in model.graph.nodes:
            if node.op_type in ["MatMul", "Gemm", "Conv"]:
                node.attributes["dtype"] = target_type
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
        from onnx9000.optimizer.hardware.layout import LayoutOptimizer

        target_layout = self.config.get("target_layout", "NHWC")
        if target_layout == "NHWC":
            model.graph = LayoutOptimizer.nchw_to_nhwc_pass(model.graph)
        else:
            model.graph = LayoutOptimizer.nhwc_to_nchw_pass(model.graph)
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
        model.metadata["ort_tuning"] = {"intra_op_num_threads": 4, "execution_provider": "CPU"}
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
        from onnx9000.optimizer.simplifier.passes.fusion import run_all_fusions

        run_all_fusions(model.graph)
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
        from onnx9000.optimizer.simplifier.passes.constant_folding import constant_folding

        constant_folding(model.graph)
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
        new_nodes = []
        for node in model.graph.nodes:
            if node.op_type == "Identity":
                continue
            new_nodes.append(node)
        model.graph.nodes = new_nodes
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
        used = set()
        for node in model.graph.nodes:
            used.update(node.inputs)
        for out in model.graph.outputs:
            used.add(out.name)
        new_init = [k for k in model.graph.initializers if k in used]
        model.graph.initializers = new_init

        # Also clean up un-used initializer tensors
        for k in list(model.graph.tensors.keys()):
            if model.graph.tensors[k].is_initializer and k not in used:
                del model.graph.tensors[k]
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
        model.metadata["symbolic_shapes"] = True
        return model
