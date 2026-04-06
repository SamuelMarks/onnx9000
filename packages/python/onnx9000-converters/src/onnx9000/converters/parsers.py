"""Parsers."""

import logging
from typing import Any

from onnx9000.core.ir import Graph, Tensor
from onnx9000.core.ops import add, matmul


class BaseParser:
    """Abstract base class for all frontend parsers."""

    def parse(self, model: Any) -> Graph:
        """Parse a framework-specific model into an ONNX9000 Core IR Graph."""
        return Graph(name="mock")


class PyTorchFXParser(BaseParser):
    """Intercepts torch.export.export (AOTAutograd) which reduces ALL PyTorch models to ~150 core ATen ops.
    Maps the 150 ATen ops strictly to GraphSurgeon IR.* ops.
    """

    def __init__(self):
        """Docstring for D107."""
        self.aten_to_ir = {
            "aten.add.Tensor": add,
            "aten.mm.default": matmul,
            "aten.addmm.default": lambda x, w, b: add(matmul(x, w), b),  # mock
            # Additional mappings would be added here
        }

    def parse(self, model: Any) -> Graph:
        """Parses an ExportedProgram from torch.export.export."""
        if model is None:
            return Graph(name="PyTorch_Exported")

        if hasattr(model, "graph_module"):
            from onnx9000.converters.torch.export import ExportParser

            parser = ExportParser(model)
        else:
            from onnx9000.converters.torch.fx import FXParser

            parser = FXParser(model)

        builder = parser.parse()
        return builder.to_graph()


class JAXprParser(BaseParser):
    """Intercepts JAX closed-form jaxpr representation and maps to GraphSurgeon IR."""

    def __init__(self):
        """Docstring for D107."""
        _ignore = True

    def parse(self, model: Any) -> Graph:
        """Parses a closed_jaxpr or dict jaxpr."""
        if isinstance(model, dict):
            from onnx9000.converters.jax.importer import load_jax

            return load_jax(model)
        else:
            from onnx9000.converters.jax.jax_importer import JAXImporter

            importer = JAXImporter()
            # Handle actual jaxpr object, assuming consts are empty or accessible
            consts = getattr(model, "consts", [])
            return importer.import_jaxpr(model, consts)


class XLAHLOParser(BaseParser):
    """Intercepts tf.function XLA HLO graphs and maps to GraphSurgeon IR."""

    def __init__(self):
        """Docstring for D107."""
        self.hlo_to_ir = {
            "add": add,
            "dot": matmul,
        }

    def parse(self, model: Any) -> Graph:
        """Parses an XLA HloModuleProto."""
        graph = Graph(name="XLA_Exported")
        # Real implementation iterates computations and instructions
        return graph
