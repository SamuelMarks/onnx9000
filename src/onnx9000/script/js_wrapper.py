"""
JS Wrapper for Pyodide environments.
Allows JavaScript developers to use the builder fluently:
const builder = new onnx9000.GraphBuilder('MyGraph');
"""

from onnx9000.script.builder import GraphBuilder
from onnx9000.core.dtypes import DType
import typing


class JSGraphBuilder:
    """Provides a JavaScript-friendly interface for building ONNX graphs."""

    def __init__(self, name: str):
        """Initializes the JSGraphBuilder with a given graph name."""
        self._builder = GraphBuilder(name)

    def add_input(self, name: str, dtype_str: str, shape: typing.List[int]) -> None:
        """Adds an input tensor to the graph with a specified data type and shape."""
        dtype = getattr(DType, dtype_str.upper())
        self._builder.add_input(name, dtype, tuple(shape))

    def add_output(self, name: str) -> None:
        """Adds an output tensor to the graph by its variable name."""
        from onnx9000.script.var import Var

        self._builder.add_output(Var(name))

    def build_to_bytes(self) -> bytes:
        """Builds the ONNX graph and serializes it to a byte array."""
        model = self._builder.to_onnx()
        return model.SerializeToString()
