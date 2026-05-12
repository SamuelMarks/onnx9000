"""This module exposes the main components for scripting ONNX graphs.

It provides the GraphBuilder, operational definitions, script parser, and
symbolic variables used to construct and manipulate ONNX models programmatically.
"""

from onnx9000.core.ir import Graph
from onnx9000.toolkit.script.builder import GraphBuilder
from onnx9000.toolkit.script.op import op
from onnx9000.toolkit.script.parser import script
from onnx9000.toolkit.script.var import Var


def parse_and_compile(script_path: str) -> Graph:
    """Parses a Python file containing a @script decorated function and compiles it."""
    import runpy

    module = runpy.run_path(script_path)
    # Find the first decorated script function
    for name, obj in module.items():
        if hasattr(obj, "_is_onnx_script"):
            return obj.to_builder().build()
    raise ValueError(f"No @script decorated function found in {script_path}")


__all__ = ["GraphBuilder", "op", "script", "Var", "parse_and_compile"]
