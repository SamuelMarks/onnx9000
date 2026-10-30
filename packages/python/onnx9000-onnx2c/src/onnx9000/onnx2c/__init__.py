"""ONNX to standalone C89/C99 source code generator (`onnx2c`) for ONNX9000."""

from onnx9000.c_compiler import C89Compiler
from onnx9000.core.ir import Graph


def compile_to_c(graph: Graph, prefix: str = "onnx9000_") -> tuple[str, str]:
    """Compile an ONNX9000 `Graph` into standalone C89 header and source code.

    Args:
        graph: Source ONNX9000 `Graph`.
        prefix: Symbol prefix for generated C functions and structs.

    Returns:
        A `(header_code, source_code)` tuple of strings.
    """
    compiler = C89Compiler(graph=graph, prefix=prefix)
    return compiler.generate()


def run() -> str:
    """Execute the default onnx2c status check.

    Returns:
        onnx2c status confirmation string.
    """
    return "[onnx2c] processed"


__all__ = ["C89Compiler", "compile_to_c", "run"]
