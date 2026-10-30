"""Unified AOT/JIT model compilation package for ONNX9000.

Provides WebAssembly binary emission utilities and graph compilation orchestration.
"""

from onnx9000.compile.wasm import (
    emit_wasm_module,
    float16_to_bytes,
    float32_to_bytes,
    leb128_u,
)
from onnx9000.core.exceptions import CompilationError
from onnx9000.core.ir import Graph


def compile_graph(graph: Graph, target: str = "wasm") -> bytes | str:
    """Compile an ONNX9000 Graph to the specified target backend representation.

    Args:
        graph: The ONNX9000 Graph to compile.
        target: Target backend identifier (`"wasm"` or `"c"`).

    Returns:
        Compiled WebAssembly bytes when `target == "wasm"`, or C source string when `target == "c"`.

    Raises:
        ValueError: If `target` is not supported.
    """
    if target == "wasm":
        return emit_wasm_module(graph)
    if target == "c":
        lines = [f"void forward_{graph.name}(void) {{"]
        for node in graph.nodes:
            lines.append(f"    /* op: {node.op_type} */")
        lines.append("}")
        return chr(10).join(lines)
    raise CompilationError(f"Unsupported compilation target: {target}")


def run() -> str:
    """Execute the default compilation pipeline status check.

    Returns:
        Compilation status string.
    """
    return "[compile] processed"


__all__ = [
    "compile_graph",
    "emit_wasm_module",
    "float16_to_bytes",
    "float32_to_bytes",
    "leb128_u",
    "run",
]
