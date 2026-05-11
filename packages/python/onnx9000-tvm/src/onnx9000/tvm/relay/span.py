"""TVM submodule for AST and optimization."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Span:
    """Tracks source map from ONNX node to IR node."""

    source_name: str
    line: int
    column: int
    onnx_node_name: str | None = None
    onnx_op_type: str | None = None


def set_span(expr, span: Span):
    """Set the span of an expression for source map tracking."""
    expr.span = span
    return expr
