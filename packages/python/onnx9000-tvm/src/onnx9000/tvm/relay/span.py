from dataclasses import dataclass
from typing import Optional


@dataclass
class Span:
    """Tracks source map from ONNX node to IR node."""

    source_name: str
    line: int
    column: int
    onnx_node_name: Optional[str] = None
    onnx_op_type: Optional[str] = None


def set_span(expr, span: Span):
    """Sets the span of an expression for source map tracking."""
    expr.span = span
    return expr
