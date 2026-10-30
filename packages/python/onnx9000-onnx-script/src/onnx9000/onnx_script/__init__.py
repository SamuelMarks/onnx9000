"""Python DSL (`@script`, `GraphBuilder`, `op`, `Var`) for authoring ONNX9000 graphs."""

from onnx9000.toolkit.script import GraphBuilder, Var, op, parse_and_compile, script


def run() -> str:
    """Execute the default ONNX-Script status check.

    Returns:
        ONNX-Script status confirmation string.
    """
    return "[onnx-script] processed"


__all__ = ["GraphBuilder", "Var", "op", "parse_and_compile", "run", "script"]
