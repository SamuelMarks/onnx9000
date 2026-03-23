"""Control Flow & Subgraph Translation logic for ONNX to C89 generation."""

from onnx9000.c_compiler.ast_builder import C89Builder
from onnx9000.core.ir import Node


def generate_if(b: C89Builder, node: Node, cond_name: str, then_branch: str, else_branch: str):
    """Generate C if/else block for ONNX If nodes."""
    b.emit(f"/* {node.op_type} */")
    b.emit(f"if ({cond_name}[0]) {{")
    b.push_indent()
    b.emit(f"/* Invoke then branch graph: {then_branch} */")
    b.emit(
        f"/* {then_branch}_predict(ctx, ...); */"
    )  # Note: pointer assignments need proper routing in complete subgraphs
    b.pop_indent()
    b.emit("} else {")
    b.push_indent()
    b.emit(f"/* Invoke else branch graph: {else_branch} */")
    b.emit(f"/* {else_branch}_predict(ctx, ...); */")
    b.pop_indent()
    b.emit("}")


def generate_loop(b: C89Builder, node: Node, max_trip_count: str, cond: str, body_graph: str):
    """Generate C while loop for ONNX Loop nodes."""
    b.emit(f"/* {node.op_type} */")
    b.emit("{")
    b.push_indent()
    b.emit("int64_t iter;")
    b.emit("int cond_state = 1;")
    b.emit(f"if ({cond}) cond_state = {cond}[0];")

    b.emit(
        f"for (iter = 0; iter < ({max_trip_count} ? {max_trip_count}[0] : 9223372036854775807LL) && cond_state; ++iter) {{"
    )
    b.push_indent()
    b.emit(f"/* Invoke loop body graph: {body_graph} */")
    b.emit(f"/* {body_graph}_predict(ctx, ...); */")
    b.pop_indent()
    b.emit("}")
    b.pop_indent()
    b.emit("}")
