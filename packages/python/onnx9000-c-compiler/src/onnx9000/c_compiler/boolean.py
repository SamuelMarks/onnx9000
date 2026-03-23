"""Logical, Relational & Boolean operations for ONNX to C89 generation."""

from onnx9000.c_compiler.ast_builder import C89Builder
from onnx9000.c_compiler.operations import resolve_broadcast_indices
from onnx9000.core.ir import Node, Tensor
from onnx9000.core.profiler import resolve_volume


def generate_boolean_binary(
    b: C89Builder,
    node: Node,
    op_char: str,
    out_tensor: Tensor,
    in1_tensor: Tensor,
    in2_tensor: Tensor,
    in1: str,
    in2: str,
    out: str,
):
    """Generate Equal, Less, Greater, And, Or, etc."""
    b.emit(f"/* Boolean {node.op_type} */")
    b.emit("{")
    b.push_indent()
    b.emit("int i;")
    size_var = str(resolve_volume(out_tensor.shape)) if out_tensor and out_tensor.shape else "1"

    b.emit(f"for (i = 0; i < {size_var}; ++i) {{")
    b.push_indent()

    idx1 = resolve_broadcast_indices(out_tensor.shape, in1_tensor.shape if in1_tensor else [])
    idx2 = resolve_broadcast_indices(out_tensor.shape, in2_tensor.shape if in2_tensor else [])

    # Cast bool to uint8_t
    if node.op_type in ["And", "Or", "Xor"]:
        # Logic operators assume bool (uint8_t mapping)
        if node.op_type == "Xor":
            b.emit(f"{out}[i] = ({in1}[{idx1}] != {in2}[{idx2}]) ? 1 : 0;")
        else:
            b.emit(f"{out}[i] = ({in1}[{idx1}] {op_char} {in2}[{idx2}]) ? 1 : 0;")
    else:
        # Relational operators on floats
        b.emit(f"{out}[i] = ({in1}[{idx1}] {op_char} {in2}[{idx2}]) ? 1 : 0;")

    b.pop_indent()
    b.emit("}")
    b.pop_indent()
    b.emit("}")


def generate_boolean_unary(
    b: C89Builder,
    node: Node,
    op_char: str,
    out_tensor: Tensor,
    in1_tensor: Tensor,
    in1: str,
    out: str,
):
    """Generate Not."""
    b.emit(f"/* Boolean {node.op_type} */")
    b.emit("{")
    b.push_indent()
    b.emit("int i;")
    size_var = str(resolve_volume(out_tensor.shape)) if out_tensor and out_tensor.shape else "1"

    b.emit(f"for (i = 0; i < {size_var}; ++i) {{")
    b.push_indent()
    idx1 = resolve_broadcast_indices(out_tensor.shape, in1_tensor.shape if in1_tensor else [])
    b.emit(f"{out}[i] = {op_char}({in1}[{idx1}]) ? 1 : 0;")
    b.pop_indent()
    b.emit("}")
    b.pop_indent()
    b.emit("}")


def generate_where(
    b: C89Builder,
    node: Node,
    out_tensor: Tensor,
    cond_tensor: Tensor,
    x_tensor: Tensor,
    y_tensor: Tensor,
    cond: str,
    x: str,
    y: str,
    out: str,
):
    """Generate Where."""
    b.emit(f"/* {node.op_type} */")
    b.emit("{")
    b.push_indent()
    b.emit("int i;")
    size_var = str(resolve_volume(out_tensor.shape)) if out_tensor and out_tensor.shape else "1"

    b.emit(f"for (i = 0; i < {size_var}; ++i) {{")
    b.push_indent()
    idx_cond = resolve_broadcast_indices(out_tensor.shape, cond_tensor.shape if cond_tensor else [])
    idx_x = resolve_broadcast_indices(out_tensor.shape, x_tensor.shape if x_tensor else [])
    idx_y = resolve_broadcast_indices(out_tensor.shape, y_tensor.shape if y_tensor else [])

    b.emit(f"{out}[i] = {cond}[{idx_cond}] ? {x}[{idx_x}] : {y}[{idx_y}];")
    b.pop_indent()
    b.emit("}")
    b.pop_indent()
    b.emit("}")
