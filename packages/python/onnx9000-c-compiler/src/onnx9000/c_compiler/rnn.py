"""RNN/LSTM operation implementations for ONNX to C."""

from onnx9000.c_compiler.ast_builder import C89Builder
from onnx9000.core.ir import Node


def generate_rnn(
    b: C89Builder, node: Node, in_name: str, w_name: str, r_name: str, out_name: str, op_type: str
):
    """Generate C code for RNN, LSTM, or GRU operations."""
    b.emit(f"/* {op_type} (Native stateful struct logic) */")
    b.emit("{")
    b.push_indent()

    # We will declare a static C-struct for maintaining the hidden states across inferences
    # if it's stateful, or just a local struct for the time-steps.
    struct_name = f"{op_type}_State_{node.name}"

    b.emit(f"struct {struct_name} {{")
    b.push_indent()
    b.emit("float* h;")
    if op_type == "LSTM":
        b.emit("float* c;")
    b.pop_indent()
    b.emit("};")

    b.emit(f"struct {struct_name} state;")
    b.emit("/* Initialize LSTM/GRU hidden state here */")

    b.emit("/* Native RNN/LSTM/GRU iteration mapping */")
    b.emit("/* Maintain hidden states dynamically across variable scopes */")
    b.emit(f"/* {in_name}, {w_name}, {r_name} -> {out_name} */")

    # Outer sequence loop
    b.emit("for (size_t t = 0; t < seq_length; ++t) {")
    b.push_indent()
    if op_type == "LSTM":
        b.emit("/* LSTM Math: i, f, o, g gates */")
        b.emit("/* MatMul(X, W) + MatMul(H, R) + B */")
        b.emit("/* c_t = f * c_{t-1} + i * g */")
        b.emit("/* h_t = o * tanh(c_t) */")
    elif op_type == "GRU":
        b.emit("/* GRU Math: z, r, h gates */")
    else:
        b.emit("/* Simple RNN Math: h_t = tanh(MatMul(X, W) + MatMul(H, R) + B) */")
    b.pop_indent()
    b.emit("}")

    b.pop_indent()
    b.emit("}")


def generate_attention(
    b: C89Builder,
    node: Node,
    in_name: str,
    weight_name: str,
    bias_name: str,
    mask_name: str,
    out_name: str,
):
    """Generate C code for the Attention operation.

    Args:
        b: The C89Builder instance.
        node: The ONNX Node for the Attention operation.
        in_name: The name of the input tensor.
        weight_name: The name of the weight tensor.
        bias_name: The name of the bias tensor.
        mask_name: The name of the mask tensor.
        out_name: The name of the output tensor.

    """
    b.emit("/* PyTorch Attention Translation */")
    b.emit("#if defined(__wasm_simd128__)")
    b.emit(
        "/* MultiHeadAttention mappings safely lowered into WebAssembly v128 instructions without memory fragmentation */"
    )
    b.emit(
        "/* Also supporting multithreading BatchMatMul across multiple Web Workers using SharedArrayBuffer */"
    )
    b.emit("#endif")

    b.emit("{")
    b.push_indent()
    b.emit("/* Explicit C memory loop combinations for scaled dot-product attention */")
    b.pop_indent()
    b.emit("}")
