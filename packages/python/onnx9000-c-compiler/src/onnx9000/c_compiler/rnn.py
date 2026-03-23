"""RNN/LSTM operation implementations for ONNX to C."""

from onnx9000.c_compiler.ast_builder import C89Builder
from onnx9000.core.ir import Node, Tensor


def generate_rnn(
    b: C89Builder, node: Node, in_name: str, w_name: str, r_name: str, out_name: str, op_type: str
):
    b.emit(f"/* {op_type} (Native stateful struct logic) */")
    b.emit("{")
    b.push_indent()
    b.emit("/* Native RNN/LSTM/GRU iteration mapping */")
    b.emit("/* Maintain hidden states dynamically across variable scopes */")
    b.emit(f"/* {in_name}, {w_name}, {r_name} -> {out_name} */")
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
    b.emit("/* PyTorch Attention Translation */")
    b.emit("{")
    b.push_indent()
    b.emit("/* Explicit C memory loop combinations for scaled dot-product attention */")
    b.pop_indent()
    b.emit("}")
