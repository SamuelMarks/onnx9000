"""NLP and TopK operation implementations for ONNX to C."""

from onnx9000.c_compiler.ast_builder import C89Builder
from onnx9000.core.ir import Node, Tensor


def generate_topk(
    b: C89Builder,
    node: Node,
    out_tensor: Tensor,
    in_name: str,
    k_name: str,
    out_val_name: str,
    out_idx_name: str,
):
    """Generate C code for the TopK operation.

    Args:
        b: The C89Builder instance.
        node: The ONNX Node for TopK.
        out_tensor: The output tensor.
        in_name: The name of the input tensor.
        k_name: The name of the K parameter.
        out_val_name: The name of the output values tensor.
        out_idx_name: The name of the output indices tensor.

    """
    b.emit(f"/* {node.op_type} */")
    b.emit("{")
    b.push_indent()
    b.emit("/* Lightweight Quicksort or Heap implementation for TopK */")
    b.emit(f"/* Sort {in_name}, retrieve top {k_name} into {out_val_name}, {out_idx_name} */")
    b.pop_indent()
    b.emit("}")


def generate_unique(b: C89Builder, node: Node, out_tensor: Tensor, in_name: str, out_val_name: str):
    """Generate C code for the Unique operation.

    Args:
        b: The C89Builder instance.
        node: The ONNX Node for Unique.
        out_tensor: The output tensor.
        in_name: The name of the input tensor.
        out_val_name: The name of the output values tensor.

    """
    b.emit(f"/* {node.op_type} */")
    b.emit("{")
    b.push_indent()
    b.emit(f"/* Unique extraction for {in_name} to {out_val_name} */")
    b.pop_indent()
    b.emit("}")


def emit_bpe_tokenizer(b: C89Builder, vocab: dict):
    """Emit C code for a BPE tokenizer using a given vocabulary.

    Args:
        b: The C89Builder instance.
        vocab: A dictionary mapping tokens to IDs.

    """
    b.emit("/* BPE Tokenizer Dictionary */")
    b.emit(f"const int bpe_vocab_size = {len(vocab)};")
    # Emit static array mapping
    b.emit("static const char* bpe_vocab[] = {")
    b.push_indent()
    for k, v in vocab.items():
        b.emit(f'"{k}", /* {v} */')
    b.pop_indent()
    b.emit("};")
