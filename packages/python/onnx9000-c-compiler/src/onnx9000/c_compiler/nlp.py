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
    b.emit(f"/* {node.op_type} */")
    b.emit("{")
    b.push_indent()
    b.emit("/* Lightweight Quicksort or Heap implementation for TopK */")
    b.emit(f"/* Sort {in_name}, retrieve top {k_name} into {out_val_name}, {out_idx_name} */")
    b.pop_indent()
    b.emit("}")


def generate_unique(b: C89Builder, node: Node, out_tensor: Tensor, in_name: str, out_val_name: str):
    b.emit(f"/* {node.op_type} */")
    b.emit("{")
    b.push_indent()
    b.emit(f"/* Unique extraction for {in_name} to {out_val_name} */")
    b.pop_indent()
    b.emit("}")


def emit_bpe_tokenizer(b: C89Builder, vocab: dict):
    b.emit("/* BPE Tokenizer Dictionary */")
    b.emit(f"const int bpe_vocab_size = {len(vocab)};")
    # Emit static array mapping
    b.emit("static const char* bpe_vocab[] = {")
    b.push_indent()
    for k, v in vocab.items():
        b.emit(f'"{k}", /* {v} */')
    b.pop_indent()
    b.emit("};")
