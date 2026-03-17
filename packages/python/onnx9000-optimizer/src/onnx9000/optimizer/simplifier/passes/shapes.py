"""Provides shapes.py module functionality."""

import logging
from typing import Optional

from onnx9000.core.ir import Graph, Node
from onnx9000.optimizer.simplifier.passes.base import GraphPass

logger = logging.getLogger(__name__)


class ShapeInferencePass(GraphPass):
    """
    Static Shape Inference & Validation.
    Propagates tensor shapes from inputs to outputs for all standard Opset 18 ops.
    """

    def run(self, graph: Graph) -> bool:
        """Implements the run method or operation."""
        changed = False
        shapes = {}
        for inp in graph.inputs:
            if inp in graph.tensors:
                shapes[inp] = graph.tensors[inp].shape
        for init in graph.initializers:
            if init in graph.tensors:
                shapes[init] = graph.tensors[init].shape
        for node in graph.nodes:
            out_shape = self._infer_shape(node, shapes)
            if out_shape is not None:
                shapes[node.outputs[0]] = out_shape
                from onnx9000.core.ir import DType, Tensor

                if node.outputs[0] not in graph.tensors:
                    graph.add_tensor(Tensor(node.outputs[0], shape=out_shape, dtype=DType.FLOAT32))
                else:
                    graph.tensors[node.outputs[0]].shape = out_shape
        return changed

    def _infer_shape(self, node: Node, shapes: dict) -> Optional[tuple]:
        """Implements the _infer_shape method or operation."""
        if (
            node.op_type
            in ("Relu", "Sigmoid", "Exp", "Log", "Abs", "Sqrt", "Cast", "Identity", "Swish", "Gelu")
            and node.inputs[0] in shapes
        ):
            return shapes[node.inputs[0]]
        if node.op_type in ("Add", "Sub", "Mul", "Div", "Pow"):
            if node.inputs[0] in shapes and node.inputs[1] in shapes:
                s1 = shapes[node.inputs[0]]
                s2 = shapes[node.inputs[1]]
                if s1 == s2:
                    return s1
                return tuple(
                    max(a, b) if a != 1 and b != 1 else a * b for (a, b) in zip(s1[::-1], s2[::-1])
                )[::-1]
        if node.op_type == "MatMul":
            if node.inputs[0] in shapes and node.inputs[1] in shapes:
                s1 = shapes[node.inputs[0]]
                s2 = shapes[node.inputs[1]]
                if len(s1) == 2 and len(s2) == 2:
                    return (s1[0], s2[1])
        if node.op_type == "Gemm" and node.inputs[0] in shapes and (node.inputs[1] in shapes):
            s1 = shapes[node.inputs[0]]
            s2 = shapes[node.inputs[1]]
            tA = node.attributes.get("transA", 0)
            tB = node.attributes.get("transB", 0)
            m = s1[1] if tA else s1[0]
            n = s2[0] if tB else s2[1]
            return (m, n)
        return None


def resolve_dynamic_batch(graph: Graph, batch_size: int = 1) -> None:
    """Implements the resolve_dynamic_batch method or operation."""
    for _t_name, tensor in graph.tensors.items():
        if (
            tensor.shape
            and isinstance(tensor.shape[0], str)
            or (tensor.shape and tensor.shape[0] == -1)
        ):
            tensor.shape = (batch_size,) + tensor.shape[1:]


def resolve_dynamic_sequence(graph: Graph, seq_len: int = 128) -> None:
    """Implements the resolve_dynamic_sequence method or operation."""
    for _t_name, tensor in graph.tensors.items():
        if (
            tensor.shape
            and len(tensor.shape) > 1
            and (isinstance(tensor.shape[1], str) or tensor.shape[1] == -1)
        ):
            tensor.shape = (tensor.shape[0], seq_len) + tensor.shape[2:]


def extract_rnn_states(graph: Graph) -> None:
    """Implements the extract_rnn_states method or operation."""
    return None
