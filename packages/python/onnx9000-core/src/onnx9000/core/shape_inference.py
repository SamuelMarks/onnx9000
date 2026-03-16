"""Static shape inference module."""

from typing import Union
from onnx9000.core.exceptions import ShapeInferenceError
from onnx9000.core.ir import DynamicDim, Graph, Tensor, ValueInfo
from onnx9000.core.utils import topological_sort


def infer_shapes_and_types(graph: Graph) -> None:
    """
    Performs static shape and type inference on the given graph.
    Updates the graph.tensors and graph.value_info intrinsically.
    """
    try:
        sorted_nodes = topological_sort(graph)
    except Exception as e:
        raise ShapeInferenceError(f"Cannot infer shapes on a cyclic graph: {e}") from e
    env: dict[str, ValueInfo] = {}
    for inp in graph.inputs:
        if isinstance(inp, str) and inp in graph.tensors:
            t = graph.tensors[inp]
            env[inp] = ValueInfo(t.name, t.shape, t.dtype)
        elif hasattr(inp, "name"):
            env[inp.name] = inp
    for _tensor_name, tensor in graph.tensors.items():
        if tensor.is_initializer:
            env[tensor.name] = ValueInfo(tensor.name, tensor.shape, tensor.dtype)
    for node in sorted_nodes:
        if (
            node.op_type == "Add"
            or node.op_type == "Sub"
            or node.op_type == "Mul"
            or (node.op_type == "Div")
        ):
            if len(node.inputs) < 2:
                continue
            in1 = env.get(node.inputs[0])
            in2 = env.get(node.inputs[1])
            if not in1 or not in2:
                continue
            out_shape = in1.shape if len(in1.shape) >= len(in2.shape) else in2.shape
            for out_name in node.outputs:
                out_info = ValueInfo(out_name, out_shape, in1.dtype)
                env[out_name] = out_info
                if out_name not in graph.tensors:
                    graph.add_tensor(Tensor(out_name, out_shape, in1.dtype))
        elif node.op_type == "MatMul":
            if len(node.inputs) < 2:
                continue
            in1 = env.get(node.inputs[0])
            in2 = env.get(node.inputs[1])
            if not in1 or not in2:
                continue
            shape1 = in1.shape
            shape2 = in2.shape
            out_shape: tuple[Union[int, DynamicDim], ...] = ()
            if len(shape1) >= 2 and len(shape2) >= 2:
                out_shape = tuple(list(shape1[:-1]) + [shape2[-1]])
            for out_name in node.outputs:
                out_info = ValueInfo(out_name, out_shape, in1.dtype)
                env[out_name] = out_info
                if out_name not in graph.tensors:
                    graph.add_tensor(Tensor(out_name, out_shape, in1.dtype))
        elif node.op_type in ["Relu", "Sigmoid", "Tanh", "Exp", "Log"]:
            if len(node.inputs) < 1:
                continue
            in1 = env.get(node.inputs[0])
            if in1:
                for out_name in node.outputs:
                    out_info = ValueInfo(out_name, in1.shape, in1.dtype)
                    env[out_name] = out_info
                    if out_name not in graph.tensors:
                        graph.add_tensor(Tensor(out_name, in1.shape, in1.dtype))
