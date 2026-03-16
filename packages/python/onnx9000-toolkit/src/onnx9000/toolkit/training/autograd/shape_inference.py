"""
Shape Inference for Autograd

Propagates shapes through dynamically generated backward nodes to ensure
the resulting graph is valid and can be properly memory-mapped.
"""

from onnx9000.core.ir import Graph


def infer_backward_shapes(graph: Graph) -> None:
    """
    Infers shapes for the dynamically generated backward pass nodes.
    For simplicity in this engine, we assume gradient shapes perfectly
    mirror their corresponding forward activations and parameters.
    """
    for node in graph.nodes:
        if node.op_type.endswith("Grad") or "bwd" in node.name:
            for out_name in node.outputs:
                if out_name.startswith("grad_"):
                    orig_name = out_name.replace("grad_", "").split("_wrt_")[0]
                    if orig_name in graph.tensors:
                        orig_tensor = graph.tensors[orig_name]
                        if out_name not in graph.tensors:
                            from onnx9000.core.ir import Tensor

                            graph.tensors[out_name] = Tensor(
                                shape=orig_tensor.shape, dtype=orig_tensor.dtype, name=out_name
                            )
