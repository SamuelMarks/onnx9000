"""
Shape Inference for Autograd

Propagates shapes through dynamically generated backward nodes to ensure
the resulting graph is valid and can be properly memory-mapped.
"""

from onnx9000.ir import Graph


def infer_backward_shapes(graph: Graph) -> None:
    """
    Infers shapes for the dynamically generated backward pass nodes.
    For simplicity in this engine, we assume gradient shapes perfectly
    mirror their corresponding forward activations and parameters.
    """
    # Iterate through nodes and infer shapes.
    # In a full ONNX implementation, this calls the type/shape inference C++ API.
    # Here we mock it by mapping grad_X to X.
    for node in graph.nodes:
        if node.op_type.endswith("Grad") or "bwd" in node.name:
            for out_name in node.outputs:
                if out_name.startswith("grad_"):
                    # Find the original tensor name
                    orig_name = out_name.replace("grad_", "").split("_wrt_")[0]
                    if orig_name in graph.tensors:
                        # Copy shape and dtype
                        orig_tensor = graph.tensors[orig_name]
                        if out_name not in graph.tensors:
                            import onnx9000

                            graph.tensors[out_name] = onnx9000.Tensor(
                                shape=orig_tensor.shape,
                                dtype=orig_tensor.dtype,
                                name=out_name,
                            )
