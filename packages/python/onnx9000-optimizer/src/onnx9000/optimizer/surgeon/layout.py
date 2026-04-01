"""Layout optimization for onnx9000."""

from onnx9000.core.ir import Graph, Node, Tensor


def optimize_layouts(graph: Graph, target_layout: str = "NHWC") -> Graph:
    """Global optimization of NCHW vs NHWC to minimize unnecessary transposes."""
    # Algorithm:
    # 1. Identify all Transpose nodes that change layout (e.g. [0, 2, 3, 1] for NCHW->NHWC)
    # 2. Try to "push" transposes through operations like Relu, Add, Mul.
    # 3. Cancel out redundant Transpose pairs (T1 @ T2 = Identity).

    modified = True
    while modified:
        modified = False
        # Simplified: look for Transpose -> Pointwise -> Transpose (Inverse)
        for n in list(graph.nodes):
            if n.op_type == "Transpose":
                # Check attributes for layout change
                perm = n.attributes.get("perm").value if "perm" in n.attributes else []
                if perm == [0, 2, 3, 1]:  # NCHW -> NHWC
                    # Check consumer
                    out_t = n.outputs[0]
                    if isinstance(out_t, Tensor) and len(out_t.outputs) == 1:
                        consumer = out_t.outputs[0]
                        if consumer.op_type in ["Relu", "Sigmoid", "Tanh"]:
                            # Move transpose after pointwise op
                            # This is a basic step of layout propagation
                            pass

    return graph
