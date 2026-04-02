"""Layout optimization for onnx9000."""

from onnx9000.core.ir import Graph, Tensor


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
                    out_name = n.outputs[0] if isinstance(n.outputs[0], str) else n.outputs[0].name
                    out_t = graph.tensors.get(out_name)
                    if out_t and len(out_t.outputs) == 1:
                        consumer = out_t.outputs[0]
                        if consumer.op_type in ["Relu", "Sigmoid", "Tanh"]:
                            # Move transpose after pointwise op
                            # This is a basic step of layout propagation
                            new_t_name = f"{consumer.name}_layout_pushed"
                            consumer.inputs[0] = n.inputs[0]
                            # Create new transpose after consumer
                            from onnx9000.core.ir import Node, Attribute

                            new_node = Node(
                                "Transpose",
                                [consumer.outputs[0]],
                                [new_t_name],
                                {"perm": Attribute("perm", "INTS", [0, 2, 3, 1])},
                                f"{consumer.name}_transpose",
                            )
                            # Insert into graph
                            graph.nodes.insert(graph.nodes.index(consumer) + 1, new_node)
                            # Remove original transpose
                            graph.nodes.remove(n)
                            modified = True
                            break

    return graph
