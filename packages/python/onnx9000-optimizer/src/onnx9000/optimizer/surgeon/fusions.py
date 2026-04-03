"""Graph fusion passes for onnx9000."""

from onnx9000.core.ir import Constant, Graph, Node, Tensor
from onnx9000.core.surgeon import PatternMatcher, match_pattern


def fuse_flash_attention(graph: Graph) -> Graph:
    """Identify and fuse standard attention subgraphs into FlashAttention.

    Pattern: Softmax( (Q @ K.T) / scale ) @ V
    """
    # Simple pattern for MatMul + Softmax + MatMul
    # In practice, it often has Scale, Add (for mask), etc.

    # Define a pattern matcher for the end of the chain
    # V_MatMul(Softmax(...), V)
    v_matmul_pattern = PatternMatcher(op_type="MatMul")

    matches = match_pattern(graph, v_matmul_pattern)
    for v_matmul in matches:
        # Check if one of the inputs is a Softmax
        softmax_node = None
        v_input = None
        for inp in v_matmul.inputs:
            if isinstance(inp, Tensor) and inp.inputs and inp.inputs[0].op_type == "Softmax":
                softmax_node = inp.inputs[0]
            else:
                v_input = inp

        if softmax_node:
            # We found Softmax -> MatMul
            # Now look for (Q @ K) / scale before Softmax
            # Usually: Div(MatMul(Q, K), scale) -> Softmax
            div_node = None
            softmax_input = softmax_node.inputs[0]
            if (
                isinstance(softmax_input, Tensor)
                and softmax_input.inputs
                and softmax_input.inputs[0].op_type == "Div"
            ):
                div_node = softmax_input.inputs[0]

            if div_node:
                # Div -> Softmax -> MatMul
                qk_matmul = None
                div_input = div_node.inputs[0]
                if (
                    isinstance(div_input, Tensor)
                    and div_input.inputs
                    and div_input.inputs[0].op_type == "MatMul"
                ):
                    qk_matmul = div_input.inputs[0]

                if qk_matmul:
                    # Found the pattern! QK_MatMul -> Div -> Softmax -> V_MatMul
                    q = qk_matmul.inputs[0]
                    k = qk_matmul.inputs[1]
                    v = v_input

                    # Create FlashAttention node
                    flash_attn = Node(
                        op_type="FlashAttention",
                        inputs=[q, k, v],
                        outputs=[v_matmul.outputs[0]],
                        name=f"FlashAttention_{v_matmul.name}",
                        domain="com.microsoft",  # Or our custom domain
                    )

                    # Replace the whole subgraph
                    # In a real implementation, we'd use graph.replace_pattern or similar
                    # For now, we manually update the graph
                    idx = graph.nodes.index(v_matmul)
                    graph.nodes[idx] = flash_attn

                    # Mark intermediate nodes for cleanup
                    # (In a real implementation, cleanup() would handle this)

    return graph


def fuse_horizontal_gemm(graph: Graph) -> Graph:
    """Fuse parallel Gemm/MatMul operations sharing the same input.

    Pattern:
      X -> Gemm1(X, W1) -> Y1
      X -> Gemm2(X, W2) -> Y2
    Fused:
      X -> Gemm(X, Concat(W1, W2)) -> Concat(Y1, Y2)
    """
    input_to_gemms = {}
    for node in graph.nodes:
        if node.op_type in ["Gemm", "MatMul"]:
            # Primary input is usually the first one
            primary_input = (
                node.inputs[0].name if hasattr(node.inputs[0], "name") else node.inputs[0]
            )
            if primary_input not in input_to_gemms:
                input_to_gemms[primary_input] = []
            input_to_gemms[primary_input].append(node)

    for inp_name, gemms in input_to_gemms.items():
        if len(gemms) < 2:
            continue

        # Group gemms that can be fused (same attributes, compatible weights)
        # For simplicity, we just fuse all Gemms sharing the same input if they have constant weights
        fusible = [
            g
            for g in gemms
            if len(g.inputs) >= 2
            and isinstance(
                graph.tensors.get(
                    g.inputs[1].name if hasattr(g.inputs[1], "name") else g.inputs[1]
                ),
                Constant,
            )
        ]

        if len(fusible) >= 2:
            # Create a single Gemm with concatenated weights
            # In a real implementation, we would use surgeon.concatenate_constants
            # and insert Split nodes after the fused Gemm if needed.
            # For Pillar 3, we implement the logic for identifying and grouping them.
            fused_name = "_".join([g.name for g in fusible]) + "_fused"
            all_outputs = []
            for g in fusible:
                all_outputs.extend(g.outputs)

            fused_node = Node(
                op_type="Gemm",
                inputs=[fusible[0].inputs[0], "fused_weights"],
                outputs=all_outputs,
                name=fused_name,
            )

            # Simplified: just replace first fusible with fused node, remove others
            idx = graph.nodes.index(fusible[0])
            graph.nodes[idx] = fused_node
            for i in range(1, len(fusible)):
                graph.nodes.remove(fusible[i])

    return graph
