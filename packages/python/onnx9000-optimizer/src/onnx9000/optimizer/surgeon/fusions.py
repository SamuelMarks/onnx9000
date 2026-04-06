"""Graph fusion passes for onnx9000."""

from onnx9000.core.ir import Constant, Graph, Node, Tensor
from onnx9000.core.surgeon import PatternMatcher, match_pattern


def fuse_flash_attention(graph: Graph) -> Graph:
    """Identify and fuse standard attention subgraphs into FlashAttention.

    Pattern: Softmax( (Q @ K.T) / scale ) @ V
    """
    v_matmul_pattern = PatternMatcher(op_type="MatMul")

    matches = match_pattern(graph, v_matmul_pattern)
    for v_matmul in matches:
        softmax_node = None
        v_input = None
        for inp in v_matmul.inputs:
            if isinstance(inp, Tensor) and inp.inputs and inp.inputs[0].op_type == "Softmax":
                softmax_node = inp.inputs[0]
            else:
                v_input = inp

        if softmax_node:
            div_node = None
            softmax_input = softmax_node.inputs[0]
            if (
                isinstance(softmax_input, Tensor)
                and softmax_input.inputs
                and softmax_input.inputs[0].op_type == "Div"
            ):
                div_node = softmax_input.inputs[0]

            if div_node:
                qk_matmul = None
                div_input = div_node.inputs[0]
                if (
                    isinstance(div_input, Tensor)
                    and div_input.inputs
                    and div_input.inputs[0].op_type == "MatMul"
                ):
                    qk_matmul = div_input.inputs[0]

                if qk_matmul:
                    q = qk_matmul.inputs[0]
                    k = qk_matmul.inputs[1]
                    v = v_input

                    flash_attn = Node(
                        op_type="FlashAttention",
                        inputs=[q, k, v],
                        outputs=[v_matmul.outputs[0]],
                        name=f"FlashAttention_{v_matmul.name}",
                        domain="com.microsoft",
                    )

                    idx = graph.nodes.index(v_matmul)
                    graph.nodes[idx] = flash_attn

                    if qk_matmul in graph.nodes:
                        graph.nodes.remove(qk_matmul)
                    if div_node in graph.nodes:
                        graph.nodes.remove(div_node)
                    if softmax_node in graph.nodes:
                        graph.nodes.remove(softmax_node)

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
            primary_input = (
                node.inputs[0].name if hasattr(node.inputs[0], "name") else node.inputs[0]
            )
            if primary_input not in input_to_gemms:
                input_to_gemms[primary_input] = []
            input_to_gemms[primary_input].append(node)

    for inp_name, gemms in input_to_gemms.items():
        if len(gemms) < 2:
            continue

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
            and len(
                graph.tensors.get(
                    g.inputs[1].name if hasattr(g.inputs[1], "name") else g.inputs[1]
                ).shape
            )
            == 2
        ]

        if len(fusible) >= 2:
            import struct
            from onnx9000.core.dtypes import DType

            concat_data = b""
            shapes = []
            for g in fusible:
                weight_name = g.inputs[1].name if hasattr(g.inputs[1], "name") else g.inputs[1]
                weight_c = graph.tensors.get(weight_name)
                if weight_c.data is not None:
                    concat_data += bytes(weight_c.data)
                shapes.append(weight_c.shape)

            fused_weights_name = "_".join([g.name for g in fusible]) + "_fused_weights"
            fused_name = "_".join([g.name for g in fusible]) + "_fused"
            fused_out = fused_name + "_out"

            ref_weight_c = graph.tensors.get(
                fusible[0].inputs[1].name
                if hasattr(fusible[0].inputs[1], "name")
                else fusible[0].inputs[1]
            )

            fused_weight_c = Constant(
                name=fused_weights_name,
                values=concat_data,
                shape=(shapes[0][0], sum(s[1] for s in shapes)),
                dtype=ref_weight_c.dtype,
            )
            graph.add_tensor(fused_weight_c)

            split_node = Node(
                op_type="Split",
                inputs=[fused_out],
                outputs=[
                    g.outputs[0].name if hasattr(g.outputs[0], "name") else g.outputs[0]
                    for g in fusible
                ],
                attributes={"axis": 1, "split": [s[1] for s in shapes]},
                name=fused_name + "_split",
            )

            fused_node = Node(
                op_type="Gemm" if fusible[0].op_type == "Gemm" else "MatMul",
                inputs=[fusible[0].inputs[0], fused_weights_name],
                outputs=[fused_out],
                name=fused_name,
                attributes=fusible[0].attributes,
            )

            idx = graph.nodes.index(fusible[0])
            graph.nodes[idx] = fused_node
            graph.nodes.insert(idx + 1, split_node)

            for i in range(1, len(fusible)):
                graph.nodes.remove(fusible[i])

    return graph
