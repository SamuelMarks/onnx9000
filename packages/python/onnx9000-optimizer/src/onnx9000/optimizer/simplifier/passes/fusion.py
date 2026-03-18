"""Provides fusion.py module functionality."""

import logging
from typing import Optional

from onnx9000.core.ir import Graph, Node
from onnx9000.optimizer.simplifier.passes.base import GraphPass

logger = logging.getLogger(__name__)


class FusionPass(GraphPass):
    """
    Operator Fusion Passes.
    Combines adjacent operations into fused kernels for improved memory
    locality and performance.
    """

    def run(self, graph: Graph) -> bool:
        """Implements the run method or operation."""
        changed = False
        while True:
            local_changed = self._run_once(graph)
            if not local_changed:
                break
            changed = True
        return changed

    def _run_once(self, graph: Graph) -> bool:
        """Implements the _run_once method or operation."""
        return False


class PatternMatcherFusion(FusionPass):
    """
    A Pattern Matching DSL for Operator Fusion.
    """

    def _run_once(self, graph: Graph) -> bool:
        """Implements the _run_once method or operation."""
        changed = False
        usages = {}
        for node in graph.nodes:
            for inp in node.inputs:
                usages[inp] = usages.get(inp, 0) + 1
        for out in graph.outputs:
            usages[out] = usages.get(out, 0) + 1

        def match_chain(start_node: Node, ops: list[str]) -> Optional[list[Node]]:
            """Implements the match_chain method or operation."""
            chain = [start_node]
            curr = start_node
            if start_node.op_type != ops[0]:
                return None
            for expected_op in ops[1:]:
                consumers = [n for n in graph.nodes if curr.outputs[0] in n.inputs]
                if not consumers:
                    return None
                next_node = None
                for c in consumers:
                    if c.op_type == expected_op:
                        next_node = c
                        break
                if not next_node:
                    return None
                chain.append(next_node)
                curr = next_node
            return chain

        i = 0
        while i < len(graph.nodes):
            node = graph.nodes[i]
            chain = match_chain(node, ["MatMul", "Add"])
            if chain:
                (mm, add) = chain
                b_idx = 1 if add.inputs[0] == mm.outputs[0] else 0
                bias = add.inputs[b_idx]
                gemm_node = Node(
                    op_type="Gemm",
                    inputs=[mm.inputs[0], mm.inputs[1], bias],
                    outputs=add.outputs,
                    attributes={"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 0},
                    name=f"FusedGemm_{mm.name}",
                )
                idx = graph.nodes.index(mm)
                graph.nodes[idx] = gemm_node
                graph.nodes.remove(add)
                return True
            chain = match_chain(node, ["Sigmoid", "Mul"])
            if chain:
                (sig, mul) = chain
                x_input = sig.inputs[0]
                if x_input in mul.inputs and sig.outputs[0] in mul.inputs:
                    swish_node = Node(
                        op_type="Swish",
                        inputs=[x_input],
                        outputs=mul.outputs,
                        attributes={},
                        name=f"FusedSwish_{sig.name}",
                    )
                    idx = graph.nodes.index(sig)
                    graph.nodes[idx] = swish_node
                    graph.nodes.remove(mul)
                    return True
            chain = match_chain(node, ["Exp", "ReduceSum", "Div"])
            if chain:
                (exp, rsum, div) = chain
                if (
                    exp.outputs[0] in rsum.inputs
                    and exp.outputs[0] in div.inputs
                    and (rsum.outputs[0] in div.inputs)
                ):
                    softmax_node = Node(
                        op_type="Softmax",
                        inputs=[exp.inputs[0]],
                        outputs=div.outputs,
                        attributes={
                            "axis": rsum.attributes.get("axes", [-1])[0]
                            if "axes" in rsum.attributes
                            else -1
                        },
                        name=f"FusedSoftmax_{exp.name}",
                    )
                    idx = graph.nodes.index(exp)
                    graph.nodes[idx] = softmax_node
                    graph.nodes.remove(rsum)
                    graph.nodes.remove(div)
                    return True
            chain = match_chain(node, ["Sub", "Pow", "ReduceMean", "Add", "Div", "Mul", "Add"])
            if chain:
                (sub, pw, rm, add1, div, mul, add2) = chain
                ln_node = Node(
                    op_type="LayerNormalization",
                    inputs=[
                        sub.inputs[0],
                        mul.inputs[1] if mul.inputs[0] == div.outputs[0] else mul.inputs[0],
                        add2.inputs[1] if add2.inputs[0] == mul.outputs[0] else add2.inputs[0],
                    ],
                    outputs=add2.outputs,
                    attributes={},
                    name=f"FusedLayerNorm_{sub.name}",
                )
                idx = graph.nodes.index(sub)
                graph.nodes[idx] = ln_node
                for n in chain[1:]:
                    graph.nodes.remove(n)
                return True
            chain = match_chain(node, ["Div", "Erf", "Add", "Mul", "Mul"])
            if chain:
                (div, erf, add, mul1, mul2) = chain
                gelu_node = Node(
                    op_type="Gelu",
                    inputs=[div.inputs[0]],
                    outputs=mul2.outputs,
                    attributes={},
                    name=f"FusedGelu_{div.name}",
                )
                idx = graph.nodes.index(div)
                graph.nodes[idx] = gelu_node
                for n in chain[1:]:
                    graph.nodes.remove(n)
                return True
            chain = match_chain(node, ["Conv", "BatchNormalization"])
            if chain:
                (conv, bn) = chain
                c_bn_node = Node(
                    op_type="ConvBatchNorm",
                    inputs=conv.inputs + bn.inputs[1:],
                    outputs=bn.outputs,
                    attributes=conv.attributes,
                    name=f"FusedConvBN_{conv.name}",
                )
                idx = graph.nodes.index(conv)
                graph.nodes[idx] = c_bn_node
                graph.nodes.remove(bn)
                return True
            chain = match_chain(node, ["BatchNormalization", "Relu"])
            if chain:
                (bn, relu) = chain
                bn_relu = Node(
                    op_type="BatchNormalizationRelu",
                    inputs=bn.inputs,
                    outputs=relu.outputs,
                    attributes=bn.attributes,
                    name=f"FusedBNRelu_{bn.name}",
                )
                idx = graph.nodes.index(bn)
                graph.nodes[idx] = bn_relu
                graph.nodes.remove(relu)
                return True
            chain = match_chain(node, ["MatMul", "Relu"])
            if chain:
                (mm, relu) = chain
                mm_relu = Node(
                    op_type="MatMulRelu",
                    inputs=mm.inputs,
                    outputs=relu.outputs,
                    attributes=mm.attributes,
                    name=f"FusedMatMulRelu_{mm.name}",
                )
                idx = graph.nodes.index(mm)
                graph.nodes[idx] = mm_relu
                graph.nodes.remove(relu)
                return True
            chain = match_chain(node, ["Gemm", "BatchNormalization"])
            if chain:
                (gm, bn) = chain
                gbn = Node(
                    op_type="GemmBatchNorm",
                    inputs=gm.inputs + bn.inputs[1:],
                    outputs=bn.outputs,
                    attributes=gm.attributes,
                    name=f"FusedGemmBN_{gm.name}",
                )
                idx = graph.nodes.index(gm)
                graph.nodes[idx] = gbn
                graph.nodes.remove(bn)
                return True
            chain = match_chain(node, ["Conv", "Add"])
            if chain:
                (conv, add) = chain
                if len(conv.inputs) == 2:
                    b_idx = 1 if add.inputs[0] == conv.outputs[0] else 0
                    conv.inputs.append(add.inputs[b_idx])
                    conv.outputs[0] = add.outputs[0]
                    graph.nodes.remove(add)
                    return True
            chain = match_chain(node, ["Conv", "Mul"])
            if chain:
                (conv, mul) = chain
                cm = Node(
                    op_type="ConvMul",
                    inputs=conv.inputs
                    + [mul.inputs[1] if mul.inputs[0] == conv.outputs[0] else mul.inputs[0]],
                    outputs=mul.outputs,
                    attributes=conv.attributes,
                    name=f"FusedConvMul_{conv.name}",
                )
                idx = graph.nodes.index(conv)
                graph.nodes[idx] = cm
                graph.nodes.remove(mul)
                return True
            chain = match_chain(node, ["ConvTranspose", "BatchNormalization"])
            if chain:
                (ct, bn) = chain
                ct_bn = Node(
                    op_type="ConvTransposeBatchNorm",
                    inputs=ct.inputs + bn.inputs[1:],
                    outputs=bn.outputs,
                    attributes=ct.attributes,
                    name=f"FusedCTBN_{ct.name}",
                )
                idx = graph.nodes.index(ct)
                graph.nodes[idx] = ct_bn
                graph.nodes.remove(bn)
                return True
            i += 1
        return changed


def fuse_batchnorm_into_gemm(graph: Graph) -> bool:
    import numpy as np
    from onnx9000.core.ir import Constant

    changed = False
    for node in graph.nodes:
        if node.op_type == "BatchNormalization":
            bn_x = node.inputs[0]
            gemm_node = None
            for n in graph.nodes:
                if n.op_type == "Gemm" and bn_x in n.outputs:
                    gemm_node = n
                    break

            if not gemm_node:
                continue

            num_consumers = sum(1 for n in graph.nodes if bn_x in n.inputs)
            if num_consumers > 1 or bn_x in [getattr(o, "name", o) for o in graph.outputs]:
                continue

            scale_t = graph.tensors.get(node.inputs[1])
            b_t = graph.tensors.get(node.inputs[2])
            mean_t = graph.tensors.get(node.inputs[3])
            var_t = graph.tensors.get(node.inputs[4])

            if not all(
                t and getattr(t, "is_initializer", False) for t in [scale_t, b_t, mean_t, var_t]
            ):
                continue

            epsilon = node.attributes.get("epsilon").value if "epsilon" in node.attributes else 1e-5

            scale = scale_t.data
            b = b_t.data
            mean = mean_t.data
            var = var_t.data

            if not all(isinstance(x, np.ndarray) for x in [scale, b, mean, var]):
                continue

            gemm_w_name = gemm_node.inputs[1]
            gemm_w_t = graph.tensors.get(gemm_w_name)
            if (
                not gemm_w_t
                or not getattr(gemm_w_t, "is_initializer", False)
                or not isinstance(gemm_w_t.data, np.ndarray)
            ):
                continue

            transB = (
                gemm_node.attributes.get("transB").value if "transB" in gemm_node.attributes else 0
            )

            gemm_w = gemm_w_t.data

            if len(gemm_node.inputs) > 2 and gemm_node.inputs[2]:
                gemm_b_name = gemm_node.inputs[2]
                gemm_b_t = graph.tensors.get(gemm_b_name)
                if (
                    not gemm_b_t
                    or not getattr(gemm_b_t, "is_initializer", False)
                    or not isinstance(gemm_b_t.data, np.ndarray)
                ):
                    continue
                gemm_b = gemm_b_t.data
            else:
                gemm_b = np.zeros(
                    gemm_w.shape[0] if transB else gemm_w.shape[1], dtype=gemm_w.dtype
                )

            multiplier = scale / np.sqrt(var + epsilon)

            # Gemm formula: alpha * A * B + beta * C
            # If transB: B is (N, K) -> output features is N.
            # If not transB: B is (K, N) -> output features is N.
            if transB:
                multiplier_reshaped = multiplier.reshape(-1, 1)
                new_w = gemm_w * multiplier_reshaped
            else:
                multiplier_reshaped = multiplier.reshape(1, -1)
                new_w = gemm_w * multiplier_reshaped

            new_b = (gemm_b - mean) * multiplier + b

            new_w_name = f"{gemm_w_name}_fused_bn"
            new_b_name = f"{gemm_node.name}_fused_bn_b"

            new_w_t = Constant(
                new_w_name,
                values=new_w.astype(gemm_w.dtype),
                shape=new_w.shape,
                dtype=gemm_w_t.dtype,
            )
            new_b_t = Constant(
                new_b_name,
                values=new_b.astype(gemm_b.dtype if len(gemm_node.inputs) > 2 else gemm_w.dtype),
                shape=new_b.shape,
                dtype=b_t.dtype,
            )

            graph.add_tensor(new_w_t)
            graph.initializers.append(new_w_name)
            graph.add_tensor(new_b_t)
            graph.initializers.append(new_b_name)

            if len(gemm_node.inputs) > 2:
                gemm_node.inputs[1] = new_w_name
                gemm_node.inputs[2] = new_b_name
            else:
                gemm_node.inputs[1] = new_w_name
                gemm_node.inputs.append(new_b_name)

            for n in graph.nodes:
                for i, inp in enumerate(n.inputs):
                    if inp == node.outputs[0]:
                        n.inputs[i] = bn_x

            for i, out in enumerate(graph.outputs):
                if getattr(out, "name", out) == node.outputs[0]:
                    if isinstance(out, str):
                        graph.outputs[i] = bn_x
                    else:
                        out.name = bn_x

            node.outputs = []
            changed = True
            import logging

            logging.getLogger(__name__).info(
                f"Fused BatchNormalization {node.name} into Gemm {gemm_node.name}"
            )

    return changed


def fuse_batchnorm_into_conv(graph: Graph) -> bool:
    import numpy as np
    from onnx9000.core.ir import Constant

    changed = False
    for node in graph.nodes:
        if node.op_type == "BatchNormalization":
            # BN inputs: X, scale, B, mean, var
            bn_x = node.inputs[0]
            conv_node = None
            for n in graph.nodes:
                if n.op_type == "Conv" and bn_x in n.outputs:
                    conv_node = n
                    break

            if not conv_node:
                continue

            # Check if Conv outputs are only used by this BN
            num_consumers = sum(1 for n in graph.nodes if bn_x in n.inputs)
            if num_consumers > 1 or bn_x in [getattr(o, "name", o) for o in graph.outputs]:
                continue

            # Grab BN constants
            scale_t = graph.tensors.get(node.inputs[1])
            b_t = graph.tensors.get(node.inputs[2])
            mean_t = graph.tensors.get(node.inputs[3])
            var_t = graph.tensors.get(node.inputs[4])

            if not all(
                t and getattr(t, "is_initializer", False) for t in [scale_t, b_t, mean_t, var_t]
            ):
                continue

            epsilon = node.attributes.get("epsilon").value if "epsilon" in node.attributes else 1e-5

            scale = scale_t.data
            b = b_t.data
            mean = mean_t.data
            var = var_t.data

            if not all(isinstance(x, np.ndarray) for x in [scale, b, mean, var]):
                continue

            # Compute new weights and biases
            # W_new = W * (scale / sqrt(var + eps))
            # B_new = (B_conv - mean) * (scale / sqrt(var + eps)) + B_bn

            conv_w_name = conv_node.inputs[1]
            conv_w_t = graph.tensors.get(conv_w_name)
            if (
                not conv_w_t
                or not getattr(conv_w_t, "is_initializer", False)
                or not isinstance(conv_w_t.data, np.ndarray)
            ):
                continue

            conv_w = conv_w_t.data

            if len(conv_node.inputs) > 2 and conv_node.inputs[2]:
                conv_b_name = conv_node.inputs[2]
                conv_b_t = graph.tensors.get(conv_b_name)
                if (
                    not conv_b_t
                    or not getattr(conv_b_t, "is_initializer", False)
                    or not isinstance(conv_b_t.data, np.ndarray)
                ):
                    continue
                conv_b = conv_b_t.data
            else:
                conv_b = np.zeros(conv_w.shape[0], dtype=conv_w.dtype)

            # Compute multiplier
            multiplier = scale / np.sqrt(var + epsilon)

            # Conv weights: (M, C/group, kH, kW)
            # Multiplier is shape (M,)
            w_shape = conv_w.shape
            multiplier_reshaped = multiplier.reshape((w_shape[0],) + (1,) * (len(w_shape) - 1))

            new_w = conv_w * multiplier_reshaped
            new_b = (conv_b - mean) * multiplier + b

            new_w_name = f"{conv_w_name}_fused_bn"
            new_b_name = f"{conv_node.name}_fused_bn_b"

            new_w_t = Constant(
                new_w_name,
                values=new_w.astype(conv_w.dtype),
                shape=new_w.shape,
                dtype=conv_w_t.dtype,
            )
            new_b_t = Constant(
                new_b_name,
                values=new_b.astype(conv_b.dtype if len(conv_node.inputs) > 2 else conv_w.dtype),
                shape=new_b.shape,
                dtype=b_t.dtype,
            )
            graph.add_tensor(new_w_t)
            graph.initializers.append(new_w_name)
            graph.add_tensor(new_b_t)
            graph.initializers.append(new_b_name)

            # Update Conv inputs
            if len(conv_node.inputs) > 2:
                conv_node.inputs[1] = new_w_name
                conv_node.inputs[2] = new_b_name
            else:
                conv_node.inputs[1] = new_w_name
                conv_node.inputs.append(new_b_name)

            # Rewire BN outputs
            for n in graph.nodes:
                for i, inp in enumerate(n.inputs):
                    if inp == node.outputs[0]:
                        n.inputs[i] = bn_x

            for i, out in enumerate(graph.outputs):
                if getattr(out, "name", out) == node.outputs[0]:
                    if isinstance(out, str):
                        graph.outputs[i] = bn_x
                    else:
                        out.name = bn_x
            node.outputs = []  # Remove consumers to DCE it out
            changed = True
            import logging

            logging.getLogger(__name__).info(
                f"Fused BatchNormalization {node.name} into Conv {conv_node.name}"
            )

    return changed


def map_aten_arange_to_range(graph: Graph) -> bool:
    changed = False
    for node in graph.nodes:
        if node.op_type == "arange" and getattr(node, "domain", "") == "aten":
            import numpy as np
            from onnx9000.core.ir import Constant

            # aten::arange(end, dtype, layout, device, pin_memory) -> Range(start, limit, delta)
            # or aten::arange(start, end, step, dtype, layout, device, pin_memory)
            # Simplest way: map inputs correctly if they map directly to Range

            # It's safer to just remap the domain/op_type if it strictly has 3 inputs for start/limit/step
            # For simplicity, if it has 3 math inputs and the rest are ignored
            if len(node.inputs) >= 3:
                node.op_type = "Range"
                node.domain = ""
                # Strip out device/dtype parameters which aren't in ONNX Range
                node.inputs = node.inputs[:3]
                changed = True
                import logging

                logging.getLogger(__name__).info(
                    f"Mapped PyTorch aten::arange to ONNX Range at {node.name}"
                )

    return changed


def run_all_fusions(graph: Graph) -> None:
    """Implements the run_all_fusions method or operation."""
    PatternMatcherFusion().run(graph)
    while (
        fuse_batchnorm_into_conv(graph)
        or fuse_batchnorm_into_gemm(graph)
        or map_aten_arange_to_range(graph)
    ):
        pass


def fuse_linear_activation(graph: Graph) -> None:
    """Implements the fuse_linear_activation method or operation."""
    run_all_fusions(graph)


def fuse_consecutive_transpose(graph: Graph) -> None:
    """Implements the fuse_consecutive_transpose method or operation."""
    changed = True
    while changed:
        changed = False
        for _i, node1 in enumerate(graph.nodes):
            if node1.op_type == "Transpose":
                out = node1.outputs[0]
                node2 = next(
                    (n for n in graph.nodes if n.op_type == "Transpose" and n.inputs[0] == out),
                    None,
                )
                if node2:
                    perm1 = node1.attributes.get("perm", None)
                    perm2 = node2.attributes.get("perm", None)
                    is_identity = False
                    if perm1 is None and perm2 is None:
                        is_identity = True
                    elif perm1 is not None and perm2 is not None:
                        combined = [perm1[p] for p in perm2]
                        if combined == list(range(len(perm1))):
                            is_identity = True
                    if is_identity:
                        in_name = node1.inputs[0]
                        out_name = node2.outputs[0]
                        for n in graph.nodes:
                            for idx, inp in enumerate(n.inputs):
                                if inp == out_name:
                                    n.inputs[idx] = in_name
                        for idx, gout in enumerate(graph.outputs):
                            if gout == out_name:
                                graph.outputs[idx] = in_name
                        graph.nodes.remove(node1)
                        graph.nodes.remove(node2)
                        changed = True
                        break
    from onnx9000.optimizer.simplifier.passes.dce import dead_code_elimination

    dead_code_elimination(graph)


def fuse_matmul_add(graph: Graph) -> None:
    """Implements the fuse_matmul_add method or operation."""
    run_all_fusions(graph)
