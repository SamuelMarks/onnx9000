"""Provides fusion.py module functionality."""

import logging
from typing import List, Optional

import numpy as np

from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.optimize.simplifier.passes.base import GraphPass

logger = logging.getLogger(__name__)


class FusionPass(GraphPass):
    """
    Operator Fusion Passes.
    Combines adjacent operations into fused kernels for improved memory
    locality and performance.
    """

    def run(self, graph: Graph) -> bool:
        """Provides run functionality and verification."""
        changed = False
        while True:
            local_changed = self._run_once(graph)
            if not local_changed:
                break
            changed = True
        return changed

    def _run_once(self, graph: Graph) -> bool:
        """Provides  run once functionality and verification."""
        return False


class PatternMatcherFusion(FusionPass):
    """
    A Pattern Matching DSL for Operator Fusion.
    """

    def _run_once(self, graph: Graph) -> bool:
        """Provides  run once functionality and verification."""
        changed = False

        usages = {}
        for node in graph.nodes:
            for inp in node.inputs:
                usages[inp] = usages.get(inp, 0) + 1
        for out in graph.outputs:
            usages[out] = usages.get(out, 0) + 1

        def match_chain(start_node: Node, ops: List[str]) -> Optional[List[Node]]:
            """Provides match chain functionality and verification."""
            chain = [start_node]
            curr = start_node
            if start_node.op_type != ops[0]:
                return None

            for expected_op in ops[1:]:
                # find the next node
                consumers = [n for n in graph.nodes if curr.outputs[0] in n.inputs]
                if not consumers:
                    return None

                # For linear chains, we expect exactly 1 consumer that matches
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

            # 066: MatMul + Add -> Gemm
            chain = match_chain(node, ["MatMul", "Add"])
            if chain:
                mm, add = chain
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

            # 076: Swish / SiLU fusion (Sigmoid -> Mul) -> Swish
            chain = match_chain(node, ["Sigmoid", "Mul"])
            if chain:
                sig, mul = chain
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

            # 075: Softmax fusion (Exp -> ReduceSum -> Div)
            chain = match_chain(node, ["Exp", "ReduceSum", "Div"])
            if chain:
                exp, rsum, div = chain
                if (
                    exp.outputs[0] in rsum.inputs
                    and exp.outputs[0] in div.inputs
                    and rsum.outputs[0] in div.inputs
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

            # 072: LayerNorm fusion (Sub -> Pow -> ReduceMean -> Add -> Div -> Mul -> Add)
            chain = match_chain(
                node, ["Sub", "Pow", "ReduceMean", "Add", "Div", "Mul", "Add"]
            )
            if chain:
                sub, pw, rm, add1, div, mul, add2 = chain
                # Simplified check for LayerNorm
                ln_node = Node(
                    op_type="LayerNormalization",
                    inputs=[
                        sub.inputs[0],
                        mul.inputs[1]
                        if mul.inputs[0] == div.outputs[0]
                        else mul.inputs[0],
                        add2.inputs[1]
                        if add2.inputs[0] == mul.outputs[0]
                        else add2.inputs[0],
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

            # 073: Gelu fusion (Div -> Erf -> Add -> Mul -> Mul)
            chain = match_chain(node, ["Div", "Erf", "Add", "Mul", "Mul"])
            if chain:
                div, erf, add, mul1, mul2 = chain
                gelu_node = Node(
                    op_type="Gelu",
                    inputs=[div.inputs[0]],  # x
                    outputs=mul2.outputs,
                    attributes={},
                    name=f"FusedGelu_{div.name}",
                )
                idx = graph.nodes.index(div)
                graph.nodes[idx] = gelu_node
                for n in chain[1:]:
                    graph.nodes.remove(n)
                return True

            # 064: Conv + BatchNorm (represented as ConvBatchNorm or folded if possible)
            # Actually we can check if it's Conv -> BatchNormalization
            chain = match_chain(node, ["Conv", "BatchNormalization"])
            if chain:
                conv, bn = chain
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

            # 070: BatchNorm + Relu
            chain = match_chain(node, ["BatchNormalization", "Relu"])
            if chain:
                bn, relu = chain
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

            # 071: MatMul + Relu
            chain = match_chain(node, ["MatMul", "Relu"])
            if chain:
                mm, relu = chain
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

            # 067: Gemm + BatchNorm
            chain = match_chain(node, ["Gemm", "BatchNormalization"])
            if chain:
                gm, bn = chain
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

            # 068: Conv + Add
            chain = match_chain(node, ["Conv", "Add"])
            if chain:
                conv, add = chain
                if len(conv.inputs) == 2:
                    b_idx = 1 if add.inputs[0] == conv.outputs[0] else 0
                    conv.inputs.append(add.inputs[b_idx])
                    conv.outputs[0] = add.outputs[0]
                    graph.nodes.remove(add)
                    return True

            # 069: Conv + Mul
            chain = match_chain(node, ["Conv", "Mul"])
            if chain:
                conv, mul = chain
                cm = Node(
                    op_type="ConvMul",
                    inputs=conv.inputs
                    + [
                        mul.inputs[1]
                        if mul.inputs[0] == conv.outputs[0]
                        else mul.inputs[0]
                    ],
                    outputs=mul.outputs,
                    attributes=conv.attributes,
                    name=f"FusedConvMul_{conv.name}",
                )
                idx = graph.nodes.index(conv)
                graph.nodes[idx] = cm
                graph.nodes.remove(mul)
                return True

            # 065: ConvTranspose + BatchNorm
            chain = match_chain(node, ["ConvTranspose", "BatchNormalization"])
            if chain:
                ct, bn = chain
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


def run_all_fusions(graph: Graph) -> None:
    """Provides run all fusions functionality and verification."""
    PatternMatcherFusion().run(graph)


def fuse_linear_activation(graph: Graph) -> None:
    """Provides fuse linear activation functionality and verification."""
    run_all_fusions(graph)


def fuse_consecutive_transpose(graph: Graph) -> None:
    """Provides fuse consecutive transpose functionality and verification."""
    changed = True
    while changed:
        changed = False
        for i, node1 in enumerate(graph.nodes):
            if node1.op_type == "Transpose":
                # find node2 that uses node1's output
                out = node1.outputs[0]
                node2 = next(
                    (
                        n
                        for n in graph.nodes
                        if n.op_type == "Transpose" and n.inputs[0] == out
                    ),
                    None,
                )
                if node2:
                    # check if perm attributes are inverse or if applying both is identity
                    perm1 = node1.attributes.get("perm", None)
                    perm2 = node2.attributes.get("perm", None)

                    # If perm is missing, it's a reverse by default in ONNX
                    # Two consecutive reverses cancel out.
                    is_identity = False
                    if perm1 is None and perm2 is None:
                        is_identity = True
                    elif perm1 is not None and perm2 is not None:
                        # apply perm2 on perm1
                        combined = [perm1[p] for p in perm2]
                        if combined == list(range(len(perm1))):
                            is_identity = True

                    if is_identity:
                        # Reroute node2 outputs to node1 inputs
                        in_name = node1.inputs[0]
                        out_name = node2.outputs[0]

                        # Replace all uses of out_name with in_name
                        for n in graph.nodes:
                            for idx, inp in enumerate(n.inputs):
                                if inp == out_name:
                                    n.inputs[idx] = in_name

                        # If out_name is a graph output, we might need an Identity node or just rename the graph output
                        for idx, gout in enumerate(graph.outputs):
                            if gout == out_name:
                                graph.outputs[idx] = in_name

                        graph.nodes.remove(node1)
                        graph.nodes.remove(node2)
                        changed = True
                        break
    from onnx9000.optimize.simplifier.passes.dce import dead_code_elimination

    dead_code_elimination(graph)


def fuse_matmul_add(graph: Graph) -> None:
    """Provides fuse matmul add functionality and verification."""
    run_all_fusions(graph)
