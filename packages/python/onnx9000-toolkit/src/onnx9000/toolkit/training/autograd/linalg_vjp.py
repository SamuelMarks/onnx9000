"""Linear algebra and reduction VJP rules."""

from __future__ import annotations

from onnx9000.core.ir import Node
from onnx9000.toolkit.training.autograd.vjp import VJPRule


class MatMulVJP(VJPRule):
    """Autograd rule for the MatMulVJP operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Autograd rule for the build_backward_nodes operation."""
        grad_out = grad_outputs[0]
        (in_a, in_b) = (fwd_node.inputs[0], fwd_node.inputs[1])
        b_t = f"{in_b}_T"
        a_t = f"{in_a}_T"
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        grad_b_name = f"grad_{in_b}_wrt_{fwd_node.name}"
        node_b_t = Node(
            "Transpose",
            [in_b],
            [b_t],
            {"perm": [1, 0]},
            name=f"{fwd_node.name}_bwd_trans_b",
        )
        node_a_t = Node(
            "Transpose",
            [in_a],
            [a_t],
            {"perm": [1, 0]},
            name=f"{fwd_node.name}_bwd_trans_a",
        )
        node_grad_a = Node(
            "MatMul",
            [grad_out, b_t],
            [grad_a_name],
            {},
            name=f"{fwd_node.name}_bwd_matmul_a",
        )
        node_grad_b = Node(
            "MatMul",
            [a_t, grad_out],
            [grad_b_name],
            {},
            name=f"{fwd_node.name}_bwd_matmul_b",
        )
        return (
            [node_b_t, node_a_t, node_grad_a, node_grad_b],
            [grad_a_name, grad_b_name],
        )


class ReduceSumVJP(VJPRule):
    """Autograd rule for the ReduceSum operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        node_a = Node(
            "Expand",
            [grad_out, f"shape_{in_a}"],
            [grad_a_name],
            {},
            name=f"{fwd_node.name}_bwd_reducesum",
        )
        return ([node_a], [grad_a_name])


class ReduceMeanVJP(VJPRule):
    """Autograd rule for the ReduceMean operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        node_expand = Node(
            "Expand",
            [grad_out, f"shape_{in_a}"],
            [f"{grad_out}_expanded"],
            {},
            name=f"{fwd_node.name}_bwd_reducemean_expand",
        )
        node_a = Node(
            "Div",
            [f"{grad_out}_expanded", f"size_{in_a}"],
            [grad_a_name],
            {},
            name=f"{fwd_node.name}_bwd_reducemean_div",
        )
        return ([node_expand, node_a], [grad_a_name])


class ReduceMaxVJP(VJPRule):
    """Autograd rule for the ReduceMax operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        out_name = fwd_node.outputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        node_a = Node(
            "ReduceMaxGrad",
            [grad_out, in_a, out_name],
            [grad_a_name],
            fwd_node.attributes,
            name=f"{fwd_node.name}_bwd_reducemax",
        )
        return ([node_a], [grad_a_name])


class ReduceMinVJP(VJPRule):
    """Autograd rule for the ReduceMin operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        out_name = fwd_node.outputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        node_a = Node(
            "ReduceMinGrad",
            [grad_out, in_a, out_name],
            [grad_a_name],
            fwd_node.attributes,
            name=f"{fwd_node.name}_bwd_reducemin",
        )
        return ([node_a], [grad_a_name])


class ReduceProdVJP(VJPRule):
    """Autograd rule for the ReduceProd operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        out_name = fwd_node.outputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        node_a = Node(
            "ReduceProdGrad",
            [grad_out, in_a, out_name],
            [grad_a_name],
            fwd_node.attributes,
            name=f"{fwd_node.name}_bwd_reduceprod",
        )
        return ([node_a], [grad_a_name])


class ReduceL1VJP(VJPRule):
    """Autograd rule for the ReduceL1 operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        node_a = Node(
            "ReduceL1Grad",
            [grad_out, in_a],
            [grad_a_name],
            fwd_node.attributes,
            name=f"{fwd_node.name}_bwd_reducel1",
        )
        return ([node_a], [grad_a_name])


class ReduceL2VJP(VJPRule):
    """Autograd rule for the ReduceL2 operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        out_name = fwd_node.outputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        node_a = Node(
            "ReduceL2Grad",
            [grad_out, in_a, out_name],
            [grad_a_name],
            fwd_node.attributes,
            name=f"{fwd_node.name}_bwd_reducel2",
        )
        return ([node_a], [grad_a_name])


class ReduceLogSumVJP(VJPRule):
    """Autograd rule for the ReduceLogSum operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        out_name = fwd_node.outputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        node_a = Node(
            "ReduceLogSumGrad",
            [grad_out, in_a, out_name],
            [grad_a_name],
            fwd_node.attributes,
            name=f"{fwd_node.name}_bwd_reducelogsum",
        )
        return ([node_a], [grad_a_name])


class ReduceLogSumExpVJP(VJPRule):
    """Autograd rule for the ReduceLogSumExp operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        out_name = fwd_node.outputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        node_a = Node(
            "ReduceLogSumExpGrad",
            [grad_out, in_a, out_name],
            [grad_a_name],
            fwd_node.attributes,
            name=f"{fwd_node.name}_bwd_reducelogsumexp",
        )
        return ([node_a], [grad_a_name])


class ReduceSumSquareVJP(VJPRule):
    """Autograd rule for the ReduceSumSquare operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        node_a = Node(
            "ReduceSumSquareGrad",
            [grad_out, in_a],
            [grad_a_name],
            fwd_node.attributes,
            name=f"{fwd_node.name}_bwd_reducesumsquare",
        )
        return ([node_a], [grad_a_name])


class GemmVJP(VJPRule):
    """Autograd rule for the Gemm operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        in_b = fwd_node.inputs[1]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        grad_b_name = f"grad_{in_b}_wrt_{fwd_node.name}"
        node_a = Node(
            "GemmGradA",
            [grad_out, in_a, in_b],
            [grad_a_name],
            fwd_node.attributes,
            name=f"{fwd_node.name}_bwd_gemm_a",
        )
        node_b = Node(
            "GemmGradB",
            [grad_out, in_a, in_b],
            [grad_b_name],
            fwd_node.attributes,
            name=f"{fwd_node.name}_bwd_gemm_b",
        )
        nodes = [node_a, node_b]
        names = [grad_a_name, grad_b_name]
        if len(fwd_node.inputs) > 2:
            in_c = fwd_node.inputs[2]
            grad_c_name = f"grad_{in_c}_wrt_{fwd_node.name}"
            node_c = Node(
                "GemmGradC",
                [grad_out],
                [grad_c_name],
                fwd_node.attributes,
                name=f"{fwd_node.name}_bwd_gemm_c",
            )
            nodes.append(node_c)
            names.append(grad_c_name)
        return (nodes, names)


class TopKVJP(VJPRule):
    """Autograd rule for the TopK operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        indices = fwd_node.outputs[1]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        node_a = Node(
            "TopKGrad",
            [grad_out, indices, in_a],
            [grad_a_name],
            fwd_node.attributes.copy(),
            name=f"{fwd_node.name}_bwd_topk_a",
        )
        return ([node_a], [grad_a_name])


class CumSumVJP(VJPRule):
    """Autograd rule for the CumSum operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        axis = fwd_node.inputs[1]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        attrs = fwd_node.attributes.copy()
        attrs["reverse"] = 1 if not attrs.get("reverse", 0) else 0
        node_a = Node(
            "CumSum",
            [grad_out, axis],
            [grad_a_name],
            attrs,
            name=f"{fwd_node.name}_bwd_cumsum",
        )
        return ([node_a], [grad_a_name, f"grad_{axis}_wrt_{fwd_node.name}"])


class TriluVJP(VJPRule):
    """Autograd rule for the Trilu operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        k = fwd_node.inputs[1] if len(fwd_node.inputs) > 1 else None
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        inputs = [grad_out]
        if k is not None:
            inputs.append(k)
        node_a = Node(
            "Trilu",
            inputs,
            [grad_a_name],
            fwd_node.attributes.copy(),
            name=f"{fwd_node.name}_bwd_trilu",
        )
        grads = [grad_a_name]
        if k is not None:
            grads.append(f"grad_{k}_wrt_{fwd_node.name}")
        return ([node_a], grads)


class EinsumVJP(VJPRule):
    """Autograd rule for the Einsum operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        nodes = []
        grads = []
        for i, in_x in enumerate(fwd_node.inputs):
            g_name = f"grad_{in_x}_wrt_{fwd_node.name}"
            n = Node(
                "EinsumGrad",
                [grad_out, *list(fwd_node.inputs)],
                [g_name],
                fwd_node.attributes.copy(),
                name=f"{fwd_node.name}_bwd_einsum_{i}",
            )
            nodes.append(n)
            grads.append(g_name)
        return (nodes, grads)
