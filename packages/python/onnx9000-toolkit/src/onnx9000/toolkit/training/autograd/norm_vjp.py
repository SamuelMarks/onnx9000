"""Normalization, regularization, and loss VJP rules."""

from __future__ import annotations

from onnx9000.core.ir import Node
from onnx9000.toolkit.training.autograd.vjp import VJPRule


class SoftmaxVJP(VJPRule):
    """Autograd rule for the Softmax operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        out_name = fwd_node.outputs[0]
        grad_a_name = f"grad_{fwd_node.inputs[0]}_wrt_{fwd_node.name}"
        node_a = Node(
            "SoftmaxGrad",
            [grad_out, out_name],
            [grad_a_name],
            fwd_node.attributes,
            name=f"{fwd_node.name}_bwd_softmax_a",
        )
        return ([node_a], [grad_a_name])


class LogSoftmaxVJP(VJPRule):
    """Autograd rule for the LogSoftmax operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        out_name = fwd_node.outputs[0]
        grad_a_name = f"grad_{fwd_node.inputs[0]}_wrt_{fwd_node.name}"
        node_a = Node(
            "LogSoftmaxGrad",
            [grad_out, out_name],
            [grad_a_name],
            fwd_node.attributes,
            name=f"{fwd_node.name}_bwd_logsoftmax_a",
        )
        return ([node_a], [grad_a_name])


class LayerNormalizationVJP(VJPRule):
    """Autograd rule for the LayerNormalization operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        x = fwd_node.inputs[0]
        scale = fwd_node.inputs[1]
        grad_x_name = f"grad_{x}_wrt_{fwd_node.name}"
        grad_scale_name = f"grad_{scale}_wrt_{fwd_node.name}"
        nodes = []
        names = [grad_x_name, grad_scale_name]
        if len(fwd_node.inputs) > 2:
            b = fwd_node.inputs[2]
            grad_b_name = f"grad_{b}_wrt_{fwd_node.name}"
            node_grad = Node(
                "LayerNormalizationGrad",
                [grad_out, *fwd_node.inputs],
                [grad_x_name, grad_scale_name, grad_b_name],
                fwd_node.attributes,
                name=f"{fwd_node.name}_bwd_layernorm",
            )
            nodes.append(node_grad)
            names.append(grad_b_name)
        else:
            node_grad = Node(
                "LayerNormalizationGrad",
                [grad_out, *fwd_node.inputs],
                [grad_x_name, grad_scale_name],
                fwd_node.attributes,
                name=f"{fwd_node.name}_bwd_layernorm",
            )
            nodes.append(node_grad)
        return (nodes, names)


class InstanceNormalizationVJP(VJPRule):
    """Autograd rule for the InstanceNormalization operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        x = fwd_node.inputs[0]
        scale = fwd_node.inputs[1]
        b = fwd_node.inputs[2]
        grad_x_name = f"grad_{x}_wrt_{fwd_node.name}"
        grad_scale_name = f"grad_{scale}_wrt_{fwd_node.name}"
        grad_b_name = f"grad_{b}_wrt_{fwd_node.name}"
        node_grad = Node(
            "InstanceNormalizationGrad",
            [grad_out, *fwd_node.inputs],
            [grad_x_name, grad_scale_name, grad_b_name],
            fwd_node.attributes,
            name=f"{fwd_node.name}_bwd_instancenorm",
        )
        return ([node_grad], [grad_x_name, grad_scale_name, grad_b_name])


class DropoutVJP(VJPRule):
    """Autograd rule for the Dropout operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        mask_name = fwd_node.outputs[1] if len(fwd_node.outputs) > 1 else f"mask_{fwd_node.name}"
        node_a = Node(
            "DropoutGrad",
            [grad_out, mask_name],
            [grad_a_name],
            fwd_node.attributes,
            name=f"{fwd_node.name}_bwd_dropout",
        )
        return ([node_a], [grad_a_name])


class BatchNormalizationVJP(VJPRule):
    """Autograd rule for the BatchNormalization operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        x = fwd_node.inputs[0]
        scale = fwd_node.inputs[1]
        grad_x_name = f"grad_{x}_wrt_{fwd_node.name}"
        grad_scale_name = f"grad_{scale}_wrt_{fwd_node.name}"
        grad_b_name = f"grad_{fwd_node.inputs[2]}_wrt_{fwd_node.name}"
        node_grad = Node(
            "BatchNormalizationGrad",
            [grad_out, *fwd_node.inputs],
            [grad_x_name, grad_scale_name, grad_b_name],
            fwd_node.attributes.copy(),
            name=f"{fwd_node.name}_bwd_batchnorm",
        )
        # Pad with None or constant 0s for running mean/var since they don't get gradients
        grads = [grad_x_name, grad_scale_name, grad_b_name]
        for inp in fwd_node.inputs[3:]:
            g_name = f"grad_{inp}_wrt_{fwd_node.name}"
            grads.append(g_name)

        # We need to output constant 0s for mean/var
        nodes = [node_grad]
        for inp in fwd_node.inputs[3:]:
            g_name = f"grad_{inp}_wrt_{fwd_node.name}"
            nodes.append(
                Node(
                    "ConstantOfShape",
                    [fwd_node.inputs[1]],
                    [g_name],
                    {"value": 0.0},
                    name=f"{fwd_node.name}_bwd_bn_zero_{inp}",
                )
            )

        return (nodes, grads)


class LpNormalizationVJP(VJPRule):
    """Autograd rule for the LpNormalization operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        node_a = Node(
            "LpNormalizationGrad",
            [grad_out, in_a],
            [grad_a_name],
            fwd_node.attributes.copy(),
            name=f"{fwd_node.name}_bwd_lpnormalization",
        )
        return ([node_a], [grad_a_name])


class BCEWithLogitsLossVJP(VJPRule):
    """Autograd rule for BCEWithLogitsLoss explicitly computing gradient as Sigmoid(X) - Target."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Execute the build backward nodes operation."""
        grad_out = grad_outputs[0]
        logits = fwd_node.inputs[0]
        target = fwd_node.inputs[1]
        g_name = f"grad_{logits}_wrt_{fwd_node.name}"

        sig_out = f"{fwd_node.name}_sig"
        diff_out = f"{fwd_node.name}_diff"

        nodes = [
            Node("Sigmoid", [logits], [sig_out], {}, name=f"{fwd_node.name}_sig_node"),
            Node("Sub", [sig_out, target], [diff_out], {}, name=f"{fwd_node.name}_sub"),
            Node(
                "Mul",
                [diff_out, grad_out],
                [g_name],
                {},
                name=f"{fwd_node.name}_mul_grad",
            ),
        ]

        t_g_name = f"grad_{target}_wrt_{fwd_node.name}"
        nodes.append(
            Node(
                "ConstantOfShape",
                [target],
                [t_g_name],
                {"value": 0},
                name=f"{fwd_node.name}_t_zero",
            )
        )

        return (nodes, [g_name, t_g_name])


class BinaryCrossEntropyLossVJP(VJPRule):
    """Autograd rule for BinaryCrossEntropyLoss."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Execute the build backward nodes operation."""
        grad_out = grad_outputs[0]
        pred = fwd_node.inputs[0]
        target = fwd_node.inputs[1]
        g_name = f"grad_{pred}_wrt_{fwd_node.name}"

        # dL/dPred = (Pred - Target) / (Pred * (1 - Pred))
        one_minus_pred = f"{fwd_node.name}_1_minus_p"
        pred_diff = f"{fwd_node.name}_p_diff"
        denom = f"{fwd_node.name}_denom"
        grad_unscaled = f"{fwd_node.name}_grad_u"

        nodes = [
            Node("Sub", [pred, target], [pred_diff], {}, name=f"{fwd_node.name}_sub"),
            Node(
                "ConstantOfShape",
                [pred],
                ["_const_1"],
                {"value": 1.0},
                name=f"{fwd_node.name}_c1",
            ),
            Node(
                "Sub",
                ["_const_1", pred],
                [one_minus_pred],
                {},
                name=f"{fwd_node.name}_sub1",
            ),
            Node(
                "Mul",
                [pred, one_minus_pred],
                [denom],
                {},
                name=f"{fwd_node.name}_mul_den",
            ),
            Node(
                "Div",
                [pred_diff, denom],
                [grad_unscaled],
                {},
                name=f"{fwd_node.name}_div",
            ),
            Node(
                "Mul",
                [grad_unscaled, grad_out],
                [g_name],
                {},
                name=f"{fwd_node.name}_mul_grad",
            ),
        ]

        # Target gradient is 0
        t_g_name = f"grad_{target}_wrt_{fwd_node.name}"
        nodes.append(
            Node(
                "ConstantOfShape",
                [target],
                [t_g_name],
                {"value": 0},
                name=f"{fwd_node.name}_t_zero",
            )
        )

        return (nodes, [g_name, t_g_name])


class SoftmaxCrossEntropyLossVJP(VJPRule):
    """Autograd rule for SoftmaxCrossEntropyLoss."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Execute the build backward nodes operation."""
        grad_out = grad_outputs[0]
        logits = fwd_node.inputs[0]
        target = fwd_node.inputs[1]
        g_name = f"grad_{logits}_wrt_{fwd_node.name}"

        # dL/dLogits = Softmax(logits) - target
        softmax_name = f"{fwd_node.name}_softmax"
        diff_name = f"{fwd_node.name}_diff"

        nodes = [
            Node(
                "Softmax",
                [logits],
                [softmax_name],
                {"axis": -1},
                name=f"{fwd_node.name}_sm",
            ),
            Node(
                "Sub",
                [softmax_name, target],
                [diff_name],
                {},
                name=f"{fwd_node.name}_sub",
            ),
            Node(
                "Mul",
                [diff_name, grad_out],
                [g_name],
                {},
                name=f"{fwd_node.name}_mul_grad",
            ),
        ]

        # Target gradient is 0 (it's non-differentiable)
        t_g_name = f"grad_{target}_wrt_{fwd_node.name}"
        nodes.append(
            Node(
                "ConstantOfShape",
                [target],
                [t_g_name],
                {"value": 0},
                name=f"{fwd_node.name}_t_zero",
            )
        )

        return (nodes, [g_name, t_g_name])
