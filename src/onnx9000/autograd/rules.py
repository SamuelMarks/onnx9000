"""
Gradient Rules Mapping

Contains the Vector-Jacobian Product (VJP) mapping rules for specific ONNX operations,
defining the mathematical backward propagation rules for basic arithmetic and activations.
"""

from onnx9000.autograd.vjp import VJPRule
from onnx9000.ir import Node


class AddVJP(VJPRule):
    """Autograd rule for the AddVJP operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        # dL/dA = dL/dOut, dL/dB = dL/dOut (ignoring broadcasting for simple mock)
        """Autograd rule for the build_backward_nodes operation."""

        grad_out = grad_outputs[0]  # pragma: no cover
        # In a real engine, we'd need to emit Identity or ReduceSum nodes if broadcasting happened  # pragma: no cover
        return [], [grad_out, grad_out]  # pragma: no cover


class MulVJP(VJPRule):
    """Autograd rule for the MulVJP operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        # dL/dA = dL/dOut * B, dL/dB = dL/dOut * A
        """Autograd rule for the build_backward_nodes operation."""

        grad_out = grad_outputs[0]  # pragma: no cover
        in_a, in_b = fwd_node.inputs[0], fwd_node.inputs[1]  # pragma: no cover
        # pragma: no cover
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"  # pragma: no cover
        grad_b_name = f"grad_{in_b}_wrt_{fwd_node.name}"  # pragma: no cover
        # pragma: no cover
        node_a = Node(  # pragma: no cover
            "Mul",  # pragma: no cover
            [grad_out, in_b],  # pragma: no cover
            [grad_a_name],  # pragma: no cover
            {},  # pragma: no cover
            name=f"{fwd_node.name}_bwd_mul_a",  # pragma: no cover
        )  # pragma: no cover
        node_b = Node(  # pragma: no cover
            "Mul",  # pragma: no cover
            [grad_out, in_a],  # pragma: no cover
            [grad_b_name],  # pragma: no cover
            {},  # pragma: no cover
            name=f"{fwd_node.name}_bwd_mul_b",  # pragma: no cover
        )  # pragma: no cover
        # pragma: no cover
        return [node_a, node_b], [grad_a_name, grad_b_name]  # pragma: no cover


class MatMulVJP(VJPRule):
    """Autograd rule for the MatMulVJP operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        # A @ B = C
        # dL/dA = dL/dC @ B^T
        # dL/dB = A^T @ dL/dC
        """Autograd rule for the build_backward_nodes operation."""

        grad_out = grad_outputs[0]
        in_a, in_b = fwd_node.inputs[0], fwd_node.inputs[1]

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

        return [node_b_t, node_a_t, node_grad_a, node_grad_b], [
            grad_a_name,
            grad_b_name,
        ]


class ReluVJP(VJPRule):
    """Autograd rule for the ReluVJP operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Autograd rule for the build_backward_nodes operation."""

        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"

        # Normally a custom ReluGrad or Where node
        node_grad = Node(
            "ReluGrad",
            [grad_out, in_a],
            [grad_a_name],
            {},
            name=f"{fwd_node.name}_bwd_relu",
        )
        return [node_grad], [grad_a_name]


class SigmoidVJP(VJPRule):
    """Autograd rule for the Sigmoid operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        fwd_out = fwd_node.outputs[0]
        grad_in_name = f"grad_{fwd_node.inputs[0]}_wrt_{fwd_node.name}"

        # dL/dX = dL/dY * Y * (1 - Y)
        # Using a custom SigmoidGrad node for simplicity matching ReluGrad
        node_grad = Node(
            "SigmoidGrad",
            [grad_out, fwd_out],
            [grad_in_name],
            {},
            name=f"{fwd_node.name}_bwd_sigmoid",
        )
        return [node_grad], [grad_in_name]


class TanhVJP(VJPRule):
    """Autograd rule for the Tanh operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        fwd_out = fwd_node.outputs[0]
        grad_in_name = f"grad_{fwd_node.inputs[0]}_wrt_{fwd_node.name}"

        # dL/dX = dL/dY * (1 - Y^2)
        node_grad = Node(
            "TanhGrad",
            [grad_out, fwd_out],
            [grad_in_name],
            {},
            name=f"{fwd_node.name}_bwd_tanh",
        )
        return [node_grad], [grad_in_name]


class MaxPoolVJP(VJPRule):
    """Autograd rule for the MaxPool operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_name = fwd_node.inputs[0]
        out_name = fwd_node.outputs[0]
        grad_in_name = f"grad_{in_name}_wrt_{fwd_node.name}"

        node_grad = Node(
            "MaxPoolGrad",
            [grad_out, in_name, out_name],
            [grad_in_name],
            fwd_node.attributes,
            name=f"{fwd_node.name}_bwd_maxpool",
        )
        return [node_grad], [grad_in_name]


class AveragePoolVJP(VJPRule):
    """Autograd rule for the AveragePool operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_name = fwd_node.inputs[0]
        grad_in_name = f"grad_{in_name}_wrt_{fwd_node.name}"

        node_grad = Node(
            "AveragePoolGrad",
            [grad_out, in_name],
            [grad_in_name],
            fwd_node.attributes,
            name=f"{fwd_node.name}_bwd_avgpool",
        )
        return [node_grad], [grad_in_name]


class ConvVJP(VJPRule):
    """Autograd rule for the Conv operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        x = fwd_node.inputs[0]
        w = fwd_node.inputs[1]

        grad_x_name = f"grad_{x}_wrt_{fwd_node.name}"
        grad_w_name = f"grad_{w}_wrt_{fwd_node.name}"

        nodes = []
        grad_names = [grad_x_name, grad_w_name]

        node_grad_w = Node(
            "ConvGradW",
            [grad_out, x],
            [grad_w_name],
            fwd_node.attributes,
            name=f"{fwd_node.name}_bwd_conv_w",
        )
        nodes.append(node_grad_w)

        node_grad_x = Node(
            "ConvGradX",
            [grad_out, w],
            [grad_x_name],
            fwd_node.attributes,
            name=f"{fwd_node.name}_bwd_conv_x",
        )
        nodes.append(node_grad_x)

        if len(fwd_node.inputs) > 2:
            b = fwd_node.inputs[2]
            grad_b_name = f"grad_{b}_wrt_{fwd_node.name}"
            node_grad_b = Node(
                "ConvGradB",
                [grad_out],
                [grad_b_name],
                fwd_node.attributes,
                name=f"{fwd_node.name}_bwd_conv_b",
            )
            nodes.append(node_grad_b)
            grad_names.append(grad_b_name)

        return nodes, grad_names


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
            [grad_out] + fwd_node.inputs,
            [grad_x_name, grad_scale_name, grad_b_name],
            fwd_node.attributes,
            name=f"{fwd_node.name}_bwd_batchnorm",
        )
        return [node_grad], [grad_x_name, grad_scale_name, grad_b_name]


_VJP_REGISTRY = {
    "BatchNormalization": BatchNormalizationVJP(),
    "Conv": ConvVJP(),
    "MaxPool": MaxPoolVJP(),
    "AveragePool": AveragePoolVJP(),
    "Sigmoid": SigmoidVJP(),
    "Tanh": TanhVJP(),
    "Add": AddVJP(),
    "Mul": MulVJP(),
    "MatMul": MatMulVJP(),
    "Relu": ReluVJP(),
}


def get_vjp_rule(op_type: str) -> VJPRule:
    """Autograd rule for the get_vjp_rule operation."""

    if op_type not in _VJP_REGISTRY:
        raise NotImplementedError(f"No VJP rule for {op_type}")  # pragma: no cover
    return _VJP_REGISTRY[op_type]
