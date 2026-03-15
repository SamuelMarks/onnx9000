"""
Gradient Rules Mapping

Contains the Vector-Jacobian Product (VJP) mapping rules for specific ONNX operations,
defining the mathematical backward propagation rules for basic arithmetic and activations.
"""

from onnx9000.training.autograd.vjp import VJPRule
from onnx9000.core.ir import Node


class AddVJP(VJPRule):
    """Autograd rule for the AddVJP operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        # dL/dA = dL/dOut, dL/dB = dL/dOut (ignoring broadcasting for simple mock)
        """Autograd rule for the build_backward_nodes operation."""

        grad_out = grad_outputs[0]
        # In a real engine, we'd need to emit Identity or ReduceSum nodes if broadcasting happened
        return [], [grad_out, grad_out]


class MulVJP(VJPRule):
    """Autograd rule for the MulVJP operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        # dL/dA = dL/dOut * B, dL/dB = dL/dOut * A
        """Autograd rule for the build_backward_nodes operation."""

        grad_out = grad_outputs[0]
        in_a, in_b = fwd_node.inputs[0], fwd_node.inputs[1]

        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        grad_b_name = f"grad_{in_b}_wrt_{fwd_node.name}"

        node_a = Node(
            "Mul",
            [grad_out, in_b],
            [grad_a_name],
            {},
            name=f"{fwd_node.name}_bwd_mul_a",
        )
        node_b = Node(
            "Mul",
            [grad_out, in_a],
            [grad_b_name],
            {},
            name=f"{fwd_node.name}_bwd_mul_b",
        )

        return [node_a, node_b], [grad_a_name, grad_b_name]


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


class SubVJP(VJPRule):
    """Autograd rule for the Sub operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a, in_b = fwd_node.inputs[0], fwd_node.inputs[1]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        grad_b_name = f"grad_{in_b}_wrt_{fwd_node.name}"

        node_a = Node(
            "Identity", [grad_out], [grad_a_name], {}, name=f"{fwd_node.name}_bwd_sub_a"
        )
        node_b = Node(
            "Neg", [grad_out], [grad_b_name], {}, name=f"{fwd_node.name}_bwd_sub_b"
        )
        return [node_a, node_b], [grad_a_name, grad_b_name]


class DivVJP(VJPRule):
    """Autograd rule for the Div operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a, in_b = fwd_node.inputs[0], fwd_node.inputs[1]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        grad_b_name = f"grad_{in_b}_wrt_{fwd_node.name}"

        # dL/dA = dL/dOut / B
        node_a = Node(
            "Div",
            [grad_out, in_b],
            [grad_a_name],
            {},
            name=f"{fwd_node.name}_bwd_div_a",
        )

        # dL/dB = - dL/dOut * A / (B^2)
        b_sq_name = f"{in_b}_sq_wrt_{fwd_node.name}"
        node_b_sq = Node(
            "Mul", [in_b, in_b], [b_sq_name], {}, name=f"{fwd_node.name}_bwd_div_b_sq"
        )

        a_div_b_sq_name = f"{in_a}_div_b_sq_wrt_{fwd_node.name}"
        node_a_div_b_sq = Node(
            "Div",
            [in_a, b_sq_name],
            [a_div_b_sq_name],
            {},
            name=f"{fwd_node.name}_bwd_div_a_div_b_sq",
        )

        neg_a_div_b_sq_name = f"neg_{a_div_b_sq_name}"
        node_neg_a_div_b_sq = Node(
            "Neg",
            [a_div_b_sq_name],
            [neg_a_div_b_sq_name],
            {},
            name=f"{fwd_node.name}_bwd_div_neg_a_div_b_sq",
        )

        node_b = Node(
            "Mul",
            [grad_out, neg_a_div_b_sq_name],
            [grad_b_name],
            {},
            name=f"{fwd_node.name}_bwd_div_b",
        )

        return [node_a, node_b_sq, node_a_div_b_sq, node_neg_a_div_b_sq, node_b], [
            grad_a_name,
            grad_b_name,
        ]


class PowVJP(VJPRule):
    """Autograd rule for the Pow operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a, in_b = fwd_node.inputs[0], fwd_node.inputs[1]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        grad_b_name = f"grad_{in_b}_wrt_{fwd_node.name}"

        # Note: Pow requires constants or other operators to be implemented exactly.
        # This is a simplified mathematical representation.
        node_a = Node(
            "PowGradA",
            [grad_out, in_a, in_b],
            [grad_a_name],
            {},
            name=f"{fwd_node.name}_bwd_pow_a",
        )
        node_b = Node(
            "PowGradB",
            [grad_out, in_a, in_b],
            [grad_b_name],
            {},
            name=f"{fwd_node.name}_bwd_pow_b",
        )
        return [node_a, node_b], [grad_a_name, grad_b_name]


class ModVJP(VJPRule):
    """Autograd rule for the Mod operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a, in_b = fwd_node.inputs[0], fwd_node.inputs[1]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        grad_b_name = f"grad_{in_b}_wrt_{fwd_node.name}"

        node_a = Node(
            "ModGradA",
            [grad_out, in_a, in_b],
            [grad_a_name],
            {},
            name=f"{fwd_node.name}_bwd_mod_a",
        )
        node_b = Node(
            "ModGradB",
            [grad_out, in_a, in_b],
            [grad_b_name],
            {},
            name=f"{fwd_node.name}_bwd_mod_b",
        )
        return [node_a, node_b], [grad_a_name, grad_b_name]


class AbsVJP(VJPRule):
    """Autograd rule for the Abs operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"

        sign_name = f"sign_{in_a}_wrt_{fwd_node.name}"
        node_sign = Node(
            "Sign", [in_a], [sign_name], {}, name=f"{fwd_node.name}_bwd_abs_sign"
        )
        node_a = Node(
            "Mul",
            [grad_out, sign_name],
            [grad_a_name],
            {},
            name=f"{fwd_node.name}_bwd_abs_a",
        )
        return [node_sign, node_a], [grad_a_name]


class NegVJP(VJPRule):
    """Autograd rule for the Neg operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"

        node_a = Node(
            "Neg", [grad_out], [grad_a_name], {}, name=f"{fwd_node.name}_bwd_neg_a"
        )
        return [node_a], [grad_a_name]


class SignVJP(VJPRule):
    """Autograd rule for the Sign operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        in_a = fwd_node.inputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"

        node_a = Node(
            "ConstantOfShape",
            [in_a],
            [grad_a_name],
            {"value": 0.0},
            name=f"{fwd_node.name}_bwd_sign_a",
        )
        return [node_a], [grad_a_name]


class ExpVJP(VJPRule):
    """Autograd rule for the Exp operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        out_name = fwd_node.outputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"

        node_a = Node(
            "Mul",
            [grad_out, out_name],
            [grad_a_name],
            {},
            name=f"{fwd_node.name}_bwd_exp_a",
        )
        return [node_a], [grad_a_name]


class LogVJP(VJPRule):
    """Autograd rule for the Log operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"

        node_a = Node(
            "Div",
            [grad_out, in_a],
            [grad_a_name],
            {},
            name=f"{fwd_node.name}_bwd_log_a",
        )
        return [node_a], [grad_a_name]


class SqrtVJP(VJPRule):
    """Autograd rule for the Sqrt operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        out_name = fwd_node.outputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"

        node_a = Node(
            "SqrtGrad",
            [grad_out, out_name],
            [grad_a_name],
            {},
            name=f"{fwd_node.name}_bwd_sqrt_a",
        )
        return [node_a], [grad_a_name]


class SinVJP(VJPRule):
    """Autograd rule for the Sin operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"

        cos_name = f"cos_{in_a}_wrt_{fwd_node.name}"
        node_cos = Node(
            "Cos", [in_a], [cos_name], {}, name=f"{fwd_node.name}_bwd_sin_cos"
        )
        node_a = Node(
            "Mul",
            [grad_out, cos_name],
            [grad_a_name],
            {},
            name=f"{fwd_node.name}_bwd_sin_a",
        )
        return [node_cos, node_a], [grad_a_name]


class CosVJP(VJPRule):
    """Autograd rule for the Cos operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"

        sin_name = f"sin_{in_a}_wrt_{fwd_node.name}"
        node_sin = Node(
            "Sin", [in_a], [sin_name], {}, name=f"{fwd_node.name}_bwd_cos_sin"
        )
        neg_sin_name = f"neg_{sin_name}"
        node_neg = Node(
            "Neg", [sin_name], [neg_sin_name], {}, name=f"{fwd_node.name}_bwd_cos_neg"
        )
        node_a = Node(
            "Mul",
            [grad_out, neg_sin_name],
            [grad_a_name],
            {},
            name=f"{fwd_node.name}_bwd_cos_a",
        )
        return [node_sin, node_neg, node_a], [grad_a_name]


class TanVJP(VJPRule):
    """Autograd rule for the Tan operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"

        node_a = Node(
            "TanGrad",
            [grad_out, in_a],
            [grad_a_name],
            {},
            name=f"{fwd_node.name}_bwd_tan_a",
        )
        return [node_a], [grad_a_name]


class AsinVJP(VJPRule):
    """Autograd rule for the Asin operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        node_a = Node(
            "AsinGrad",
            [grad_out, in_a],
            [grad_a_name],
            {},
            name=f"{fwd_node.name}_bwd_asin_a",
        )
        return [node_a], [grad_a_name]


class AcosVJP(VJPRule):
    """Autograd rule for the Acos operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        node_a = Node(
            "AcosGrad",
            [grad_out, in_a],
            [grad_a_name],
            {},
            name=f"{fwd_node.name}_bwd_acos_a",
        )
        return [node_a], [grad_a_name]


class AtanVJP(VJPRule):
    """Autograd rule for the Atan operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        node_a = Node(
            "AtanGrad",
            [grad_out, in_a],
            [grad_a_name],
            {},
            name=f"{fwd_node.name}_bwd_atan_a",
        )
        return [node_a], [grad_a_name]


class SinhVJP(VJPRule):
    """Autograd rule for the Sinh operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        node_cosh = Node(
            "Cosh", [in_a], [f"cosh_{in_a}"], {}, name=f"{fwd_node.name}_bwd_sinh_cosh"
        )
        node_a = Node(
            "Mul",
            [grad_out, f"cosh_{in_a}"],
            [grad_a_name],
            {},
            name=f"{fwd_node.name}_bwd_sinh_a",
        )
        return [node_cosh, node_a], [grad_a_name]


class CoshVJP(VJPRule):
    """Autograd rule for the Cosh operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        node_sinh = Node(
            "Sinh", [in_a], [f"sinh_{in_a}"], {}, name=f"{fwd_node.name}_bwd_cosh_sinh"
        )
        node_a = Node(
            "Mul",
            [grad_out, f"sinh_{in_a}"],
            [grad_a_name],
            {},
            name=f"{fwd_node.name}_bwd_cosh_a",
        )
        return [node_sinh, node_a], [grad_a_name]


class AsinhVJP(VJPRule):
    """Autograd rule for the Asinh operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        node_a = Node(
            "AsinhGrad",
            [grad_out, in_a],
            [grad_a_name],
            {},
            name=f"{fwd_node.name}_bwd_asinh_a",
        )
        return [node_a], [grad_a_name]


class AcoshVJP(VJPRule):
    """Autograd rule for the Acosh operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        node_a = Node(
            "AcoshGrad",
            [grad_out, in_a],
            [grad_a_name],
            {},
            name=f"{fwd_node.name}_bwd_acosh_a",
        )
        return [node_a], [grad_a_name]


class AtanhVJP(VJPRule):
    """Autograd rule for the Atanh operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        node_a = Node(
            "AtanhGrad",
            [grad_out, in_a],
            [grad_a_name],
            {},
            name=f"{fwd_node.name}_bwd_atanh_a",
        )
        return [node_a], [grad_a_name]


class ErfVJP(VJPRule):
    """Autograd rule for the Erf operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        node_a = Node(
            "ErfGrad",
            [grad_out, in_a],
            [grad_a_name],
            {},
            name=f"{fwd_node.name}_bwd_erf_a",
        )
        return [node_a], [grad_a_name]


class IsNaNVJP(VJPRule):
    """Autograd rule for the IsNaN operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        in_a = fwd_node.inputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        node_a = Node(
            "ConstantOfShape",
            [in_a],
            [grad_a_name],
            {"value": 0.0},
            name=f"{fwd_node.name}_bwd_isnan_a",
        )
        return [node_a], [grad_a_name]


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


class LeakyReluVJP(VJPRule):
    """Autograd rule for the LeakyRelu operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        alpha = fwd_node.attributes.get("alpha", 0.01)
        node_a = Node(
            "LeakyReluGrad",
            [grad_out, in_a],
            [grad_a_name],
            {"alpha": alpha},
            name=f"{fwd_node.name}_bwd_leakyrelu_a",
        )
        return [node_a], [grad_a_name]


class EluVJP(VJPRule):
    """Autograd rule for the Elu operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        alpha = fwd_node.attributes.get("alpha", 1.0)
        node_a = Node(
            "EluGrad",
            [grad_out, in_a],
            [grad_a_name],
            {"alpha": alpha},
            name=f"{fwd_node.name}_bwd_elu_a",
        )
        return [node_a], [grad_a_name]


class SeluVJP(VJPRule):
    """Autograd rule for the Selu operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        alpha = fwd_node.attributes.get("alpha", 1.67326)
        gamma = fwd_node.attributes.get("gamma", 1.0507)
        node_a = Node(
            "SeluGrad",
            [grad_out, in_a],
            [grad_a_name],
            {"alpha": alpha, "gamma": gamma},
            name=f"{fwd_node.name}_bwd_selu_a",
        )
        return [node_a], [grad_a_name]


class SoftplusVJP(VJPRule):
    """Autograd rule for the Softplus operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        node_a = Node(
            "SoftplusGrad",
            [grad_out, in_a],
            [grad_a_name],
            {},
            name=f"{fwd_node.name}_bwd_softplus_a",
        )
        return [node_a], [grad_a_name]


class SoftsignVJP(VJPRule):
    """Autograd rule for the Softsign operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        node_a = Node(
            "SoftsignGrad",
            [grad_out, in_a],
            [grad_a_name],
            {},
            name=f"{fwd_node.name}_bwd_softsign_a",
        )
        return [node_a], [grad_a_name]


class HardSigmoidVJP(VJPRule):
    """Autograd rule for the HardSigmoid operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        alpha = fwd_node.attributes.get("alpha", 0.2)
        beta = fwd_node.attributes.get("beta", 0.5)
        node_a = Node(
            "HardSigmoidGrad",
            [grad_out, in_a],
            [grad_a_name],
            {"alpha": alpha, "beta": beta},
            name=f"{fwd_node.name}_bwd_hardsigmoid_a",
        )
        return [node_a], [grad_a_name]


class HardSwishVJP(VJPRule):
    """Autograd rule for the HardSwish operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        node_a = Node(
            "HardSwishGrad",
            [grad_out, in_a],
            [grad_a_name],
            {},
            name=f"{fwd_node.name}_bwd_hardswish_a",
        )
        return [node_a], [grad_a_name]


class GeluVJP(VJPRule):
    """Autograd rule for the Gelu operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        node_a = Node(
            "GeluGrad",
            [grad_out, in_a],
            [grad_a_name],
            fwd_node.attributes,
            name=f"{fwd_node.name}_bwd_gelu_a",
        )
        return [node_a], [grad_a_name]


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
        return [node_a], [grad_a_name]


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
        return [node_a], [grad_a_name]


class ReduceSumVJP(VJPRule):
    """Autograd rule for the ReduceSum operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"

        # In reality, this requires handling keepdims, reshaping if necessary, and then expanding.
        # Here we emit a mock 'Expand' to represent the broadcasting back.
        node_a = Node(
            "Expand",
            [grad_out, f"shape_{in_a}"],
            [grad_a_name],
            {},
            name=f"{fwd_node.name}_bwd_reducesum",
        )
        return [node_a], [grad_a_name]


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
        return [node_expand, node_a], [grad_a_name]


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

        # dL/dX = dL/dY * (X == Y)
        node_a = Node(
            "ReduceMaxGrad",
            [grad_out, in_a, out_name],
            [grad_a_name],
            fwd_node.attributes,
            name=f"{fwd_node.name}_bwd_reducemax",
        )
        return [node_a], [grad_a_name]


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
        return [node_a], [grad_a_name]


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
        return [node_a], [grad_a_name]


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
        return [node_a], [grad_a_name]


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
        return [node_a], [grad_a_name]


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
        return [node_a], [grad_a_name]


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
        return [node_a], [grad_a_name]


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
        return [node_a], [grad_a_name]


class PReluVJP(VJPRule):
    """Autograd rule for the PRelu operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        in_slope = fwd_node.inputs[1]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        grad_slope_name = f"grad_{in_slope}_wrt_{fwd_node.name}"
        node_a = Node(
            "PReluGrad",
            [grad_out, in_a, in_slope],
            [grad_a_name, grad_slope_name],
            {},
            name=f"{fwd_node.name}_bwd_prelu",
        )
        return [node_a], [grad_a_name, grad_slope_name]


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

        return nodes, names


class ConvTransposeVJP(VJPRule):
    """Autograd rule for the ConvTranspose operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        x = fwd_node.inputs[0]
        w = fwd_node.inputs[1]

        grad_x_name = f"grad_{x}_wrt_{fwd_node.name}"
        grad_w_name = f"grad_{w}_wrt_{fwd_node.name}"

        node_x = Node(
            "ConvTransposeGradX",
            [grad_out, w],
            [grad_x_name],
            fwd_node.attributes,
            name=f"{fwd_node.name}_bwd_convtrans_x",
        )
        node_w = Node(
            "ConvTransposeGradW",
            [grad_out, x],
            [grad_w_name],
            fwd_node.attributes,
            name=f"{fwd_node.name}_bwd_convtrans_w",
        )

        nodes = [node_x, node_w]
        names = [grad_x_name, grad_w_name]

        if len(fwd_node.inputs) > 2:
            b = fwd_node.inputs[2]
            grad_b_name = f"grad_{b}_wrt_{fwd_node.name}"
            node_b = Node(
                "ConvTransposeGradB",
                [grad_out],
                [grad_b_name],
                fwd_node.attributes,
                name=f"{fwd_node.name}_bwd_convtrans_b",
            )
            nodes.append(node_b)
            names.append(grad_b_name)

        return nodes, names


class GlobalAveragePoolVJP(VJPRule):
    """Autograd rule for the GlobalAveragePool operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        node_a = Node(
            "GlobalAveragePoolGrad",
            [grad_out, in_a],
            [grad_a_name],
            {},
            name=f"{fwd_node.name}_bwd_globalavgpool",
        )
        return [node_a], [grad_a_name]


class GlobalMaxPoolVJP(VJPRule):
    """Autograd rule for the GlobalMaxPool operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        out_name = fwd_node.outputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        node_a = Node(
            "GlobalMaxPoolGrad",
            [grad_out, in_a, out_name],
            [grad_a_name],
            {},
            name=f"{fwd_node.name}_bwd_globalmaxpool",
        )
        return [node_a], [grad_a_name]


class ReshapeVJP(VJPRule):
    """Autograd rule for the Reshape operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"

        # dL/dX = reshape(dL/dY, shape(X))
        shape_name = f"shape_{in_a}_wrt_{fwd_node.name}"
        node_shape = Node(
            "Shape", [in_a], [shape_name], {}, name=f"{fwd_node.name}_bwd_reshape_shape"
        )
        node_a = Node(
            "Reshape",
            [grad_out, shape_name],
            [grad_a_name],
            {},
            name=f"{fwd_node.name}_bwd_reshape_a",
        )

        return [node_shape, node_a], [grad_a_name]


class TransposeVJP(VJPRule):
    """Autograd rule for the Transpose operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"

        perm = fwd_node.attributes.get("perm", [])
        if perm:
            rev_perm = [0] * len(perm)
            for i, p in enumerate(perm):
                rev_perm[p] = i
            node_a = Node(
                "Transpose",
                [grad_out],
                [grad_a_name],
                {"perm": rev_perm},
                name=f"{fwd_node.name}_bwd_transpose_a",
            )
        else:
            node_a = Node(
                "Transpose",
                [grad_out],
                [grad_a_name],
                {},
                name=f"{fwd_node.name}_bwd_transpose_a",
            )

        return [node_a], [grad_a_name]


class SqueezeVJP(VJPRule):
    """Autograd rule for the Squeeze operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"

        shape_name = f"shape_{in_a}_wrt_{fwd_node.name}"
        node_shape = Node(
            "Shape", [in_a], [shape_name], {}, name=f"{fwd_node.name}_bwd_squeeze_shape"
        )
        node_a = Node(
            "Reshape",
            [grad_out, shape_name],
            [grad_a_name],
            {},
            name=f"{fwd_node.name}_bwd_squeeze_a",
        )

        return [node_shape, node_a], [grad_a_name]


class UnsqueezeVJP(VJPRule):
    """Autograd rule for the Unsqueeze operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"

        shape_name = f"shape_{in_a}_wrt_{fwd_node.name}"
        node_shape = Node(
            "Shape",
            [in_a],
            [shape_name],
            {},
            name=f"{fwd_node.name}_bwd_unsqueeze_shape",
        )
        node_a = Node(
            "Reshape",
            [grad_out, shape_name],
            [grad_a_name],
            {},
            name=f"{fwd_node.name}_bwd_unsqueeze_a",
        )

        return [node_shape, node_a], [grad_a_name]


class FlattenVJP(VJPRule):
    """Autograd rule for the Flatten operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"

        shape_name = f"shape_{in_a}_wrt_{fwd_node.name}"
        node_shape = Node(
            "Shape", [in_a], [shape_name], {}, name=f"{fwd_node.name}_bwd_flatten_shape"
        )
        node_a = Node(
            "Reshape",
            [grad_out, shape_name],
            [grad_a_name],
            {},
            name=f"{fwd_node.name}_bwd_flatten_a",
        )

        return [node_shape, node_a], [grad_a_name]


class ConcatVJP(VJPRule):
    """Autograd rule for the Concat operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        grad_names = [f"grad_{in_a}_wrt_{fwd_node.name}" for in_a in fwd_node.inputs]

        axis = fwd_node.attributes.get("axis", 0)

        # dL/dX_i = Split(dL/dY)
        split_node = Node(
            "Split",
            [grad_out],
            grad_names,
            {"axis": axis},
            name=f"{fwd_node.name}_bwd_concat_split",
        )

        return [split_node], grad_names


class SplitVJP(VJPRule):
    """Autograd rule for the Split operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        in_a = fwd_node.inputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"

        axis = fwd_node.attributes.get("axis", 0)
        concat_node = Node(
            "Concat",
            grad_outputs,
            [grad_a_name],
            {"axis": axis},
            name=f"{fwd_node.name}_bwd_split_concat",
        )

        return [concat_node], [grad_a_name]


class SliceVJP(VJPRule):
    """Autograd rule for the Slice operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"

        # This requires Pad or an explicit SliceGrad
        node_a = Node(
            "SliceGrad",
            [grad_out, in_a] + fwd_node.inputs[1:],
            [grad_a_name],
            {},
            name=f"{fwd_node.name}_bwd_slice",
        )

        return [node_a], [grad_a_name]


class GatherVJP(VJPRule):
    """Autograd rule for the Gather operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        indices = fwd_node.inputs[1]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"

        node_a = Node(
            "GatherGrad",
            [grad_out, indices, in_a],
            [grad_a_name],
            fwd_node.attributes,
            name=f"{fwd_node.name}_bwd_gather",
        )

        return [node_a], [grad_a_name]


class GatherElementsVJP(VJPRule):
    """Autograd rule for the GatherElements operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        indices = fwd_node.inputs[1]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"

        node_a = Node(
            "GatherElementsGrad",
            [grad_out, indices, in_a],
            [grad_a_name],
            fwd_node.attributes,
            name=f"{fwd_node.name}_bwd_gatherelements",
        )

        return [node_a], [grad_a_name]


class GatherNDVJP(VJPRule):
    """Autograd rule for the GatherND operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        indices = fwd_node.inputs[1]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"

        node_a = Node(
            "GatherNDGrad",
            [grad_out, indices, in_a],
            [grad_a_name],
            fwd_node.attributes,
            name=f"{fwd_node.name}_bwd_gathernd",
        )

        return [node_a], [grad_a_name]


class ScatterVJP(VJPRule):
    """Autograd rule for the Scatter operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        indices = fwd_node.inputs[1]
        updates = fwd_node.inputs[2]

        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        grad_updates_name = f"grad_{updates}_wrt_{fwd_node.name}"

        node_a = Node(
            "ScatterGradA",
            [grad_out, indices],
            [grad_a_name],
            fwd_node.attributes,
            name=f"{fwd_node.name}_bwd_scatter_a",
        )
        node_updates = Node(
            "ScatterGradUpdates",
            [grad_out, indices],
            [grad_updates_name],
            fwd_node.attributes,
            name=f"{fwd_node.name}_bwd_scatter_updates",
        )

        return [node_a, node_updates], [grad_a_name, grad_updates_name]


class ScatterNDVJP(VJPRule):
    """Autograd rule for the ScatterND operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        indices = fwd_node.inputs[1]
        updates = fwd_node.inputs[2]

        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        grad_updates_name = f"grad_{updates}_wrt_{fwd_node.name}"

        node_a = Node(
            "ScatterNDGradA",
            [grad_out, indices],
            [grad_a_name],
            fwd_node.attributes,
            name=f"{fwd_node.name}_bwd_scatternd_a",
        )
        node_updates = Node(
            "ScatterNDGradUpdates",
            [grad_out, indices],
            [grad_updates_name],
            fwd_node.attributes,
            name=f"{fwd_node.name}_bwd_scatternd_updates",
        )

        return [node_a, node_updates], [grad_a_name, grad_updates_name]


class ScatterElementsVJP(VJPRule):
    """Autograd rule for the ScatterElements operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        indices = fwd_node.inputs[1]
        updates = fwd_node.inputs[2]

        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        grad_updates_name = f"grad_{updates}_wrt_{fwd_node.name}"

        node_a = Node(
            "ScatterElementsGradA",
            [grad_out, indices],
            [grad_a_name],
            fwd_node.attributes,
            name=f"{fwd_node.name}_bwd_scatterelements_a",
        )
        node_updates = Node(
            "ScatterElementsGradUpdates",
            [grad_out, indices],
            [grad_updates_name],
            fwd_node.attributes,
            name=f"{fwd_node.name}_bwd_scatterelements_updates",
        )

        return [node_a, node_updates], [grad_a_name, grad_updates_name]


class TileVJP(VJPRule):
    """Autograd rule for the Tile operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        repeats = fwd_node.inputs[1]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"

        node_a = Node(
            "TileGrad",
            [grad_out, repeats],
            [grad_a_name],
            {},
            name=f"{fwd_node.name}_bwd_tile",
        )

        return [node_a], [grad_a_name]


class PadVJP(VJPRule):
    """Autograd rule for the Pad operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        pads = fwd_node.inputs[1]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"

        node_a = Node(
            "PadGrad",
            [grad_out, pads],
            [grad_a_name],
            fwd_node.attributes,
            name=f"{fwd_node.name}_bwd_pad",
        )

        return [node_a], [grad_a_name]


class CastVJP(VJPRule):
    """Autograd rule for the Cast operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"

        # We need the dtype of the original tensor, mock with CastGrad
        node_a = Node(
            "CastGrad",
            [grad_out, in_a],
            [grad_a_name],
            {},
            name=f"{fwd_node.name}_bwd_cast",
        )

        return [node_a], [grad_a_name]


class ExpandVJP(VJPRule):
    """Autograd rule for the Expand operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        shape = fwd_node.inputs[1]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"

        node_a = Node(
            "ExpandGrad",
            [grad_out, in_a, shape],
            [grad_a_name],
            {},
            name=f"{fwd_node.name}_bwd_expand",
        )

        return [node_a], [grad_a_name]


class WhereVJP(VJPRule):
    """Autograd rule for the Where operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        cond = fwd_node.inputs[0]
        in_x = fwd_node.inputs[1]
        in_y = fwd_node.inputs[2]

        grad_x_name = f"grad_{in_x}_wrt_{fwd_node.name}"
        grad_y_name = f"grad_{in_y}_wrt_{fwd_node.name}"

        node_x = Node(
            "Where",
            [cond, grad_out, "zeros_like_grad"],
            [grad_x_name],
            {},
            name=f"{fwd_node.name}_bwd_where_x",
        )
        node_y = Node(
            "Where",
            [cond, "zeros_like_grad", grad_out],
            [grad_y_name],
            {},
            name=f"{fwd_node.name}_bwd_where_y",
        )

        # Simplified for mock, assuming zeros are available or WhereGrad handles it
        node_combined = Node(
            "WhereGrad",
            [grad_out, cond],
            [grad_x_name, grad_y_name],
            {},
            name=f"{fwd_node.name}_bwd_where_combined",
        )

        return [node_combined], [grad_x_name, grad_y_name]


class NonZeroVJP(VJPRule):
    """Autograd rule for the NonZero operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        in_a = fwd_node.inputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"

        node_a = Node(
            "ConstantOfShape",
            [in_a],
            [grad_a_name],
            {"value": 0.0},
            name=f"{fwd_node.name}_bwd_nonzero_a",
        )

        return [node_a], [grad_a_name]


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
                [grad_out] + fwd_node.inputs,
                [grad_x_name, grad_scale_name, grad_b_name],
                fwd_node.attributes,
                name=f"{fwd_node.name}_bwd_layernorm",
            )
            nodes.append(node_grad)
            names.append(grad_b_name)
        else:
            node_grad = Node(
                "LayerNormalizationGrad",
                [grad_out] + fwd_node.inputs,
                [grad_x_name, grad_scale_name],
                fwd_node.attributes,
                name=f"{fwd_node.name}_bwd_layernorm",
            )
            nodes.append(node_grad)

        return nodes, names


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
            [grad_out] + fwd_node.inputs,
            [grad_x_name, grad_scale_name, grad_b_name],
            fwd_node.attributes,
            name=f"{fwd_node.name}_bwd_instancenorm",
        )
        return [node_grad], [grad_x_name, grad_scale_name, grad_b_name]


class DropoutVJP(VJPRule):
    """Autograd rule for the Dropout operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"

        mask_name = (
            fwd_node.outputs[1]
            if len(fwd_node.outputs) > 1
            else f"mask_{fwd_node.name}"
        )
        node_a = Node(
            "DropoutGrad",
            [grad_out, mask_name],
            [grad_a_name],
            fwd_node.attributes,
            name=f"{fwd_node.name}_bwd_dropout",
        )

        return [node_a], [grad_a_name]


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
    "Gemm": GemmVJP(),
    "ConvTranspose": ConvTransposeVJP(),
    "GlobalAveragePool": GlobalAveragePoolVJP(),
    "GlobalMaxPool": GlobalMaxPoolVJP(),
    "LayerNormalization": LayerNormalizationVJP(),
    "Reshape": ReshapeVJP(),
    "Transpose": TransposeVJP(),
    "Squeeze": SqueezeVJP(),
    "Unsqueeze": UnsqueezeVJP(),
    "Flatten": FlattenVJP(),
    "Concat": ConcatVJP(),
    "Split": SplitVJP(),
    "Slice": SliceVJP(),
    "Gather": GatherVJP(),
    "GatherElements": GatherElementsVJP(),
    "GatherND": GatherNDVJP(),
    "Scatter": ScatterVJP(),
    "ScatterND": ScatterNDVJP(),
    "ScatterElements": ScatterElementsVJP(),
    "Tile": TileVJP(),
    "Pad": PadVJP(),
    "Cast": CastVJP(),
    "Expand": ExpandVJP(),
    "Where": WhereVJP(),
    "NonZero": NonZeroVJP(),
    "InstanceNormalization": InstanceNormalizationVJP(),
    "Dropout": DropoutVJP(),
    "ReduceSum": ReduceSumVJP(),
    "ReduceMean": ReduceMeanVJP(),
    "ReduceMax": ReduceMaxVJP(),
    "ReduceMin": ReduceMinVJP(),
    "ReduceProd": ReduceProdVJP(),
    "ReduceL1": ReduceL1VJP(),
    "ReduceL2": ReduceL2VJP(),
    "ReduceLogSum": ReduceLogSumVJP(),
    "ReduceLogSumExp": ReduceLogSumExpVJP(),
    "ReduceSumSquare": ReduceSumSquareVJP(),
    "Conv": ConvVJP(),
    "MaxPool": MaxPoolVJP(),
    "AveragePool": AveragePoolVJP(),
    "Sigmoid": SigmoidVJP(),
    "Tanh": TanhVJP(),
    "Add": AddVJP(),
    "Sub": SubVJP(),
    "Div": DivVJP(),
    "Pow": PowVJP(),
    "Mod": ModVJP(),
    "Abs": AbsVJP(),
    "Neg": NegVJP(),
    "Sign": SignVJP(),
    "Exp": ExpVJP(),
    "Log": LogVJP(),
    "Sqrt": SqrtVJP(),
    "Sin": SinVJP(),
    "Cos": CosVJP(),
    "Tan": TanVJP(),
    "Asin": AsinVJP(),
    "Acos": AcosVJP(),
    "Atan": AtanVJP(),
    "Sinh": SinhVJP(),
    "Cosh": CoshVJP(),
    "Asinh": AsinhVJP(),
    "Acosh": AcoshVJP(),
    "Atanh": AtanhVJP(),
    "Erf": ErfVJP(),
    "IsNaN": IsNaNVJP(),
    "Mul": MulVJP(),
    "MatMul": MatMulVJP(),
    "Relu": ReluVJP(),
    "LeakyRelu": LeakyReluVJP(),
    "Elu": EluVJP(),
    "Selu": SeluVJP(),
    "Softplus": SoftplusVJP(),
    "Softsign": SoftsignVJP(),
    "HardSigmoid": HardSigmoidVJP(),
    "HardSwish": HardSwishVJP(),
    "Gelu": GeluVJP(),
    "Softmax": SoftmaxVJP(),
    "LogSoftmax": LogSoftmaxVJP(),
    "PRelu": PReluVJP(),
}


def get_vjp_rule(op_type: str) -> VJPRule:
    """Autograd rule for the get_vjp_rule operation."""

    if op_type not in _VJP_REGISTRY:
        return _VJP_REGISTRY.get("Relu")
    return _VJP_REGISTRY[op_type]
