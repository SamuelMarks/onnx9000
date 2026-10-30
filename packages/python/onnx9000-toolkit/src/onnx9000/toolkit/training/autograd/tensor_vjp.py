"""Tensor shape, slicing, indexing, gather/scatter, and sequence VJP rules."""

from __future__ import annotations

from onnx9000.core.ir import Node
from onnx9000.toolkit.training.autograd.elementwise_vjp import RoundVJP
from onnx9000.toolkit.training.autograd.vjp import VJPRule


class ReshapeVJP(VJPRule):
    """Autograd rule for the Reshape operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
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
        return ([node_shape, node_a], [grad_a_name])


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
        return ([node_a], [grad_a_name])


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
        return ([node_shape, node_a], [grad_a_name])


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
        return ([node_shape, node_a], [grad_a_name])


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
        return ([node_shape, node_a], [grad_a_name])


class ConcatVJP(VJPRule):
    """Autograd rule for the Concat operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        grad_names = [f"grad_{in_a}_wrt_{fwd_node.name}" for in_a in fwd_node.inputs]
        axis = fwd_node.attributes.get("axis", 0)
        split_node = Node(
            "Split",
            [grad_out],
            grad_names,
            {"axis": axis},
            name=f"{fwd_node.name}_bwd_concat_split",
        )
        return ([split_node], grad_names)


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
        return ([concat_node], [grad_a_name])


class SliceVJP(VJPRule):
    """Autograd rule for the Slice operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        node_a = Node(
            "SliceGrad",
            [grad_out, in_a, *fwd_node.inputs[1:]],
            [grad_a_name],
            {},
            name=f"{fwd_node.name}_bwd_slice",
        )
        return ([node_a], [grad_a_name])


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
        return ([node_a], [grad_a_name])


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
        return ([node_a], [grad_a_name])


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
        return ([node_a], [grad_a_name])


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
        return ([node_a, node_updates], [grad_a_name, grad_updates_name])


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
        return ([node_a, node_updates], [grad_a_name, grad_updates_name])


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
        return ([node_a, node_updates], [grad_a_name, grad_updates_name])


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
        return ([node_a], [grad_a_name])


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
        return ([node_a], [grad_a_name])


class CastVJP(VJPRule):
    """Autograd rule for the Cast operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        node_a = Node(
            "CastGrad",
            [grad_out, in_a],
            [grad_a_name],
            {},
            name=f"{fwd_node.name}_bwd_cast",
        )
        return ([node_a], [grad_a_name])


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
        return ([node_a], [grad_a_name])


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
        Node(
            "Where",
            [cond, grad_out, "zeros_like_grad"],
            [grad_x_name],
            {},
            name=f"{fwd_node.name}_bwd_where_x",
        )
        Node(
            "Where",
            [cond, "zeros_like_grad", grad_out],
            [grad_y_name],
            {},
            name=f"{fwd_node.name}_bwd_where_y",
        )
        node_combined = Node(
            "WhereGrad",
            [grad_out, cond],
            [grad_x_name, grad_y_name],
            {},
            name=f"{fwd_node.name}_bwd_where_combined",
        )
        return ([node_combined], [grad_x_name, grad_y_name])


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
        return ([node_a], [grad_a_name])


class SpaceToDepthVJP(VJPRule):
    """Autograd rule for the SpaceToDepth operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        node_a = Node(
            "DepthToSpace",
            [grad_out],
            [grad_a_name],
            fwd_node.attributes.copy(),
            name=f"{fwd_node.name}_bwd_spacetodepth",
        )
        return ([node_a], [grad_a_name])


class DepthToSpaceVJP(VJPRule):
    """Autograd rule for the DepthToSpace operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        node_a = Node(
            "SpaceToDepth",
            [grad_out],
            [grad_a_name],
            fwd_node.attributes.copy(),
            name=f"{fwd_node.name}_bwd_depthtospace",
        )
        return ([node_a], [grad_a_name])


class ReverseSequenceVJP(VJPRule):
    """Autograd rule for the ReverseSequence operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        seq_lens = fwd_node.inputs[1]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        node_a = Node(
            "ReverseSequence",
            [grad_out, seq_lens],
            [grad_a_name],
            fwd_node.attributes.copy(),
            name=f"{fwd_node.name}_bwd_reversesequence",
        )
        return ([node_a], [grad_a_name, f"grad_{seq_lens}_wrt_{fwd_node.name}"])


class CompressVJP(VJPRule):
    """Autograd rule for the Compress operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        condition = fwd_node.inputs[1]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        node_a = Node(
            "CompressGrad",
            [grad_out, in_a, condition],
            [grad_a_name],
            fwd_node.attributes.copy(),
            name=f"{fwd_node.name}_bwd_compress",
        )
        return ([node_a], [grad_a_name, f"grad_{condition}_wrt_{fwd_node.name}"])


class SpaceToBatchNDVJP(VJPRule):
    """Autograd rule for the SpaceToBatchND operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        node_a = Node(
            "BatchToSpaceND",
            [grad_out, *fwd_node.inputs[1:]],
            [grad_a_name],
            fwd_node.attributes.copy(),
            name=f"{fwd_node.name}_bwd_spacetobatchnd",
        )
        grads = [grad_a_name] + [f"grad_{inp}_wrt_{fwd_node.name}" for inp in fwd_node.inputs[1:]]
        return ([node_a], grads)


class BatchToSpaceNDVJP(VJPRule):
    """Autograd rule for the BatchToSpaceND operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct and attach backward nodes to the computational graph."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        grad_a_name = f"grad_{in_a}_wrt_{fwd_node.name}"
        node_a = Node(
            "SpaceToBatchND",
            [grad_out, *fwd_node.inputs[1:]],
            [grad_a_name],
            fwd_node.attributes.copy(),
            name=f"{fwd_node.name}_bwd_batchtospacend",
        )
        grads = [grad_a_name] + [f"grad_{inp}_wrt_{fwd_node.name}" for inp in fwd_node.inputs[1:]]
        return ([node_a], grads)


class SplitToSequenceVJP(VJPRule):
    """Autograd rule for the SplitToSequence operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct backward nodes to concatenate gradients."""
        grad_out = grad_outputs[0]
        in_a = fwd_node.inputs[0]
        g_name = f"grad_{in_a}_wrt_{fwd_node.name}"

        # ConcatFromSequence uses axis attribute or default 0.
        # SplitToSequence uses axis attribute.
        axis = fwd_node.attributes.get("axis", 0)

        concat_node = Node(
            "ConcatFromSequence",
            [grad_out],
            [g_name],
            {"axis": axis, "new_axis": 0},
            name=f"{fwd_node.name}_bwd_concat",
        )
        nodes = [concat_node]
        grads = [g_name]

        # If there are split sizes as the second input, its gradient is 0
        if len(fwd_node.inputs) > 1:
            split_in = fwd_node.inputs[1]
            split_g_name = f"grad_{split_in}_wrt_{fwd_node.name}"
            # Split input is a non-differentiable shape tensor, returning 0
            const_zero = Node(
                "ConstantOfShape",
                [split_in],
                [split_g_name],
                {"value": 0},
                name=f"{fwd_node.name}_bwd_split_zero",
            )
            nodes.append(const_zero)
            grads.append(split_g_name)

        return (nodes, grads)


class SequenceConstructVJP(VJPRule):
    """Autograd rule for the SequenceConstruct operation."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Construct backward nodes to slice gradients for SequenceConstruct."""
        grad_out = grad_outputs[0]
        nodes = []
        grads = []
        for i, in_x in enumerate(fwd_node.inputs):
            g_name = f"grad_{in_x}_wrt_{fwd_node.name}"
            # Create a constant index node
            idx_name = f"{fwd_node.name}_bwd_seq_idx_{i}"
            idx_node = Node(
                "Constant",
                [],
                [idx_name],
                {"value": i},
                name=f"{fwd_node.name}_const_{i}",
            )
            # Use SequenceAt to extract the gradient sequence element
            seq_at_node = Node(
                "SequenceAt",
                [grad_out, idx_name],
                [g_name],
                {},
                name=f"{fwd_node.name}_bwd_seq_at_{i}",
            )
            nodes.extend([idx_node, seq_at_node])
            grads.append(g_name)
        return (nodes, grads)


class ShapeVJP(RoundVJP):
    """Autograd rule for Shape (non-differentiable -> 0 gradient)."""

    def __init__(self) -> None:
        """Initialize ShapeVJP."""
        super().__init__()


class SizeVJP(RoundVJP):
    """Autograd rule for Size (non-differentiable -> 0 gradient)."""

    def __init__(self) -> None:
        """Initialize SizeVJP."""
        super().__init__()


class StopGradientVJP(RoundVJP):
    """Autograd rule for StopGradient (non-differentiable -> 0 gradient)."""

    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Build Backward Nodes function logic implementation."""
        nodes = []
        grads = []
        for i, in_x in enumerate(fwd_node.inputs):
            g_name = f"grad_{in_x}_wrt_{fwd_node.name}"
            n = Node(
                "ConstantOfShape",
                [in_x],
                [g_name],
                {"value": 0.0},
                name=f"{fwd_node.name}_bwd_bitshift_{i}",
            )
            nodes.append(n)
            grads.append(g_name)
        return (nodes, grads)
