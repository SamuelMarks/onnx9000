"""Module providing tensor ops functionality."""

from typing import Callable

from onnx9000.converters.tf.builder import TFToONNXGraphBuilder
from onnx9000.converters.tf.parsers import TFNode


def _map_identity(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map identity operation."""
    return builder.make_node("Identity", node.inputs, {}, node.name)


def _map_identity_n(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map identity n operation."""
    outs = []
    for i, inp in enumerate(node.inputs):
        outs.extend(builder.make_node("Identity", [inp], {}, f"{node.name}_{i}"))
    return outs


def _map_reshape(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map reshape operation."""
    return builder.make_node("Reshape", node.inputs, {}, node.name)


def _map_squeeze(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map squeeze operation."""
    axes = builder.extract_attr(node, "squeeze_dims", [])
    if axes:
        axes_tensor = builder.add_constant(f"{node.name}_axes", axes, 7, (len(axes),))
        return builder.make_node("Squeeze", [node.inputs[0], axes_tensor], {}, node.name)
    else:
        return builder.make_node("Squeeze", [node.inputs[0]], {}, node.name)


def _map_expand_dims(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map expand dims operation."""
    return builder.make_node("Unsqueeze", node.inputs, {}, node.name)


def _map_transpose(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map transpose operation."""
    return builder.make_node("Transpose", node.inputs, {}, node.name)


def _map_conjugate_transpose(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map conjugate transpose operation."""
    trans = builder.make_node("Transpose", node.inputs, {}, f"{node.name}_trans")[0]
    return builder.make_node("Custom_Conjugate", [trans], {}, node.name)


def _map_concat(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map concat operation."""
    return builder.make_node("Custom_TFConcat", node.inputs, {}, node.name)


def _map_pack(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map pack operation."""
    axis = builder.extract_attr(node, "axis", 0)
    unsqueezed = []
    axes_const = builder.add_constant(f"{node.name}_axes", [axis], 7, (1,))
    for i, inp in enumerate(node.inputs):
        u = builder.make_node("Unsqueeze", [inp, axes_const], {}, f"{node.name}_unsq_{i}")[0]
        unsqueezed.append(u)
    return builder.make_node("Concat", unsqueezed, {"axis": axis}, node.name)


def _map_unpack(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map unpack operation."""
    axis = builder.extract_attr(node, "axis", 0)
    num = builder.extract_attr(node, "num", len(node.inputs))
    splits = builder.make_node(
        "Split",
        [node.inputs[0]],
        {"axis": axis, "num_outputs": num},
        f"{node.name}_split",
        num_outputs=num,
    )
    axes_const = builder.add_constant(f"{node.name}_axes", [axis], 7, (1,))
    outs = []
    for i, s in enumerate(splits):
        sq = builder.make_node("Squeeze", [s, axes_const], {}, f"{node.name}_sq_{i}")[0]
        outs.append(sq)
    return outs


def _map_split(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map split operation."""
    num = builder.extract_attr(node, "num_split", 1)
    return builder.make_node(
        "Custom_TFSplit", node.inputs, {"num_outputs": num}, node.name, num_outputs=num
    )


def _map_slice(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map slice operation."""
    return builder.make_node("Slice", node.inputs, {}, node.name)


def _map_tile(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map tile operation."""
    return builder.make_node("Tile", node.inputs, {}, node.name)


def _map_pad(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map pad operation."""
    return builder.make_node("Pad", node.inputs, {"mode": "constant"}, node.name)


def _map_pad_v2(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map pad v2 operation."""
    return builder.make_node("Pad", node.inputs, {"mode": "constant"}, node.name)


def _map_mirror_pad(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map mirror pad operation."""
    mode_str = builder.extract_attr(node, "mode", b"REFLECT")
    if isinstance(mode_str, bytes):
        mode_str = mode_str.decode("utf-8")
    mode = "reflect" if mode_str == "REFLECT" else "symmetric"
    return builder.make_node("Pad", node.inputs, {"mode": mode}, node.name)


def _map_gather(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map gather operation."""
    return builder.make_node("Gather", node.inputs, {}, node.name)


def _map_gather_nd(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map gather nd operation."""
    return builder.make_node("GatherND", node.inputs, {}, node.name)


def _map_scatter_nd(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map scatter nd operation."""
    return builder.make_node("ScatterND", node.inputs, {}, node.name)


def _map_tensor_scatter_update(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map tensor scatter update operation."""
    return builder.make_node("ScatterND", node.inputs, {}, node.name)


def _map_tensor_scatter_add(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map tensor scatter add operation."""
    return builder.make_node("ScatterND", node.inputs, {"reduction": "add"}, node.name)


def _map_space_to_batch(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map space to batch operation."""
    return builder.make_node("SpaceToDepth", node.inputs, {}, node.name)


def _map_batch_to_space(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map batch to space operation."""
    return builder.make_node("DepthToSpace", node.inputs, {}, node.name)


def _map_reverse(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map reverse operation."""
    return builder.make_node("ReverseSequence", node.inputs, {}, node.name)


def _map_roll(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map roll operation."""
    return builder.make_node("Custom_Roll", node.inputs, {}, node.name)


def _map_matrix_diag(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map matrix diag operation."""
    return builder.make_node("Custom_MatrixDiag", node.inputs, {}, node.name)


def _map_cast(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map cast operation."""
    to_dtype = builder.extract_attr(node, "DstT", 1)
    return builder.make_node("Cast", node.inputs, {"to": to_dtype}, node.name)


def _map_bitcast(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map bitcast operation."""
    return builder.make_node("Cast", node.inputs, {}, node.name)


def _map_shape(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map shape operation."""
    return builder.make_node("Shape", node.inputs, {}, node.name)


def _map_shape_n(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map shape n operation."""
    outs = []
    for i, inp in enumerate(node.inputs):
        outs.extend(builder.make_node("Shape", [inp], {}, f"{node.name}_{i}"))
    return outs


def _map_size(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map size operation."""
    return builder.make_node("Size", node.inputs, {}, node.name)


def _map_rank(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map rank operation."""
    shape = builder.make_node("Shape", node.inputs, {}, f"{node.name}_shape")[0]
    return builder.make_node("Size", [shape], {}, node.name)


def _map_zeros_like(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map zeros like operation."""
    shape = builder.make_node("Shape", node.inputs, {}, f"{node.name}_shape")[0]
    return builder.make_node("ConstantOfShape", [shape], {"value": 0.0}, node.name)


def _map_ones_like(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map ones like operation."""
    shape = builder.make_node("Shape", node.inputs, {}, f"{node.name}_shape")[0]
    return builder.make_node("ConstantOfShape", [shape], {"value": 1.0}, node.name)


def _map_fill(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map fill operation."""
    return builder.make_node("Custom_Fill", node.inputs, {}, node.name)


def _map_broadcast_to(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map broadcast to operation."""
    return builder.make_node("Expand", node.inputs, {}, node.name)


def _map_where(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map where operation."""
    return builder.make_node("Where", node.inputs, {}, node.name)


def _map_select(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map select operation."""
    return builder.make_node("Where", node.inputs, {}, node.name)


TENSOR_OPS_MAPPING: dict[str, Callable[[TFToONNXGraphBuilder, TFNode], list[str]]] = {
    "Identity": _map_identity,
    "IdentityN": _map_identity_n,
    "Reshape": _map_reshape,
    "Squeeze": _map_squeeze,
    "ExpandDims": _map_expand_dims,
    "Transpose": _map_transpose,
    "ConjugateTranspose": _map_conjugate_transpose,
    "Concat": _map_concat,
    "ConcatV2": _map_concat,
    "Pack": _map_pack,
    "Unpack": _map_unpack,
    "Split": _map_split,
    "SplitV": _map_split,
    "Slice": _map_slice,
    "StridedSlice": _map_slice,
    "Tile": _map_tile,
    "Pad": _map_pad,
    "PadV2": _map_pad_v2,
    "MirrorPad": _map_mirror_pad,
    "Gather": _map_gather,
    "GatherV2": _map_gather,
    "GatherNd": _map_gather_nd,
    "ScatterNd": _map_scatter_nd,
    "TensorScatterUpdate": _map_tensor_scatter_update,
    "TensorScatterAdd": _map_tensor_scatter_add,
    "SpaceToBatchND": _map_space_to_batch,
    "BatchToSpaceND": _map_batch_to_space,
    "SpaceToDepth": _map_space_to_batch,
    "DepthToSpace": _map_batch_to_space,
    "Reverse": _map_reverse,
    "ReverseV2": _map_reverse,
    "Roll": _map_roll,
    "MatrixDiag": _map_matrix_diag,
    "MatrixDiagV2": _map_matrix_diag,
    "MatrixDiagV3": _map_matrix_diag,
    "MatrixSetDiag": _map_matrix_diag,
    "MatrixBandPart": _map_matrix_diag,
    "Cast": _map_cast,
    "Bitcast": _map_bitcast,
    "Shape": _map_shape,
    "ShapeN": _map_shape_n,
    "Size": _map_size,
    "Rank": _map_rank,
    "ZerosLike": _map_zeros_like,
    "OnesLike": _map_ones_like,
    "Fill": _map_fill,
    "BroadcastTo": _map_broadcast_to,
    "BroadcastArgs": _map_matrix_diag,
    "Where": _map_where,
    "Select": _map_select,
    "SelectV2": _map_select,
}
