from onnx9000.frontend.tf.builder import TFToONNXGraphBuilder
from onnx9000.frontend.tf.parsers import TFNode
from onnx9000.frontend.tf.tensor_ops import TENSOR_OPS_MAPPING


def test_tensor_ops_mapping_simple() -> None:
    builder = TFToONNXGraphBuilder()
    node = TFNode("n1", "Identity", inputs=["a"])
    outs = TENSOR_OPS_MAPPING["Identity"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Identity"
    node = TFNode("n2", "IdentityN", inputs=["a", "b"])
    outs = TENSOR_OPS_MAPPING["IdentityN"](builder, node)
    assert len(outs) == 2
    assert builder.graph.nodes[-1].op_type == "Identity"
    assert builder.graph.nodes[-2].op_type == "Identity"


def test_tensor_ops_reshape_squeeze_expand() -> None:
    builder = TFToONNXGraphBuilder()
    TENSOR_OPS_MAPPING["Reshape"](builder, TFNode("n", "Reshape", inputs=["a", "b"]))
    assert builder.graph.nodes[-1].op_type == "Reshape"
    TENSOR_OPS_MAPPING["Squeeze"](builder, TFNode("n", "Squeeze", inputs=["a"]))
    assert builder.graph.nodes[-1].op_type == "Squeeze"
    assert len(builder.graph.nodes[-1].inputs) == 1
    TENSOR_OPS_MAPPING["Squeeze"](
        builder, TFNode("n", "Squeeze", inputs=["a"], attr={"squeeze_dims": [1, 2]})
    )
    assert builder.graph.nodes[-1].op_type == "Squeeze"
    assert len(builder.graph.nodes[-1].inputs) == 2
    TENSOR_OPS_MAPPING["ExpandDims"](builder, TFNode("n", "ExpandDims", inputs=["a", "dim"]))
    assert builder.graph.nodes[-1].op_type == "Unsqueeze"


def test_tensor_ops_transpose() -> None:
    builder = TFToONNXGraphBuilder()
    TENSOR_OPS_MAPPING["Transpose"](builder, TFNode("n", "Transpose", inputs=["a", "perm"]))
    assert builder.graph.nodes[-1].op_type == "Transpose"
    TENSOR_OPS_MAPPING["ConjugateTranspose"](
        builder, TFNode("n", "ConjugateTranspose", inputs=["a", "perm"])
    )
    assert builder.graph.nodes[-1].op_type == "Custom_Conjugate"
    assert builder.graph.nodes[-2].op_type == "Transpose"


def test_tensor_ops_concat_pack_unpack() -> None:
    builder = TFToONNXGraphBuilder()
    outs = TENSOR_OPS_MAPPING["Concat"](builder, TFNode("n", "Concat", inputs=["dim", "a", "b"]))
    assert builder.graph.nodes[-1].op_type == "Custom_TFConcat"
    outs = TENSOR_OPS_MAPPING["Pack"](
        builder, TFNode("n", "Pack", inputs=["a", "b"], attr={"axis": 1})
    )
    assert builder.graph.nodes[-1].op_type == "Concat"
    assert builder.graph.nodes[-1].attributes["axis"] == 1
    assert builder.graph.nodes[-2].op_type == "Unsqueeze"
    outs = TENSOR_OPS_MAPPING["Unpack"](
        builder, TFNode("n", "Unpack", inputs=["a"], attr={"axis": 1, "num": 2})
    )
    assert len(outs) == 2
    assert builder.graph.nodes[-1].op_type == "Squeeze"


def test_tensor_ops_split_slice_tile() -> None:
    builder = TFToONNXGraphBuilder()
    TENSOR_OPS_MAPPING["Split"](
        builder, TFNode("n", "Split", inputs=["dim", "a"], attr={"num_split": 3})
    )
    assert builder.graph.nodes[-1].op_type == "Custom_TFSplit"
    TENSOR_OPS_MAPPING["Slice"](builder, TFNode("n", "Slice", inputs=["a", "b", "c"]))
    assert builder.graph.nodes[-1].op_type == "Slice"
    TENSOR_OPS_MAPPING["Tile"](builder, TFNode("n", "Tile", inputs=["a", "b"]))
    assert builder.graph.nodes[-1].op_type == "Tile"


def test_tensor_ops_pad() -> None:
    builder = TFToONNXGraphBuilder()
    TENSOR_OPS_MAPPING["Pad"](builder, TFNode("n", "Pad", inputs=["a", "b"]))
    assert builder.graph.nodes[-1].op_type == "Pad"
    assert builder.graph.nodes[-1].attributes["mode"] == "constant"
    TENSOR_OPS_MAPPING["PadV2"](builder, TFNode("n", "PadV2", inputs=["a", "b", "c"]))
    assert builder.graph.nodes[-1].op_type == "Pad"
    assert builder.graph.nodes[-1].attributes["mode"] == "constant"
    TENSOR_OPS_MAPPING["MirrorPad"](
        builder, TFNode("n", "MirrorPad", inputs=["a", "b"], attr={"mode": b"REFLECT"})
    )
    assert builder.graph.nodes[-1].op_type == "Pad"
    assert builder.graph.nodes[-1].attributes["mode"] == "reflect"
    TENSOR_OPS_MAPPING["MirrorPad"](
        builder, TFNode("n", "MirrorPad", inputs=["a", "b"], attr={"mode": "SYMMETRIC"})
    )
    assert builder.graph.nodes[-1].op_type == "Pad"
    assert builder.graph.nodes[-1].attributes["mode"] == "symmetric"


def test_tensor_ops_gather_scatter() -> None:
    builder = TFToONNXGraphBuilder()
    for op, onnx_op in [
        ("Gather", "Gather"),
        ("GatherNd", "GatherND"),
        ("ScatterNd", "ScatterND"),
        ("TensorScatterUpdate", "ScatterND"),
    ]:
        TENSOR_OPS_MAPPING[op](builder, TFNode(f"n_{op}", op, inputs=["a", "b"]))
        assert builder.graph.nodes[-1].op_type == onnx_op
    TENSOR_OPS_MAPPING["TensorScatterAdd"](
        builder, TFNode("n", "TensorScatterAdd", inputs=["a", "b", "c"])
    )
    assert builder.graph.nodes[-1].op_type == "ScatterND"
    assert builder.graph.nodes[-1].attributes["reduction"] == "add"


def test_tensor_ops_space_depth() -> None:
    builder = TFToONNXGraphBuilder()
    TENSOR_OPS_MAPPING["SpaceToBatchND"](
        builder, TFNode("n", "SpaceToBatchND", inputs=["a", "b", "c"])
    )
    assert builder.graph.nodes[-1].op_type == "SpaceToDepth"
    TENSOR_OPS_MAPPING["BatchToSpaceND"](
        builder, TFNode("n", "BatchToSpaceND", inputs=["a", "b", "c"])
    )
    assert builder.graph.nodes[-1].op_type == "DepthToSpace"


def test_tensor_ops_reverse_roll_diag() -> None:
    builder = TFToONNXGraphBuilder()
    TENSOR_OPS_MAPPING["Reverse"](builder, TFNode("n", "Reverse", inputs=["a", "b"]))
    assert builder.graph.nodes[-1].op_type == "ReverseSequence"
    TENSOR_OPS_MAPPING["Roll"](builder, TFNode("n", "Roll", inputs=["a", "b", "c"]))
    assert builder.graph.nodes[-1].op_type == "Custom_Roll"
    TENSOR_OPS_MAPPING["MatrixDiag"](builder, TFNode("n", "MatrixDiag", inputs=["a"]))
    assert builder.graph.nodes[-1].op_type == "Custom_MatrixDiag"


def test_tensor_ops_cast_shape_size() -> None:
    builder = TFToONNXGraphBuilder()
    outs = TENSOR_OPS_MAPPING["Cast"](builder, TFNode("n", "Cast", inputs=["a"], attr={"DstT": 3}))
    assert builder.graph.nodes[-1].op_type == "Cast"
    assert builder.graph.nodes[-1].attributes["to"] == 3
    outs = TENSOR_OPS_MAPPING["Bitcast"](builder, TFNode("n", "Bitcast", inputs=["a"]))
    assert builder.graph.nodes[-1].op_type == "Cast"
    outs = TENSOR_OPS_MAPPING["Shape"](builder, TFNode("n", "Shape", inputs=["a"]))
    assert builder.graph.nodes[-1].op_type == "Shape"
    outs = TENSOR_OPS_MAPPING["ShapeN"](builder, TFNode("n", "ShapeN", inputs=["a", "b"]))
    assert len(outs) == 2
    assert builder.graph.nodes[-1].op_type == "Shape"
    outs = TENSOR_OPS_MAPPING["Size"](builder, TFNode("n", "Size", inputs=["a"]))
    assert builder.graph.nodes[-1].op_type == "Size"
    outs = TENSOR_OPS_MAPPING["Rank"](builder, TFNode("n", "Rank", inputs=["a"]))
    assert builder.graph.nodes[-1].op_type == "Size"
    assert builder.graph.nodes[-2].op_type == "Shape"


def test_tensor_ops_like_fill_broadcast_where() -> None:
    builder = TFToONNXGraphBuilder()
    TENSOR_OPS_MAPPING["ZerosLike"](builder, TFNode("n", "ZerosLike", inputs=["a"]))
    assert builder.graph.nodes[-1].op_type == "ConstantOfShape"
    assert builder.graph.nodes[-1].attributes["value"] == 0.0
    TENSOR_OPS_MAPPING["OnesLike"](builder, TFNode("n", "OnesLike", inputs=["a"]))
    assert builder.graph.nodes[-1].op_type == "ConstantOfShape"
    assert builder.graph.nodes[-1].attributes["value"] == 1.0
    TENSOR_OPS_MAPPING["Fill"](builder, TFNode("n", "Fill", inputs=["a", "b"]))
    assert builder.graph.nodes[-1].op_type == "Custom_Fill"
    TENSOR_OPS_MAPPING["BroadcastTo"](builder, TFNode("n", "BroadcastTo", inputs=["a", "b"]))
    assert builder.graph.nodes[-1].op_type == "Expand"
    TENSOR_OPS_MAPPING["Where"](builder, TFNode("n", "Where", inputs=["a"]))
    assert builder.graph.nodes[-1].op_type == "Where"
    TENSOR_OPS_MAPPING["Select"](builder, TFNode("n", "Select", inputs=["a", "b", "c"]))
    assert builder.graph.nodes[-1].op_type == "Where"
    TENSOR_OPS_MAPPING["SelectV2"](builder, TFNode("n", "SelectV2", inputs=["a", "b", "c"]))
    assert builder.graph.nodes[-1].op_type == "Where"
