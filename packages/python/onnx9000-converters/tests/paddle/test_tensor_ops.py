"""Tests the tensor ops module functionality."""

from onnx9000.converters.paddle.builder import PaddleToONNXGraphBuilder
from onnx9000.converters.paddle.parsers import PaddleNode
from onnx9000.converters.paddle.tensor_ops import TENSOR_OPS_MAPPING


def test_paddle_tensor_ops_reshape_transpose_flatten() -> None:
    """Tests the paddle tensor ops reshape transpose flatten functionality."""
    builder = PaddleToONNXGraphBuilder()
    n = PaddleNode("n", "reshape", inputs={"X": ["a"], "Shape": ["s"]})
    TENSOR_OPS_MAPPING["reshape"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Reshape"
    assert builder.graph.nodes[-1].inputs == ["a", "s"]
    n = PaddleNode("n", "reshape", inputs={"X": ["a"]}, attrs={"shape": [2, 3]})
    TENSOR_OPS_MAPPING["reshape"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Reshape"
    assert "n_shape_attr" in builder.graph.nodes[-1].inputs[-1]
    n = PaddleNode("n", "transpose2", inputs={"X": ["a"]}, attrs={"axis": [1, 0]})
    TENSOR_OPS_MAPPING["transpose2"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Transpose"
    assert builder.graph.nodes[-1].attributes["perm"] == [1, 0]
    n = PaddleNode("n", "flatten", inputs={"X": ["a"]}, attrs={"axis": 2})
    TENSOR_OPS_MAPPING["flatten"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Flatten"
    assert builder.graph.nodes[-1].attributes["axis"] == 2


def test_paddle_tensor_ops_squeeze_unsqueeze() -> None:
    """Tests the paddle tensor ops squeeze unsqueeze functionality."""
    builder = PaddleToONNXGraphBuilder()
    n = PaddleNode("n", "squeeze", inputs={"X": ["a"]}, attrs={"axes": [1]})
    TENSOR_OPS_MAPPING["squeeze"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Squeeze"
    assert "n_axes" in builder.graph.nodes[-1].inputs[-1]
    n = PaddleNode("n", "unsqueeze", inputs={"X": ["a"]}, attrs={"axes": [1]})
    TENSOR_OPS_MAPPING["unsqueeze"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Unsqueeze"


def test_paddle_tensor_ops_concat_stack() -> None:
    """Tests the paddle tensor ops concat stack functionality."""
    builder = PaddleToONNXGraphBuilder()
    n = PaddleNode("n", "concat", inputs={"X": ["a", "b"]}, attrs={"axis": 1})
    outs = TENSOR_OPS_MAPPING["concat"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Concat"
    assert builder.graph.nodes[-1].attributes["axis"] == 1
    n = PaddleNode("n", "stack", inputs={"X": ["a", "b"]})
    outs = TENSOR_OPS_MAPPING["stack"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Concat"
    assert builder.graph.nodes[-2].op_type == "Unsqueeze"
    n = PaddleNode(
        "n", "unstack", inputs={"X": ["a"]}, outputs={"Y": ["o1", "o2"]}, attrs={"axis": 0}
    )
    outs = TENSOR_OPS_MAPPING["unstack"](builder, n)
    assert outs == ["o1", "o2"]
    assert builder.graph.nodes[-1].op_type == "Squeeze"
    n = PaddleNode("n", "split", inputs={"X": ["a"]}, attrs={"num": 2})
    outs = TENSOR_OPS_MAPPING["split"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Split"
    assert builder.graph.nodes[-1].attributes["num_outputs"] == 2


def test_paddle_tensor_ops_slice_gather_scatter() -> None:
    """Tests the paddle tensor ops slice gather scatter functionality."""
    builder = PaddleToONNXGraphBuilder()
    n = PaddleNode("n", "slice", inputs={"X": ["a"]})
    TENSOR_OPS_MAPPING["slice"](builder, n)
    assert (
        builder.graph.nodes[-1].op_type == "Identity"
        or builder.graph.nodes[-1].op_type == "Squeeze"
        or builder.graph.nodes[-1].op_type == "Slice"
    )
    n = PaddleNode("n", "gather", inputs={"X": ["a"], "Index": ["i"]}, attrs={"axis": 1})
    TENSOR_OPS_MAPPING["gather"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Gather"
    assert builder.graph.nodes[-1].attributes["axis"] == 1
    n = PaddleNode("n", "gather_nd", inputs={"X": ["a"], "Index": ["i"]})
    TENSOR_OPS_MAPPING["gather_nd"](builder, n)
    assert builder.graph.nodes[-1].op_type == "GatherND"
    n = PaddleNode("n", "scatter", inputs={"X": ["a"], "Index": ["i"], "Updates": ["u"]})
    TENSOR_OPS_MAPPING["scatter"](builder, n)
    assert builder.graph.nodes[-1].op_type == "ScatterElements"
    n = PaddleNode("n", "scatter_nd", inputs={"X": ["a"], "Index": ["i"], "Updates": ["u"]})
    TENSOR_OPS_MAPPING["scatter_nd"](builder, n)
    assert builder.graph.nodes[-1].op_type == "ScatterND"
    n = PaddleNode("n", "scatter_nd_add", inputs={"X": ["a"], "Index": ["i"], "Updates": ["u"]})
    TENSOR_OPS_MAPPING["scatter_nd_add"](builder, n)
    assert builder.graph.nodes[-1].op_type == "ScatterND"
    assert builder.graph.nodes[-1].attributes["reduction"] == "add"


def test_paddle_tensor_ops_misc() -> None:
    """Tests the paddle tensor ops misc functionality."""
    builder = PaddleToONNXGraphBuilder()
    n = PaddleNode("n", "tile", inputs={"X": ["a"], "RepeatTimes": ["rt"]})
    TENSOR_OPS_MAPPING["tile"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Tile"
    n = PaddleNode("n", "expand", inputs={"X": ["a"], "Shape": ["s"]})
    TENSOR_OPS_MAPPING["expand"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Expand"
    n = PaddleNode("n", "expand_as", inputs={"X": ["a"], "Y": ["y"]})
    TENSOR_OPS_MAPPING["expand_as"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Expand"
    assert builder.graph.nodes[-2].op_type == "Shape"
    n = PaddleNode("n", "cast", inputs={"X": ["a"]}, attrs={"out_dtype": 5})
    TENSOR_OPS_MAPPING["cast"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Cast"
    n = PaddleNode("n", "shape", inputs={"Input": ["a"]})
    TENSOR_OPS_MAPPING["shape"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Shape"
    n = PaddleNode("n", "size", inputs={"Input": ["a"]})
    TENSOR_OPS_MAPPING["size"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Size"


def test_paddle_tensor_ops_constants() -> None:
    """Tests the paddle tensor ops constants functionality."""
    builder = PaddleToONNXGraphBuilder()
    n = PaddleNode("n", "fill_constant", attrs={"shape": [2], "value": 3.0})
    outs = TENSOR_OPS_MAPPING["fill_constant"](builder, n)
    assert builder.graph.nodes[-1].op_type == "ConstantOfShape"
    n = PaddleNode("n", "zeros_like", inputs={"X": ["a"]})
    outs = TENSOR_OPS_MAPPING["zeros_like"](builder, n)
    assert builder.graph.nodes[-1].op_type == "ConstantOfShape"
    assert builder.graph.nodes[-1].attributes["value"] == 0.0
    n = PaddleNode("n", "range", inputs={"Start": ["s"], "End": ["e"], "Step": ["st"]})
    outs = TENSOR_OPS_MAPPING["range"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Range"
    n = PaddleNode("n", "assign_value", attrs={"values": [1.0], "shape": [1]})
    outs = TENSOR_OPS_MAPPING["assign_value"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Identity"
    n = PaddleNode("n", "assign", inputs={"X": ["a"]})
    outs = TENSOR_OPS_MAPPING["assign"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Identity"
    n = PaddleNode("n", "where", inputs={"Condition": ["c"], "X": ["x"], "Y": ["y"]})
    outs = TENSOR_OPS_MAPPING["where"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Where"
    n = PaddleNode("n", "nonzero", inputs={"Condition": ["c"]})
    outs = TENSOR_OPS_MAPPING["nonzero"](builder, n)
    assert builder.graph.nodes[-1].op_type == "NonZero"
    for op in [
        "reshape2",
        "transpose2",
        "flatten_contiguous_range",
        "squeeze2",
        "unsqueeze2",
        "strided_slice",
        "crop_tensor",
        "index_select",
        "expand_v2",
        "fill_constant_batch_size_like",
        "fill_any_like",
        "ones_like",
        "arange",
        "where_index",
    ]:
        n = PaddleNode("n", op, inputs={"X": ["a"]})
        outs = TENSOR_OPS_MAPPING[op](builder, n)
        assert len(outs) > 0


def test_paddle_tensor_ops_slice_input_fb() -> None:
    """Tests the paddle tensor ops slice input fb functionality."""
    from onnx9000.converters.paddle.builder import PaddleToONNXGraphBuilder
    from onnx9000.converters.paddle.parsers import PaddleNode
    from onnx9000.converters.paddle.tensor_ops import TENSOR_OPS_MAPPING

    builder = PaddleToONNXGraphBuilder()
    n = PaddleNode("n", "slice", inputs={"Input": ["a"]}, attrs={"decrease_axis": [0]})
    TENSOR_OPS_MAPPING["slice"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Squeeze"


def test_tensor_ops_custom() -> None:
    """Tests the tensor ops custom functionality."""
    builder = PaddleToONNXGraphBuilder()
    n = PaddleNode("n", "unique", inputs={"X": ["a"]})
    TENSOR_OPS_MAPPING["unique"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Unique"
    n2 = PaddleNode("n2", "meshgrid", inputs={"X": ["a"], "Y": ["b"]})
    TENSOR_OPS_MAPPING["meshgrid"](builder, n2)
    assert builder.graph.nodes[-1].op_type == "Custom_Paddle_meshgrid"


def test_tensor_ops_custom2() -> None:
    """Tests the tensor ops custom2 functionality."""
    builder = PaddleToONNXGraphBuilder()
    n = PaddleNode("n", "gru_unit", inputs={"X": ["a"], "Y": ["b"]})
    TENSOR_OPS_MAPPING["gru_unit"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Custom_Paddle_gru_unit"
