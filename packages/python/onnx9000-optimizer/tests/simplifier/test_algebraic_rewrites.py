"""Tests the algebraic rewrites module functionality."""

import numpy as np
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.optimizer.simplifier.passes.dce import IdentityEliminationPass


def test_reshape_shape():
    """Tests the reshape shape functionality."""
    g = Graph("mock")
    g.add_node(Node("Shape", ["x"], ["shape"], {}, "s"))
    g.add_node(Node("Reshape", ["x", "shape"], ["y"], {}, "r"))
    g.outputs = ["y"]
    IdentityEliminationPass().run(g)
    assert g.outputs[0] == "x"


def test_expand_shape():
    """Tests the expand shape functionality."""
    g = Graph("mock")
    g.add_node(Node("Shape", ["x"], ["shape"], {}, "s"))
    g.add_node(Node("Expand", ["x", "shape"], ["y"], {}, "e"))
    g.outputs = ["y"]
    IdentityEliminationPass().run(g)
    assert g.outputs[0] == "x"


def test_pad_zero():
    """Tests the pad zero functionality."""
    g = Graph("mock")
    pads_val = np.array([0, 0, 0, 0], dtype=np.int64)
    g.tensors["pads"] = Tensor(
        "pads", shape=(4,), dtype=DType.INT64, data=pads_val, is_initializer=True
    )
    g.add_node(Node("Pad", ["x", "pads"], ["y"], {}, "p"))
    g.outputs = ["y"]
    IdentityEliminationPass().run(g)
    assert g.outputs[0] == "x"


def test_tile_ones():
    """Tests the tile ones functionality."""
    g = Graph("mock")
    repeats_val = np.array([1, 1, 1], dtype=np.int64)
    g.tensors["reps"] = Tensor(
        "reps", shape=(3,), dtype=DType.INT64, data=repeats_val, is_initializer=True
    )
    g.add_node(Node("Tile", ["x", "reps"], ["y"], {}, "t"))
    g.outputs = ["y"]
    IdentityEliminationPass().run(g)
    assert g.outputs[0] == "x"


def test_slice_max_int():
    """Tests the slice max int functionality."""
    g = Graph("mock")
    starts = np.array([0], dtype=np.int64)
    ends = np.array([2147483647], dtype=np.int64)
    axes = np.array([0], dtype=np.int64)
    steps = np.array([1], dtype=np.int64)
    g.tensors["starts"] = Tensor(
        "starts", shape=(1,), dtype=DType.INT64, data=starts, is_initializer=True
    )
    g.tensors["ends"] = Tensor(
        "ends", shape=(1,), dtype=DType.INT64, data=ends, is_initializer=True
    )
    g.tensors["axes"] = Tensor(
        "axes", shape=(1,), dtype=DType.INT64, data=axes, is_initializer=True
    )
    g.tensors["steps"] = Tensor(
        "steps", shape=(1,), dtype=DType.INT64, data=steps, is_initializer=True
    )
    g.add_node(Node("Slice", ["x", "starts", "ends", "axes", "steps"], ["y"], {}, "s"))
    g.outputs = ["y"]
    IdentityEliminationPass().run(g)
    assert g.outputs[0] == "x"


def test_reduce_sum_scalar():
    """Tests the reduce sum scalar functionality."""
    g = Graph("mock")
    scalar_val = np.array(5.0, dtype=np.float32)
    g.tensors["x"] = Tensor(
        "x", shape=(), dtype=DType.FLOAT32, data=scalar_val, is_initializer=True
    )
    g.add_node(Node("ReduceSum", ["x"], ["y"], {}, "rs"))
    g.outputs = ["y"]
    IdentityEliminationPass().run(g)
    assert g.outputs[0] == "x"


def test_reduce_mean_scalar():
    """Tests the reduce mean scalar functionality."""
    g = Graph("mock")
    scalar_val = np.array(5.0, dtype=np.float32)
    g.tensors["x"] = Tensor(
        "x", shape=(), dtype=DType.FLOAT32, data=scalar_val, is_initializer=True
    )
    g.add_node(Node("ReduceMean", ["x"], ["y"], {}, "rm"))
    g.outputs = ["y"]
    IdentityEliminationPass().run(g)
    assert g.outputs[0] == "x"


def test_constant_folding_tile():
    """Tests the constant folding tile functionality."""
    from onnx9000.optimizer.simplifier.passes.constant_folding import ConstantFoldingPass

    g = Graph("mock_tile")
    data_val = np.array([1, 2], dtype=np.float32)
    reps_val = np.array([2], dtype=np.int64)
    g.tensors["x"] = Tensor(
        "x", shape=(2,), dtype=DType.FLOAT32, data=data_val, is_initializer=True
    )
    g.tensors["reps"] = Tensor(
        "reps", shape=(1,), dtype=DType.INT64, data=reps_val, is_initializer=True
    )
    g.initializers = ["x", "reps"]
    g.add_node(Node("Tile", ["x", "reps"], ["y"], {}, "tile"))
    g.outputs = ["y"]
    ConstantFoldingPass().run(g)
    assert g.nodes[0].op_type == "Constant"
    np.testing.assert_array_equal(
        g.nodes[0].attributes["value"], np.array([1, 2, 1, 2], dtype=np.float32)
    )


def test_constant_folding_split():
    """Tests the constant folding split functionality."""
    from onnx9000.optimizer.simplifier.passes.constant_folding import ConstantFoldingPass

    g = Graph("mock_split")
    data_val = np.array([1, 2, 3, 4], dtype=np.float32)
    split_val = np.array([2, 2], dtype=np.int64)
    g.tensors["x"] = Tensor(
        "x", shape=(4,), dtype=DType.FLOAT32, data=data_val, is_initializer=True
    )
    g.tensors["s"] = Tensor("s", shape=(2,), dtype=DType.INT64, data=split_val, is_initializer=True)
    g.initializers = ["x", "s"]
    g.add_node(Node("Split", ["x", "s"], ["y1", "y2"], {"axis": 0}, "split"))
    g.outputs = ["y1", "y2"]
    ConstantFoldingPass().run(g)
    assert len(g.nodes) == 2
    assert g.nodes[0].op_type == "Constant"
    assert g.nodes[1].op_type == "Constant"
    np.testing.assert_array_equal(
        g.nodes[0].attributes["value"], np.array([1, 2], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        g.nodes[1].attributes["value"], np.array([3, 4], dtype=np.float32)
    )
