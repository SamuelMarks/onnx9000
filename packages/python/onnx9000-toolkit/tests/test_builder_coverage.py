"""Tests the builder coverage module functionality."""

import numpy as np
import pytest
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Node
from onnx9000.toolkit.script.builder import GraphBuilder


def test_rename_all_initializers() -> None:
    """Tests the rename all initializers functionality."""
    gb = GraphBuilder("test_ren")
    gb.inputs.append({"name": "i1", "shape": [1], "dtype": DType.FLOAT32})
    gb.outputs.append({"name": "o1", "shape": [1], "dtype": DType.FLOAT32})
    gb.initializers["w1"] = np.array([1.0], dtype=np.float32)
    node = Node("Relu", ["i1"], ["o1"], {}, name="n1")
    gb.add_node(node)
    gb.rename_all("prefix")
    assert gb.inputs[0]["name"] == "prefix_i1"
    assert gb.outputs[0]["name"] == "prefix_o1"
    assert "prefix_w1" in gb.initializers
    assert gb.nodes[0].name == "prefix_n1"


def test_infer_shapes() -> None:
    """Tests the infer shapes functionality."""
    gb = GraphBuilder("test_shapes")
    gb.inputs.append({"name": "i1", "shape": [2, 2], "dtype": DType.FLOAT32})
    gb.initializers["w1"] = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    gb.initializers["i2"] = np.array([1, 2], dtype=np.int64)
    node1 = Node("Add", ["i1", "w1"], ["o1"], {}, name="n1")
    node2 = Node("Constant", [], ["o2"], {"value": np.array([5.0], dtype=np.float32)}, name="n2")
    sub_gb = GraphBuilder("sub")
    node3 = Node("Relu", ["o1"], ["o3"], {}, name="n3")
    sub_gb.add_node(node3)
    node4 = Node("If", ["i2"], ["o4"], {"then_branch": sub_gb}, name="n4")
    node5 = Node("Loop", ["max_trip", "cond"], ["o5"], {"body": sub_gb}, name="n5")
    gb.add_node(node1)
    gb.add_node(node2)
    gb.add_node(node4)
    gb.add_node(node5)
    gb.infer_shapes()
    assert len(gb.nodes) == 4


def test_extract_subgraph() -> None:
    """Tests the extract subgraph functionality."""
    gb = GraphBuilder("test_extract")
    gb.inputs.append({"name": "i1", "shape": [2], "dtype": DType.FLOAT32})
    gb.initializers["w1"] = np.array([1.0, 2.0], dtype=np.float32)
    node1 = Node("Add", ["i1", "w1"], ["o1"], {}, name="n1")
    gb.add_node(node1)
    gb.outputs.append({"name": "o1", "shape": [2], "dtype": DType.FLOAT32})
    extracted = gb.extract_subgraph(["i1"], ["o1"])
    assert len(extracted.nodes) > 0
    assert "w1" in extracted.initializers


def test_control_flow() -> None:
    """Tests the control flow functionality."""
    from onnx9000.toolkit.script.var import Var

    gb = GraphBuilder("test_cf")
    cond = Var("cond")
    gb.If(cond, 1)
    max_trip = Var("max_trip")
    gb.Loop(max_trip, cond, 1)


def test_to_onnx_extras() -> None:
    """Tests the to onnx extras functionality."""
    gb = GraphBuilder("test_to_onnx_extras")
    gb.inputs.append({"name": "i1", "shape": ["batch", 1], "dtype": DType.INT64})
    gb.inputs.append({"name": "i2", "shape": [1], "dtype": DType.UNDEFINED})
    gb.outputs.append({"name": "o1", "shape": [1], "dtype": DType.FLOAT32})
    gb.initializers["init_int"] = np.array([1, 2, 3], dtype=np.int64)
    sub_gb = GraphBuilder("sub_graph_attr")
    node_sub = Node("Relu", ["sub_i"], ["sub_o"], {}, name="sub_n")
    sub_gb.add_node(node_sub)
    node = Node(
        "DummyOp",
        ["i1"],
        ["o1"],
        {
            "ints_attr": [1, 2, 3],
            "floats_attr": [1.0, 2.0, 3.0],
            "tensor_int": np.array([1], dtype=np.int64),
            "graph_attr": sub_gb,
        },
        name="dummy1",
    )
    gb.add_node(node)
    gb.metadata["doc_string"] = "hello"
    gb.metadata["version"] = 42
    gb.metadata["custom_domain"] = "ai.onnx.custom"
    model = gb.to_onnx()
    assert model.model_version == 42
    assert any(imp.domain == "ai.onnx.custom" for imp in model.opset_import)


def test_validate_cyclic() -> None:
    """Tests the validate cyclic functionality."""
    gb = GraphBuilder("test_cyclic")
    node1 = Node("Op1", ["o2"], ["o1"], {}, name="n1")
    node2 = Node("Op2", ["o1"], ["o2"], {}, name="n2")
    node3 = Node("Op3", ["o3"], ["o2"], {}, name="n3")
    node4 = Node("Op4", ["o4"], ["o3"], {}, name="n4")
    gb.add_node(node4)
    gb.add_node(node3)
    gb.add_node(node1)
    gb.add_node(node2)
    with pytest.raises(ValueError, match="Cyclic dependency detected"):
        gb.validate()


def test_extract_subgraph_with_graph() -> None:
    """Tests the extract subgraph with graph functionality."""
    gb = GraphBuilder("test_extract_graph")
    gb.inputs.append({"name": "i1", "shape": [2], "dtype": DType.FLOAT32})
    gb.outputs.append({"name": "o1", "shape": [2], "dtype": DType.FLOAT32})
    sub_gb = GraphBuilder("sub")
    sub_gb.inputs.append({"name": "sub_i", "shape": [2], "dtype": DType.FLOAT32})
    sub_gb.outputs.append({"name": "sub_o", "shape": [2], "dtype": DType.FLOAT32})
    node1 = Node("If", ["i1"], ["o1"], {"then_branch": sub_gb}, name="n1")
    gb.add_node(node1)
    extracted = gb.extract_subgraph(["i1"], ["o1"])
    assert len(extracted.nodes) > 0
