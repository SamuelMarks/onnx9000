"""Tests the profiler grouping module functionality."""

import json
import os

from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Constant, Graph, Node, Tensor
from onnx9000.core.profiler import profile
from onnx9000.core.profiler_grouping import (
    export_csv,
    export_hierarchical_json,
    group_by_namespace,
    to_pandas_dataframe,
)


def test_profiler_grouping_hierarchy():
    """Tests the profiler grouping hierarchy functionality."""
    g = Graph("test")
    g.add_tensor(Tensor("A", shape=(10, 20), dtype=DType.FLOAT32))
    g.inputs.append("A")

    n1 = Node("Relu", inputs=["A"], outputs=["B"], name="model.layer.0.relu")
    g.add_node(n1)

    n2 = Node("Sigmoid", inputs=["B"], outputs=["C"], name="model.layer.0.sigmoid")
    g.add_node(n2)

    res = g.profile()

    tree = group_by_namespace(res)
    assert tree.name == "root"
    assert "model" in tree.children

    model_node = tree.children["model"]
    assert "layer" in model_node.children

    layer_node = model_node.children["layer"]
    assert "0" in layer_node.children

    zero_node = layer_node.children["0"]
    assert "relu" in zero_node.children
    assert "sigmoid" in zero_node.children


def test_profiler_export():
    """Tests the profiler export functionality."""
    g = Graph("test")
    g.add_tensor(Tensor("A", shape=(10, 20), dtype=DType.FLOAT32))
    g.inputs.append("A")
    n1 = Node("Relu", inputs=["A"], outputs=["B"], name="model.relu")
    g.add_node(n1)

    res = g.profile()

    export_hierarchical_json(res, "test_hierarchical.json")
    assert os.path.exists("test_hierarchical.json")

    with open("test_hierarchical.json") as f:
        data = json.load(f)
        assert data["name"] == "root"

    export_csv(res, "test_hierarchical.csv")
    assert os.path.exists("test_hierarchical.csv")

    os.remove("test_hierarchical.json")
    os.remove("test_hierarchical.csv")

    df = to_pandas_dataframe(res)
    assert len(df) == 1
    assert df[0]["Node"] == "model.relu"
    assert df[0]["Layer"] == "model"
