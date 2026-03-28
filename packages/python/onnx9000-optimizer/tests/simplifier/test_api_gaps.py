"""Tests for packages/python/onnx9000-optimizer/tests/simplifier/test_api_gaps.py."""

import numpy as np
import pytest
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor, ValueInfo
from onnx9000.optimizer.simplifier.api import simplify


def testextract_scalars_array():
    """Perform testextract scalars array operation."""
    g = Graph("TestSimplifyInitArr")
    g.inputs = ["X"]
    g.outputs = ["Y"]
    g.tensors["X"] = Tensor("X", (2, 2), DType.FLOAT32)
    t_c = Tensor("C", (1,), DType.FLOAT32)
    t_c.data = np.array([2.0], dtype=np.float32)
    t_c.is_initializer = True
    g.tensors["C"] = t_c
    g.initializers.append("C")
    g.tensors["Y"] = Tensor("Y", (2, 2), DType.FLOAT32)
    n1 = Node("Add", ["X", "C"], ["Y"])
    g.nodes.append(n1)
    simplify(g, max_iterations=1, skip_shape_inference=True)
    assert g.tensors["C"].shape == ()
    assert isinstance(g.tensors["C"].data, np.ndarray)
    assert g.tensors["C"].data.ndim == 0


def test_api_flags():
    """Test api flags."""
    g = Graph("TestFlags")
    g.inputs = ["X"]
    g.outputs = ["Y"]
    g.tensors["X"] = Tensor("X", (2, 2), DType.FLOAT32)
    g.tensors["Y"] = Tensor("Y", (2, 2), DType.FLOAT32)
    simplify(
        g,
        skip_fuse_bn=True,
        skip_fusions=True,
        dry_run=True,
        size_limit_mb=0.0001,
        max_iterations=1,
        skip_shape_inference=True,
    )


def test_api_shape_inference_fail():
    """Test api shape inference fail."""
    g = Graph("TestShapeFail")
    g.inputs = ["X"]
    g.outputs = ["Y"]
    g.tensors["X"] = Tensor("X", (2, 2), DType.FLOAT32)
    n1 = Node("THIS_IS_NOT_A_REAL_OP", ["X"], ["Y"])
    g.nodes.append(n1)
    with pytest.MonkeyPatch.context() as m:
        m.setattr(
            "onnx9000.core.shape_inference.infer_shapes_and_types",
            lambda *a, **k: (_ for _ in ()).throw(ValueError("Oops")),
        )
        simplify(g, max_iterations=1, skip_shape_inference=False)


def test_api_opsets_metadata():
    """Test api opsets metadata."""
    g = Graph("TestOpsets")
    g.outputs = ["Y"]
    g.metadata_props["hello"] = "world"
    n = Node("Relu", ["X"], ["Y"])
    n.domain = "ai.onnx"
    g.nodes.append(n)
    g.opset_imports[""] = 14
    g.opset_imports["ai.onnx"] = 1
    g.opset_imports["ai.onnx.ml"] = 1
    simplify(g, target_opset=15, strip_metadata=True, max_iterations=1, skip_shape_inference=True)
    assert "" in g.opset_imports
    assert g.opset_imports[""] == 15
    assert g.opset_imports["ai.onnx"] == 1
    assert "ai.onnx.ml" not in g.opset_imports
    assert len(g.metadata_props) == 0


def test_api_kwargs_shapes_types():
    """Test api kwargs shapes types."""
    g = Graph("TestKwargs")
    v_in = ValueInfo("X", (), DType.FLOAT32)
    g.inputs.append(v_in)
    g.inputs.append("W")
    g.outputs.append("Y")
    g.tensors["X"] = Tensor("X", (), DType.FLOAT32)
    g.tensors["Y"] = Tensor("Y", (), DType.FLOAT32)
    g.tensors["Z"] = Tensor("Z", (), DType.FLOAT32)
    g.tensors["W"] = Tensor("W", (), DType.FLOAT32)
    g.nodes.append(Node("Identity", ["X"], ["Z"]))
    g.nodes.append(Node("Identity", ["Z"], ["Y"]))
    simplify(
        g,
        input_shapes={"X": [2, "batch"], "Z": [2, "batch"]},
        tensor_types={"X": "FLOAT16", "Z": "FLOAT16", "W": "INT64"},
        max_iterations=1,
        skip_shape_inference=True,
    )
    assert g.tensors["X"].shape[0] == 2
    assert getattr(g.tensors["X"].shape[1], "value", None) == "batch"
    assert g.tensors["X"].dtype == DType.FLOAT16


def testextract_scalars_direct():
    """Perform testextract scalars direct operation."""
    from onnx9000.optimizer.simplifier.api import extract_scalars

    g = Graph("TestInit")
    g.outputs = ["Y"]
    g.tensors["X"] = Tensor("X", (1,), DType.FLOAT32)
    g.tensors["X"].data = np.array([1.0], dtype=np.float32)
    g.initializers.append("X")
    n = Node("Add", ["X", "X"], ["Y"])
    g.nodes.append(n)
    extract_scalars(g)
    assert g.tensors["X"].shape == ()
    g2 = Graph("TestInit2")
    g2.outputs = ["Y"]
    g2.tensors["X"] = Tensor("X", (1,), DType.FLOAT32)
    g2.tensors["X"].data = [1.0]
    g2.initializers.append("X")
    n2 = Node("Add", ["X", "X"], ["Y"])
    g2.nodes.append(n2)
    extract_scalars(g2)
    assert g2.tensors["X"].shape == ()


def test_extract_scalars_output_overlap():
    """Test extract scalars output overlap."""
    from onnx9000.optimizer.simplifier.api import extract_scalars

    g = Graph("TestInitOut")
    g.outputs = ["X"]
    g.tensors["X"] = Tensor("X", (1,), DType.FLOAT32)
    g.tensors["X"].data = np.array([1.0], dtype=np.float32)
    g.initializers.append("X")
    extract_scalars(g)
    assert g.tensors["X"].shape == (1,)


def test_api_size_limit():
    """Test api size limit."""
    g = Graph("TestSizeLimit")
    t = Tensor("A", (100,), DType.FLOAT32)
    t.data = np.ones(100, dtype=np.float32)
    t.is_initializer = True
    g.tensors["A"] = t
    g.initializers.append("A")
    g.outputs = ["A"]
    simplify(g, size_limit_mb=0.0001)


def test_api_kwargs_all_the_things():
    """Test api kwargs all the things."""
    g = Graph("TestAll")
    g.outputs = ["Y", "X"]
    g.tensors["X"] = Tensor("X", (100,), DType.FLOAT32)
    g.tensors["X"].data = np.ones(100, dtype=np.float32)
    g.tensors["X"].is_initializer = True
    g.initializers.append("X")
    g.inputs = ["Y", "X"]
    g.value_info = [ValueInfo("Z", (), DType.FLOAT32), ValueInfo("A", (), DType.FLOAT32)]
    g.producer_name = "my_custom_producer"
    n = Node("Abs", ["X"], ["Y"])
    g.nodes.append(n)
    simplify(g, target_opset=17, sort_value_info=True, log_json_summary=True, size_limit_mb=0.0001)
    assert g.opset_imports[""] == 17
    assert g.producer_name == "my_custom_producer_onnx9000-simplifier"
    assert g.inputs == ["X", "Y"]


def test_api_producer_name_empty():
    """Test api producer name empty."""
    g = Graph("TestEmptyProd")
    g.outputs = ["X"]
    g.tensors["X"] = Tensor("X", (), DType.FLOAT32)
    g.tensors["X"].data = np.array(1.0)
    g.initializers.append("X")
    g.producer_name = ""
    simplify(g)
    assert g.producer_name == "onnx9000-simplifier"
