"""Test optimize API."""

import pytest
import numpy as np
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.core.dtypes import DType
from onnx9000.optimize.hardware.api import (
    optimize,
    quantize_dynamic,
    quantize_static,
    parse_olive_config,
    generate_optimization_report,
    run_in_pyodide,
    generate_js_wrapper,
    generate_visual_dag_comparison,
)


def create_mock_api_graph():
    """Provides create mock api graph functionality and verification."""
    g = Graph("api_graph")
    t1 = Tensor("in1", (1, 3, 224, 224), DType.FLOAT32)
    t2 = Tensor("w", (16, 3, 3, 3), DType.FLOAT32)
    g.add_tensor(t1)
    g.add_tensor(t2)
    n1 = Node("Conv", ["in1", "w"], ["out1"], {}, "conv1")
    g.add_node(n1)
    return g


def test_optimize_webgpu():
    """Provides semantic functionality and verification."""
    g = create_mock_api_graph()
    g2 = optimize(g, target="webgpu")
    assert "device" in g2.nodes[0].attributes or g2.nodes[0].op_type == "Transpose"


def test_optimize_wasm():
    """Provides semantic functionality and verification."""
    g = create_mock_api_graph()
    g2 = optimize(g, target="wasm")
    assert "device" in g2.nodes[0].attributes or g2.nodes[0].op_type == "Transpose"


def test_quantize_dynamic():
    """Provides semantic functionality and verification."""
    g = create_mock_api_graph()
    g2 = quantize_dynamic(g)
    assert len(g2.nodes) == len(g.nodes)


def test_quantize_static():
    """Provides semantic functionality and verification."""
    g = create_mock_api_graph()
    g2 = quantize_static(g, [np.ones((1,))])
    assert len(g2.nodes) == len(g.nodes)


def test_parse_olive_config():
    """Provides semantic functionality and verification."""
    config = parse_olive_config({"test": True})
    assert config["parsed"] is True


def test_generate_optimization_report():
    """Provides semantic functionality and verification."""
    g = create_mock_api_graph()
    report = generate_optimization_report(g, g)
    assert "original_vram_bytes" in report


def test_run_in_pyodide():
    """Provides semantic functionality and verification."""
    res = run_in_pyodide()
    assert isinstance(res, bool)


def test_generate_js_wrapper():
    """Provides semantic functionality and verification."""
    js = generate_js_wrapper()
    assert "ONNX9000Optimizer" in js


def test_generate_visual_dag_comparison():
    """Provides semantic functionality and verification."""
    g = create_mock_api_graph()
    cmp = generate_visual_dag_comparison(g, g)
    assert isinstance(cmp, str)


def test_quantize_dynamic_real():
    """Provides semantic functionality and verification."""
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real():
    """Provides semantic functionality and verification."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias():
    """Provides semantic functionality and verification."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_non_conv():
    """Provides semantic functionality and verification."""
    g = Graph("mock")
    n = Node("Relu", ["x"], ["y"], {}, "relu")
    g.add_node(n)
    g2 = quantize_static(g, [])
    assert g2.nodes[0].op_type == "Relu"


def test_quantize_dynamic_real():
    """Provides semantic functionality and verification."""
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real():
    """Provides semantic functionality and verification."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias():
    """Provides semantic functionality and verification."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real():
    """Provides semantic functionality and verification."""
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real():
    """Provides semantic functionality and verification."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias():
    """Provides semantic functionality and verification."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real():
    """Provides semantic functionality and verification."""
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real():
    """Provides semantic functionality and verification."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias():
    """Provides semantic functionality and verification."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real():
    """Provides semantic functionality and verification."""
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real():
    """Provides semantic functionality and verification."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias():
    """Provides semantic functionality and verification."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real():
    """Provides semantic functionality and verification."""
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real():
    """Provides semantic functionality and verification."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias():
    """Provides semantic functionality and verification."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real():
    """Provides semantic functionality and verification."""
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real():
    """Provides semantic functionality and verification."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias():
    """Provides semantic functionality and verification."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real():
    """Provides semantic functionality and verification."""
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real():
    """Provides semantic functionality and verification."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias():
    """Provides semantic functionality and verification."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real():
    """Provides semantic functionality and verification."""
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real():
    """Provides semantic functionality and verification."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias():
    """Provides semantic functionality and verification."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real():
    """Provides semantic functionality and verification."""
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real():
    """Provides semantic functionality and verification."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias():
    """Provides semantic functionality and verification."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real():
    """Provides semantic functionality and verification."""
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real():
    """Provides semantic functionality and verification."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias():
    """Provides semantic functionality and verification."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real():
    """Provides semantic functionality and verification."""
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real():
    """Provides semantic functionality and verification."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias():
    """Provides semantic functionality and verification."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real():
    """Provides semantic functionality and verification."""
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real():
    """Provides semantic functionality and verification."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias():
    """Provides semantic functionality and verification."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real():
    """Tests the test quantize dynamic real functionality."""
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real():
    """Tests the test quantize static real functionality."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias():
    """Tests the test quantize static real with bias functionality."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real():
    """Tests the test quantize dynamic real functionality."""
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real():
    """Tests the test quantize static real functionality."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias():
    """Tests the test quantize static real with bias functionality."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real():
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real():
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias():
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real():
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real():
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias():
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real():
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real():
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias():
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real():
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real():
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias():
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real():
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real():
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias():
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real():
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real():
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias():
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real():
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real():
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias():
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real():
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real():
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias():
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real():
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real():
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias():
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real():
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real():
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias():
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real():
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real():
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias():
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real():
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real():
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias():
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real():
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real():
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias():
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real():
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real():
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias():
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv
