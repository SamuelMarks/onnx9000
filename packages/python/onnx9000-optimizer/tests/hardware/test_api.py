"""Test optimize API."""

import numpy as np
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.optimizer.hardware.api import (
    generate_js_wrapper,
    generate_optimization_report,
    generate_visual_dag_comparison,
    optimize,
    parse_olive_config,
    quantize_dynamic,
    quantize_static,
    run_in_pyodide,
)


def create_mock_api_graph():
    """Tests the create_mock_api_graph functionality."""
    g = Graph("api_graph")
    t1 = Tensor("in1", (1, 3, 224, 224), DType.FLOAT32)
    t2 = Tensor("w", (16, 3, 3, 3), DType.FLOAT32)
    g.add_tensor(t1)
    g.add_tensor(t2)
    n1 = Node("Conv", ["in1", "w"], ["out1"], {}, "conv1")
    g.add_node(n1)
    return g


def test_optimize_webgpu() -> None:
    """Tests the test_optimize_webgpu functionality."""
    g = create_mock_api_graph()
    g2 = optimize(g, target="webgpu")
    assert "device" in g2.nodes[0].attributes or g2.nodes[0].op_type == "Transpose"


def test_optimize_wasm() -> None:
    """Tests the test_optimize_wasm functionality."""
    g = create_mock_api_graph()
    g2 = optimize(g, target="wasm")
    assert "device" in g2.nodes[0].attributes or g2.nodes[0].op_type == "Transpose"


def test_quantize_dynamic() -> None:
    """Tests the test_quantize_dynamic functionality."""
    g = create_mock_api_graph()
    g2 = quantize_dynamic(g)
    assert len(g2.nodes) == len(g.nodes)


def test_quantize_static() -> None:
    """Tests the test_quantize_static functionality."""
    g = create_mock_api_graph()
    g2 = quantize_static(g, [np.ones((1,))])
    assert len(g2.nodes) == len(g.nodes)


def test_parse_olive_config() -> None:
    """Tests the test_parse_olive_config functionality."""
    config = parse_olive_config({"test": True})
    assert config["parsed"] is True


def test_generate_optimization_report() -> None:
    """Tests the test_generate_optimization_report functionality."""
    g = create_mock_api_graph()
    report = generate_optimization_report(g, g)
    assert "original_vram_bytes" in report


def test_run_in_pyodide() -> None:
    """Tests the test_run_in_pyodide functionality."""
    res = run_in_pyodide()
    assert isinstance(res, bool)


def test_generate_js_wrapper() -> None:
    """Tests the test_generate_js_wrapper functionality."""
    js = generate_js_wrapper()
    assert "ONNX9000Optimizer" in js


def test_generate_visual_dag_comparison() -> None:
    """Tests the test_generate_visual_dag_comparison functionality."""
    g = create_mock_api_graph()
    cmp = generate_visual_dag_comparison(g, g)
    assert isinstance(cmp, str)


def test_quantize_dynamic_real() -> None:
    """Tests the test_quantize_dynamic_real functionality."""
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real() -> None:
    """Tests the test_quantize_static_real functionality."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias() -> None:
    """Tests the test_quantize_static_real_with_bias functionality."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_non_conv() -> None:
    """Tests the test_quantize_static_non_conv functionality."""
    g = Graph("mock")
    n = Node("Relu", ["x"], ["y"], {}, "relu")
    g.add_node(n)
    g2 = quantize_static(g, [])
    assert g2.nodes[0].op_type == "Relu"


def test_quantize_dynamic_real() -> None:
    """Tests the test_quantize_dynamic_real functionality."""
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real() -> None:
    """Tests the test_quantize_static_real functionality."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias() -> None:
    """Tests the test_quantize_static_real_with_bias functionality."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real() -> None:
    """Tests the test_quantize_dynamic_real functionality."""
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real() -> None:
    """Tests the test_quantize_static_real functionality."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias() -> None:
    """Tests the test_quantize_static_real_with_bias functionality."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real() -> None:
    """Tests the test_quantize_dynamic_real functionality."""
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real() -> None:
    """Tests the test_quantize_static_real functionality."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias() -> None:
    """Tests the test_quantize_static_real_with_bias functionality."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real() -> None:
    """Tests the test_quantize_dynamic_real functionality."""
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real() -> None:
    """Tests the test_quantize_static_real functionality."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias() -> None:
    """Tests the test_quantize_static_real_with_bias functionality."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real() -> None:
    """Tests the test_quantize_dynamic_real functionality."""
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real() -> None:
    """Tests the test_quantize_static_real functionality."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias() -> None:
    """Tests the test_quantize_static_real_with_bias functionality."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real() -> None:
    """Tests the test_quantize_dynamic_real functionality."""
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real() -> None:
    """Tests the test_quantize_static_real functionality."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias() -> None:
    """Tests the test_quantize_static_real_with_bias functionality."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real() -> None:
    """Tests the test_quantize_dynamic_real functionality."""
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real() -> None:
    """Tests the test_quantize_static_real functionality."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias() -> None:
    """Tests the test_quantize_static_real_with_bias functionality."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real() -> None:
    """Tests the test_quantize_dynamic_real functionality."""
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real() -> None:
    """Tests the test_quantize_static_real functionality."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias() -> None:
    """Tests the test_quantize_static_real_with_bias functionality."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real() -> None:
    """Tests the test_quantize_dynamic_real functionality."""
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real() -> None:
    """Tests the test_quantize_static_real functionality."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias() -> None:
    """Tests the test_quantize_static_real_with_bias functionality."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real() -> None:
    """Tests the test_quantize_dynamic_real functionality."""
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real() -> None:
    """Tests the test_quantize_static_real functionality."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias() -> None:
    """Tests the test_quantize_static_real_with_bias functionality."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real() -> None:
    """Tests the test_quantize_dynamic_real functionality."""
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real() -> None:
    """Tests the test_quantize_static_real functionality."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias() -> None:
    """Tests the test_quantize_static_real_with_bias functionality."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real() -> None:
    """Tests the test_quantize_dynamic_real functionality."""
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real() -> None:
    """Tests the test_quantize_static_real functionality."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias() -> None:
    """Tests the test_quantize_static_real_with_bias functionality."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real() -> None:
    """Tests the test quantize dynamic real functionality."""
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real() -> None:
    """Tests the test quantize static real functionality."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias() -> None:
    """Tests the test quantize static real with bias functionality."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real() -> None:
    """Tests the test quantize dynamic real functionality."""
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real() -> None:
    """Tests the test quantize static real functionality."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias() -> None:
    """Tests the test quantize static real with bias functionality."""
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real() -> None:
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real() -> None:
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias() -> None:
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real() -> None:
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real() -> None:
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias() -> None:
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real() -> None:
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real() -> None:
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias() -> None:
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real() -> None:
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real() -> None:
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias() -> None:
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real() -> None:
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real() -> None:
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias() -> None:
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real() -> None:
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real() -> None:
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias() -> None:
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real() -> None:
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real() -> None:
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias() -> None:
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real() -> None:
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real() -> None:
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias() -> None:
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real() -> None:
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real() -> None:
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias() -> None:
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real() -> None:
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real() -> None:
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias() -> None:
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real() -> None:
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real() -> None:
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias() -> None:
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real() -> None:
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real() -> None:
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias() -> None:
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real() -> None:
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real() -> None:
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias() -> None:
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_dynamic_real() -> None:
    g = Graph("mock")
    n = Node("MatMul", ["a", "b"], ["c"], {}, "mm")
    g.add_node(n)
    g2 = quantize_dynamic(g)
    has_dql = any(node.op_type == "DynamicQuantizeLinear" for node in g2.nodes)
    assert has_dql


def test_quantize_static_real() -> None:
    g = Graph("mock")
    n = Node("Conv", ["x", "w"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv


def test_quantize_static_real_with_bias() -> None:
    g = Graph("mock")
    n = Node("Conv", ["x", "w", "b"], ["y"], {}, "conv")
    g.add_node(n)
    g2 = quantize_static(g, [])
    has_qconv = any(node.op_type == "QLinearConv" for node in g2.nodes)
    assert has_qconv
