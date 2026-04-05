"""Tests for profiler coverage."""


def test_profiler_modules():
    """Docstring for D103."""
    import pytest
    from onnx9000.core.dtypes import DType
    from onnx9000.core.ir import Constant, Graph, Node, Tensor
    from onnx9000.core.profiler import dtype_size, get_attr, profile
    from onnx9000.core.profiler_checks import OptimizationAnalyzer
    from onnx9000.core.profiler_grouping import (
        HierarchicalProfileNode,
        export_csv,
        export_hierarchical_json,
        group_by_namespace,
        to_pandas_dataframe,
    )

    g = Graph("test")
    x = Tensor("x", shape=[1, 3, 224, 224], dtype=DType.FLOAT32)
    w = Tensor("w", shape=[64, 3, 7, 7], dtype=DType.FLOAT32)
    w.is_initializer = True
    b = Tensor("b", shape=[64], dtype=DType.FLOAT32)
    y = Tensor("layer1/y", shape=[1, 64, 112, 112], dtype=DType.FLOAT32)

    n = Node("Conv", inputs=["x", "w", "b"], outputs=["layer1/y"])
    n.attributes["kernel_shape"] = type(
        "obj", (object,), {"name": "kernel_shape", "value": [7, 7]}
    )()
    n.attributes["group"] = type("obj", (object,), {"name": "group", "value": 1})()
    n.name = "layer1/conv"
    g.nodes.append(n)

    n2 = Node("Add", inputs=["layer1/y", "layer1/y"], outputs=["layer1/z"])
    n2.name = "layer1/add"
    g.nodes.append(n2)

    g.tensors = {"x": x, "w": w, "b": b, "layer1/y": y, "layer1/z": y}
    g.inputs.append(x)
    g.outputs.append(y)

    report = profile(g)

    analyzer = OptimizationAnalyzer(g)
    analyzer.analyze()

    hpn = group_by_namespace(report)
    hpn.print_tree()
    hpn.to_dict()

    import os

    export_hierarchical_json(report, "test.json")
    os.remove("test.json")

    to_pandas_dataframe(report)

    export_csv(report, "test.csv")
    os.remove("test.csv")


def test_optimization_analyzer():
    """Docstring for D103."""
    from onnx9000.core.dtypes import DType
    from onnx9000.core.ir import Constant, Graph, Node, Tensor
    from onnx9000.core.profiler_checks import OptimizationAnalyzer

    g = Graph("test")
    x = Tensor("x", shape=[1], dtype=DType.FLOAT32)
    w = Tensor("w", shape=[1], dtype=DType.FLOAT32)
    w.is_initializer = True
    unused = Tensor("unused", shape=[1], dtype=DType.FLOAT32)
    unused.is_initializer = True
    b = Tensor("b", shape=[1], dtype=DType.FLOAT32)
    y = Tensor("y", shape=[1], dtype=DType.FLOAT32)
    z = Tensor("z", shape=[1], dtype=DType.FLOAT32)

    n_cast = Node("Cast", inputs=["x"], outputs=["cx"])
    n_matmul = Node("MatMul", inputs=["cx", "w"], outputs=["y"])
    n_add = Node("Add", inputs=["y", "b"], outputs=["z"])
    n_conv = Node("Conv", inputs=["x"], outputs=["cy"])
    n_bn = Node("BatchNormalization", inputs=["cy"], outputs=["zbn"])
    n_id = Node("Identity", inputs=["x"], outputs=["x2"])
    n_loop = Node("Loop", inputs=["x"], outputs=["xl"])

    g.nodes.extend([n_cast, n_matmul, n_add, n_conv, n_bn, n_id, n_loop])
    g.tensors = {
        "x": x,
        "w": w,
        "b": b,
        "y": y,
        "z": z,
        "unused": unused,
        "cx": x,
        "cy": y,
        "zbn": z,
        "x2": x,
        "xl": x,
    }

    analyzer = OptimizationAnalyzer(g)
    opps = analyzer.analyze()
    assert len(opps) > 0


def test_profiler_grouping():
    """Docstring for D103."""
    import pytest
    from onnx9000.core.profiler_grouping import HierarchicalProfileNode, extract_namespace

    n = HierarchicalProfileNode("test")
    n.add_stats(1, 2, 3, 4)
    assert n.macs == 1
    assert n.flops == 2
    assert n.params == 3
    assert n.activation_bytes == 4

    assert extract_namespace("a/b/c") == ["a", "b", "c"]
    assert extract_namespace("a.b.c") == ["a", "b", "c"]
    assert extract_namespace("a") == ["a"]


def test_export_csv_empty():
    """Docstring for D103."""
    from onnx9000.core.profiler import ProfilerResult
    from onnx9000.core.profiler_grouping import export_csv

    r = ProfilerResult()
    export_csv(r, "test_empty.csv")
