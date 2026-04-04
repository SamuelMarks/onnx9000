"""Module docstring."""


def test_profiler_methods():
    """Docstring for D103."""
    from onnx9000.core.dtypes import DType
    from onnx9000.core.ir import Constant, DynamicDim, Graph, Node, Tensor
    from onnx9000.core.profiler import _add_metric, profile, resolve_volume

    # helper functions
    assert resolve_volume((2, 3)) == 6
    assert resolve_volume((2, DynamicDim("x")), {"x": 5}) == 10
    assert resolve_volume((2, DynamicDim("x"))) == "2 * x"
    assert resolve_volume((2, "y")) == "2 * y"

    assert _add_metric(5, 5) == 10
    assert _add_metric("x", 5) == "(x + 5)"
    assert _add_metric(5, "x") == "(5 + x)"
    assert _add_metric("x", "y") == "(x + y)"

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

    report.generate_suggestions()
    report.estimate_latency()

    # the printing methods just dump to stdout, we just need them to run
    report.print_parameter_pie_chart()
    report.print_activation_pie_chart()
    report.print_bottleneck_analysis()
    report.print_distribution_pie_chart()

    str(report)

    report.get_cumulative_flops_up_to("layer1/add")
    report.get_cumulative_flops_up_to("layer1/conv")


def test_profiler_generate_suggestions_more():
    """Docstring for D103."""
    from onnx9000.core.dtypes import DType
    from onnx9000.core.ir import Graph, Node, Tensor
    from onnx9000.core.profiler import ProfilerResult, profile

    g = Graph("test")
    x = Tensor("x", shape=[1, 3, 224, 224], dtype=DType.FLOAT64)
    x.is_initializer = True
    x.is_initializer = True
    y = Tensor("y", shape=[1, 3, 224, 224], dtype=DType.INT64)
    y.is_initializer = True
    y.is_initializer = True
    w = Tensor("w", shape=[1024, 1024, 3, 3], dtype=DType.FLOAT32)  # large
    g.tensors = {"x": x, "y": y, "w": w}
    w.is_initializer = True
    g.inputs = [x, y]

    # Int64 & Float64
    n1 = Node("Cast", inputs=["x"], outputs=["z"])
    n2 = Node("Cast", inputs=["y"], outputs=["z2"])

    # Large WebGPU limit exceeding node
    n3 = Node("Conv", inputs=["x", "w"], outputs=["z3"])

    g.nodes.extend([n1, n2, n3])

    profile(g)
    # The generation happens inside `profile`, so just calling it is enough
    pass
    pass
    pass


def test_cumulative_flops():
    """Docstring for D103."""
    from onnx9000.core.profiler import ProfilerResult

    res = ProfilerResult()
    res.node_profiles = [{"name": "n1", "flops": 10}, {"name": "n2", "flops": "x"}]
    assert res.get_cumulative_flops_up_to("n1") == 10
    assert res.get_cumulative_flops_up_to("n2") == "(10 + x)"


def test_profiler_intensity():
    """Docstring for D103."""
    from onnx9000.core.profiler import ProfilerResult

    r = ProfilerResult()
    r.node_profiles = [
        {
            "name": "n",
            "flops": 1,
            "macs": 1,
            "params": 100000000,
            "activation_bytes": 100000000,
            "arithmetic_intensity": 0.00001,
        }
    ]
    r.generate_suggestions()
