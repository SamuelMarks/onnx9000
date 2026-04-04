"""Module docstring."""


def test_exhaustive_profiler():
    """Docstring for D103."""
    from onnx9000.core.dtypes import DType
    from onnx9000.core.ir import Constant, DynamicDim, Graph, Node, Tensor
    from onnx9000.core.profiler import ProfilerResult, profile

    g = Graph("test")
    x = Tensor("x", shape=[1, 3, 224, 224], dtype=DType.FLOAT32)
    w = Tensor("w", shape=[64, 3, 7, 7], dtype=DType.FLOAT32)
    w.is_initializer = True
    b = Tensor("b", shape=[64], dtype=DType.FLOAT32)
    y = Tensor("y", shape=[1, 64, 112, 112], dtype=DType.FLOAT32)

    # Conv
    n1 = Node("Conv", inputs=["x", "w", "b"], outputs=["y"])

    # ConvTranspose
    y2 = Tensor("y2", shape=[1, 64, 224, 224], dtype=DType.FLOAT32)
    n2 = Node("ConvTranspose", inputs=["y", "w", "b"], outputs=["y2"])

    # Reshape (memory bandwidth test)
    y3 = Tensor("y3", shape=[1, 64 * 224 * 224], dtype=DType.FLOAT32)
    n3 = Node("Reshape", inputs=["y2"], outputs=["y3"])

    # MultiHeadAttention
    n4 = Node("MultiHeadAttention", inputs=["y3", "y3", "y3"], outputs=["y3b"])
    n4.attributes["num_heads"] = type("obj", (object,), {"name": "num_heads", "value": 8})()

    # GroupNorm
    n5 = Node("GroupNorm", inputs=["y3b"], outputs=["y3c"])
    n5.attributes["num_groups"] = type("obj", (object,), {"name": "num_groups", "value": 32})()

    # GlobalAveragePool
    y4 = Tensor("y4", shape=[1, 64, 1, 1], dtype=DType.FLOAT32)
    n6 = Node("GlobalAveragePool", inputs=["y2"], outputs=["y4"])

    # LSTM
    h = Tensor("h", shape=[1, 1, 64], dtype=DType.FLOAT32)
    n7 = Node("LSTM", inputs=["y3c"], outputs=["h"])

    # If node
    g_then = Graph("then")
    g_then.nodes.append(Node("Add", inputs=["h", "h"], outputs=["h_out"]))
    g_then.tensors = {"h": h, "h_out": h}

    g_else = Graph("else")
    g_else.nodes.append(Node("Mul", inputs=["h", "h"], outputs=["h_out"]))
    g_else.tensors = {"h": h, "h_out": h}

    n8 = Node("If", inputs=["b"], outputs=["h_out"])
    n8.attributes["then_branch"] = type(
        "obj", (object,), {"name": "then_branch", "value": g_then}
    )()
    n8.attributes["else_branch"] = type(
        "obj", (object,), {"name": "else_branch", "value": g_else}
    )()

    # Loop
    g_body = Graph("body")
    g_body.nodes.append(Node("Add", inputs=["h_out", "h"], outputs=["h_out2"]))
    g_body.tensors = {"h_out": h, "h": h, "h_out2": h}
    n9 = Node("Loop", inputs=["b", "b"], outputs=["h_out2"])
    n9.attributes["body"] = type("obj", (object,), {"name": "body", "value": g_body})()

    # Scan
    n10 = Node("Scan", inputs=["h_out2"], outputs=["h_out3"])
    n10.attributes["body"] = type("obj", (object,), {"name": "body", "value": g_body})()

    # Identity
    n11 = Node("Identity", inputs=["h_out3"], outputs=["h_out4"])

    g.nodes.extend([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11])
    g.tensors = {
        "x": x,
        "w": w,
        "b": b,
        "y": y,
        "y2": y2,
        "y3": y3,
        "y4": y4,
        "h": h,
        "h_out": h,
        "h_out2": h,
        "h_out3": h,
        "h_out4": h,
    }

    report = profile(g)
    assert report.total_macs is not None

    # Just exercise methods
    r2 = ProfilerResult()
    r2.total_flops = "2*x"
    r2.print_distribution_pie_chart()

    r3 = ProfilerResult()
    r3.total_params = "2*x"
    r3.print_parameter_pie_chart()

    r4 = ProfilerResult()
    r4.total_activation_bytes = "2*x"
    r4.print_activation_pie_chart()


def test_profiler_dynamic():
    """Docstring for D103."""
    from onnx9000.core.dtypes import DType
    from onnx9000.core.ir import Constant, DynamicDim, Graph, Node, Tensor
    from onnx9000.core.profiler import ProfilerResult, profile

    g = Graph("test")
    x = Tensor("x", shape=[DynamicDim("b"), DynamicDim("s"), DynamicDim("e")], dtype=DType.FLOAT32)
    w = Tensor("w", shape=[64, 3, DynamicDim("k"), DynamicDim("k")], dtype=DType.FLOAT32)
    w.is_initializer = True
    b = Tensor("b", shape=[64], dtype=DType.FLOAT32)
    y = Tensor("y", shape=[DynamicDim("b"), 64, 112, 112], dtype=DType.FLOAT32)

    n1 = Node("Conv", inputs=["x", "w", "b"], outputs=["y"])

    # ConvTranspose
    n2 = Node("ConvTranspose", inputs=["y", "w", "b"], outputs=["y2"])

    # Reshape (memory bandwidth test)
    n3 = Node("Reshape", inputs=["x"], outputs=["y3"])

    # MultiHeadAttention
    n4 = Node("MultiHeadAttention", inputs=["x", "x", "x"], outputs=["y4"])
    n4.attributes["num_heads"] = type("obj", (object,), {"name": "num_heads", "value": 8})()

    # RNN
    h = Tensor("h", shape=[DynamicDim("s"), DynamicDim("b"), DynamicDim("e")], dtype=DType.FLOAT32)
    n7 = Node("RNN", inputs=["x", "x", "x"], outputs=["h"])
    n7b = Node("GRU", inputs=["x", "x", "x"], outputs=["h"])

    g.nodes.extend([n1, n2, n3, n4, n7, n7b])
    g.tensors = {"x": x, "w": w, "b": b, "y": y, "h": h}

    report = profile(g, dynamic_overrides={"b": 1, "s": 10, "e": 32, "k": 3})
    profile(g)  # no overrides

    assert report.total_macs is not None
