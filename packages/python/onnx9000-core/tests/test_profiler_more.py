import pytest
from onnx9000.core.ir import Graph, Node, Tensor, Attribute
from onnx9000.core.profiler import profile, profile_graph, get_attr, ProfilerResult
from onnx9000.core.symbolic import DynamicDim


def test_profile_graph_decorator(capsys):
    @profile_graph
    def dummy_func(graph):
        return "done"

    g = Graph("g")
    assert dummy_func(g) == "done"
    out, err = capsys.readouterr()
    assert "Profiler Result" in out


def test_get_attr():
    n = Node("Test", [], [], {"val": Attribute("val", value=10)})
    assert get_attr(n, "val") == 10
    assert get_attr(n, "missing", default=5) == 5


def test_profiler_printing(capsys):
    res = ProfilerResult()
    res.total_params = 100
    res.peak_activation_bytes = 200
    res.total_flops = 500
    res.node_profiles = [
        {"name": "n1", "op_type": "Conv", "params": 60, "activation_bytes": 100, "flops": 300},
        {"name": "n2", "op_type": "Relu", "params": 40, "activation_bytes": 100, "flops": 200},
    ]
    res.print_parameter_pie_chart()
    res.print_activation_pie_chart()
    res.print_bottleneck_analysis()
    res.print_distribution_pie_chart()

    out, err = capsys.readouterr()
    assert "Conv: 60.0% (60 Params)" in out
    assert "Relu: 50.0% (0.00 MB)" in out
    assert "1. n1 (Conv) - FLOPs: 300" in out
    assert "Conv: 60.0% (300 FLOPs)" in out

    assert res.get_cumulative_flops_up_to("n1") == 300
    assert res.get_cumulative_flops_up_to("n2") == 500
    assert res.get_cumulative_flops_up_to("missing") == 500


def test_profiler_print_empty(capsys):
    res = ProfilerResult()
    res.print_parameter_pie_chart()
    res.print_activation_pie_chart()
    res.print_distribution_pie_chart()
    out, err = capsys.readouterr()
    assert "No parameters found." in out
    assert "No activations found." in out
    assert "Cannot compute pie chart" in out


def test_profiler_webgpu_limits():
    res = ProfilerResult()
    res.node_profiles = [
        {
            "name": "n1",
            "op_type": "Conv",
            "params": 150000000,
            "activation_bytes": 150000000,
            "flops": 0,
        },
    ]
    res._check_unsupported_webgpu = lambda: (
        None
    )  # mock to prevent crash if not there, wait this is in OptimizationChecker, not ProfilerResult
    # Let's just create a ProfilerResult and check suggestions
    res.suggestions = []
    # Actually limits are checked inside check_webgpu_limits, wait, no, they are in the __init__ or run directly?
    # The source code had:
    # if isinstance(params, int) and params > 134217728:
    #     self.suggestions.append(...)
    # Let's trigger profile() with large tensors.

    g = Graph("g")
    g.add_tensor(Tensor("large_w", [20000, 20000], "float32", is_initializer=True))
    g.initializers.append("large_w")
    g.add_tensor(Tensor("large_x", [20000, 20000], "float32"))
    g.inputs.append("large_x")
    g.add_node(Node("MatMul", ["large_x", "large_w"], ["out"]))
    g.add_tensor(Tensor("out", [20000, 20000], "float32"))
    res2 = profile(g)
    res2.generate_suggestions()
    assert any("exceeds WebGPU 128MB limit" in s for s in res2.suggestions)


def test_profiler_ops_if():
    g = Graph("g")
    then_g = Graph("then")
    then_g.add_node(Node("Relu", ["x"], ["y"]))
    then_g.add_tensor(Tensor("x", [10, 10], "float32"))
    then_g.add_tensor(Tensor("y", [10, 10], "float32"))

    else_g = Graph("else")
    else_g.add_node(Node("Sigmoid", ["x"], ["y"]))
    else_g.add_tensor(Tensor("x", [10, 10], "float32"))
    else_g.add_tensor(Tensor("y", [10, 10], "float32"))

    g.add_node(
        Node(
            "If",
            ["cond"],
            ["out"],
            {
                "then_branch": Attribute("then_branch", value=then_g),
                "else_branch": Attribute("else_branch", value=else_g),
            },
        )
    )
    g.add_tensor(Tensor("cond", [1], "bool"))
    g.add_tensor(Tensor("out", [10, 10], "float32"))

    res = profile(g)
    # The max flops between Relu (100) and Sigmoid (100)
    assert res.total_flops >= 100


def test_profiler_ops_loop():
    g = Graph("g")
    body_g = Graph("body")
    body_g.add_node(Node("Relu", ["x"], ["y"]))
    body_g.add_tensor(Tensor("x", [10, 10], "float32"))
    body_g.add_tensor(Tensor("y", [10, 10], "float32"))

    g.add_node(
        Node("Loop", ["M", "cond", "v_in"], ["v_out"], {"body": Attribute("body", value=body_g)})
    )
    # Mock M tensor with values
    t_M = Tensor(
        "M", [1], "int64", data=b"\x05\x00\x00\x00\x00\x00\x00\x00"
    )  # Not using data, using values?
    # Wait, the code says `hasattr(m_tensor, "values") and m_tensor.values is not None`
    # Let's mock it
    t_M.values = [5]
    g.add_tensor(t_M)

    g.add_tensor(Tensor("cond", [1], "bool"))
    g.add_tensor(Tensor("v_in", [10, 10], "float32"))
    g.add_tensor(Tensor("v_out", [10, 10], "float32"))

    res = profile(g)
    assert res.total_flops >= 500


def test_profiler_ops_attention():
    g = Graph("g")
    g.add_tensor(Tensor("x", [2, 10, 64], "float32"))
    g.add_node(Node("Attention", ["x"], ["y"]))
    res = profile(g)
    assert res.total_flops > 0


def test_profiler_ops_rnn():
    g = Graph("g")
    g.add_tensor(Tensor("x", [10, 2, 64], "float32"))  # seq, batch, input_size
    g.add_tensor(Tensor("w", [1, 128, 64], "float32"))
    g.add_tensor(Tensor("r", [1, 128, 128], "float32"))  # hidden is 128
    g.add_node(Node("LSTM", ["x", "w", "r"], ["y"]))
    res = profile(g)
    assert res.total_flops > 0


def test_profiler_estimate_latency():
    res = ProfilerResult()
    res.total_macs = 1e12  # 1 T MAC = 2 T FLOP
    res.total_memory_bytes = 10 * 1e9  # 10 GB
    lat = res.estimate_latency(hardware_tops=2.0, hardware_bw_gbps=100.0)
    assert "compute_latency_ms" in lat
    assert "memory_latency_ms" in lat
