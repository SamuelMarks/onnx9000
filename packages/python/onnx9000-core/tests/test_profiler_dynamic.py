import pytest
from onnx9000.core.ir import Graph, Node, Tensor, Attribute
from onnx9000.core.profiler import profile, ProfilerResult
from onnx9000.core.symbolic import DynamicDim


def test_profiler_pad_resize():
    g = Graph("g")
    g.add_tensor(Tensor("x", ("N",), "float32"))
    g.add_tensor(Tensor("y", ("M",), "float32"))
    g.add_node(Node("Pad", ["x"], ["y"]))
    res = profile(g, dynamic_overrides={"N": 10, "M": 20})
    assert isinstance(res.total_flops, int)

    # Without overrides to trigger string path
    res_str = profile(g)
    print(res_str.node_profiles)
    assert "mem_bandwidth_bytes" in res_str.node_profiles[0]


def test_profiler_conv_dynamic():
    g = Graph("g")
    g.add_tensor(Tensor("x", ("N", "C", "H", "W"), "float32"))
    g.add_tensor(Tensor("w", ("O", "C", "kH", "kW"), "float32"))
    g.add_tensor(Tensor("y", ("N", "O", "H", "W"), "float32"))
    g.add_node(Node("Conv", ["x", "w"], ["y"], {"group": Attribute("group", value=2)}))

    # string ops
    res = profile(g)
    assert "(" in str(res.total_flops)

    # string ops with int dict mapping for C
    res_over = profile(g, dynamic_overrides={"C": 32})
    assert "(" in str(res_over.total_flops)


def test_profiler_matmul_dynamic():
    g = Graph("g")
    g.add_tensor(Tensor("x", ("N", "K"), "float32"))
    g.add_tensor(Tensor("w", ("K", "O"), "float32"))
    g.add_tensor(Tensor("y", ("N", "O"), "float32"))
    g.inputs.append("x")
    g.add_node(Node("MatMul", ["x", "w"], ["y"]))

    res = profile(g, dynamic_overrides={"K": 32})
    assert "(" in str(res.total_flops)
    res_no = profile(g)
    assert "(" in str(res_no.total_flops)


def test_profiler_if_dynamic():
    g = Graph("g")
    then_g = Graph("then")
    then_g.add_node(Node("Relu", ["x"], ["y"]))
    then_g.add_tensor(Tensor("x", ("N",), "float32"))
    then_g.add_tensor(Tensor("y", ("N",), "float32"))

    else_g = Graph("else")
    else_g.add_node(Node("Sigmoid", ["x"], ["y"]))
    else_g.add_tensor(Tensor("x", ("N",), "float32"))
    else_g.add_tensor(Tensor("y", ("N",), "float32"))

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
    g.add_tensor(Tensor("out", ("N",), "float32"))

    res = profile(g)
    assert "max" in str(res.total_flops)


def test_profiler_loop_dynamic():
    g = Graph("g")
    body_g = Graph("body")
    body_g.add_node(Node("Relu", ["x"], ["y"]))
    body_g.add_tensor(Tensor("x", ("N",), "float32"))
    body_g.add_tensor(Tensor("y", ("N",), "float32"))

    g.add_node(
        Node("Loop", ["M", "cond", "v_in"], ["v_out"], {"body": Attribute("body", value=body_g)})
    )
    # Without values for M so it falls back to loop_iters
    g.add_tensor(Tensor("M", [1], "int64"))
    g.add_tensor(Tensor("cond", [1], "bool"))
    g.add_tensor(Tensor("v_in", ("N",), "float32"))
    g.add_tensor(Tensor("v_out", ("N",), "float32"))

    res = profile(g)
    assert "loop_iters" in str(res.total_flops)
    res_ov = profile(g, {"loop_iters": 5})
    assert "5 *" in str(res_ov.total_flops)


def test_profiler_attention_dynamic():
    g = Graph("g")
    g.add_tensor(Tensor("x", ("B", "S", "E"), "float32"))
    g.add_node(Node("Attention", ["x"], ["y"]))
    res = profile(g)
    assert "2 *" in str(res.total_flops)


def test_profiler_rnn_dynamic():
    g = Graph("g")
    g.add_tensor(Tensor("x", ("S", "B", "I"), "float32"))
    g.add_tensor(Tensor("w", (1, "H", "I"), "float32"))
    g.add_tensor(Tensor("r", (1, "H", "H"), "float32"))
    g.add_node(Node("LSTM", ["x", "w", "r"], ["y"]))
    res = profile(g)
    assert "(" in str(res.total_flops)


def test_profiler_others_dynamic():
    g = Graph("g")
    g.add_tensor(Tensor("x", ("B", "I"), "float32"))
    g.inputs.append("x")
    g.add_node(Node("Add", ["x", "x"], ["y"]))
    res = profile(g)
    print(res.node_profiles)
    assert "B" in str(res.total_flops)


def test_profiler_batchnorm_dynamic():
    g = Graph("g")
    g.add_tensor(Tensor("x", ("B", "I"), "float32"))
    g.inputs.append("x")
    g.add_node(Node("BatchNormalization", ["x"], ["y"]))
    import onnx9000.core.profiler

    orig = onnx9000.core.profiler.infer_shapes_and_types
    onnx9000.core.profiler.infer_shapes_and_types = lambda x: None
    g.add_tensor(Tensor("y", ("B", "I"), "float32"))
    res = profile(g)
    onnx9000.core.profiler.infer_shapes_and_types = orig
    assert "B" in str(res.total_flops)


def test_profiler_reduce_dynamic():
    g = Graph("g")
    g.add_tensor(Tensor("x", ("B", "I"), "float32"))
    g.inputs.append("x")
    g.add_node(Node("ReduceMean", ["x"], ["y"]))
    res = profile(g)
    assert "B" in str(res.total_flops)


def test_profiler_pass_ops():
    g = Graph("g")
    g.add_tensor(Tensor("x", [10], "float32"))
    g.add_node(Node("Gather", ["x"], ["y"]))
    profile(g)


def test_profiler_conv_dynamicdim():
    g = Graph("g")
    g.add_tensor(Tensor("x", (1, DynamicDim("C"), 10, 10), "float32"))
    g.add_tensor(Tensor("w", (1, DynamicDim("C"), 3, 3), "float32"))
    g.inputs.append("x")
    g.add_node(Node("Conv", ["x", "w"], ["y"]))
    res = profile(g)
    assert "C" in str(res.total_flops)


def test_profiler_matmul_dynamicdim():
    g = Graph("g")
    g.add_tensor(Tensor("x", (1, DynamicDim("K")), "float32"))
    g.add_tensor(Tensor("w", (DynamicDim("K"), 1), "float32", is_initializer=True))
    g.initializers.append("w")
    g.inputs.append("x")
    g.add_node(Node("MatMul", ["x", "w"], ["y"]))
    res = profile(g)
    assert "K" in str(res.total_flops)


def test_profiler_gru():
    g = Graph("g")
    g.add_tensor(Tensor("x", (10, 2, 64), "float32"))  # seq, batch, input_size
    g.add_tensor(Tensor("w", (1, 128, 64), "float32"))
    g.add_tensor(Tensor("r", (1, 128, 128), "float32"))
    g.add_node(Node("GRU", ["x", "w", "r"], ["y"]))
    res = profile(g)
    assert res.total_flops > 0
