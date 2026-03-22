import struct
import pytest
from onnx9000.core.ir import Graph, Node, Tensor, ValueInfo
from onnx9000.tflite_exporter.compiler.layout import LayoutOptimizer
from onnx9000.core.ir import Attribute


def test_layout_keep_nchw():
    graph = Graph("test")
    opt = LayoutOptimizer(graph, keep_nchw=True)
    opt.optimize()
    assert len(graph.nodes) == 0


def test_layout_inject_transposes():
    graph = Graph("test")
    graph.inputs.append(ValueInfo("X", (1, 3, 224, 224), "float32"))
    w_data = struct.pack(f"<{64 * 27}f", *([1.0] * (64 * 27)))
    graph.tensors["W"] = Tensor(
        "W", shape=(64, 3, 3, 3), dtype="float32", is_initializer=True, data=w_data
    )
    graph.nodes.append(Node("Conv", ["X", "W"], ["Y"], name="conv1"))

    opt = LayoutOptimizer(graph, keep_nchw=False)
    opt.optimize()

    assert len(graph.nodes) == 3
    assert graph.nodes[0].op_type == "Transpose"
    assert graph.nodes[0].attributes["perm"].value == [0, 2, 3, 1]
    assert graph.nodes[1].op_type == "Conv"
    assert graph.nodes[2].op_type == "Transpose"
    assert graph.nodes[2].attributes["perm"].value == [0, 3, 1, 2]


def test_layout_push_down():
    graph = Graph("test")
    graph.inputs.append(ValueInfo("X", (1, 3, 224, 224), "float32"))
    w_data = struct.pack(f"<{64 * 27}f", *([1.0] * (64 * 27)))
    graph.tensors["W"] = Tensor(
        "W", shape=(64, 3, 3, 3), dtype="float32", is_initializer=True, data=w_data
    )

    graph.nodes.append(Node("Conv", ["X", "W"], ["Y"], name="conv1"))
    graph.nodes.append(Node("Relu", ["Y"], ["Z"], name="relu1"))

    opt = LayoutOptimizer(graph, False)
    opt.optimize()

    assert len(graph.nodes) == 3
    assert graph.nodes[0].op_type == "Transpose"
    assert graph.nodes[1].op_type == "Conv"
    assert graph.nodes[2].op_type == "Transpose"


def test_layout_cancel():
    graph = Graph("test")
    graph.inputs.append(ValueInfo("X", (1, 3, 224, 224), "float32"))
    w1_data = struct.pack(f"<{64 * 27}f", *([1.0] * (64 * 27)))
    w2_data = struct.pack(f"<{64 * 64 * 9}f", *([1.0] * (64 * 64 * 9)))

    graph.tensors["W1"] = Tensor(
        "W1", shape=(64, 3, 3, 3), dtype="float32", is_initializer=True, data=w1_data
    )
    graph.tensors["W2"] = Tensor(
        "W2", shape=(64, 64, 3, 3), dtype="float32", is_initializer=True, data=w2_data
    )

    graph.nodes.append(Node("Conv", ["X", "W1"], ["Y"], name="conv1"))
    graph.nodes.append(Node("Conv", ["Y", "W2"], ["Z"], name="conv2"))

    opt = LayoutOptimizer(graph, False)
    opt.optimize()

    assert len(graph.nodes) == 4
    assert graph.nodes[0].op_type == "Transpose"
    assert graph.nodes[1].op_type == "Conv"
    assert graph.nodes[2].op_type == "Conv"
    assert graph.nodes[3].op_type == "Transpose"


def test_expand_1d_spatial_ops_pool():
    from onnx9000.core.ir import Attribute
    import struct

    graph = Graph("test")
    graph.inputs.append(ValueInfo("X", (1, 3, 224), "float32"))

    w_data = struct.pack(f"<{3 * 3 * 3}f", *([1.0] * 27))
    graph.tensors["W"] = Tensor(
        "W", shape=(3, 3, 3), dtype="float32", is_initializer=True, data=w_data
    )
    graph.nodes.append(
        Node(
            "MaxPool",
            ["X"],
            ["Y"],
            {
                "kernel_shape": Attribute("kernel_shape", "INTS", [3]),
                "strides": Attribute("strides", "INTS", [2]),
                "dilations": Attribute("dilations", "INTS", [1]),
            },
            name="pool1",
        )
    )

    opt = LayoutOptimizer(graph, keep_nchw=False)
    opt.expand_1d_spatial_ops()

    assert graph.nodes[0].op_type == "Unsqueeze"
    assert graph.nodes[1].op_type == "MaxPool"
    assert graph.nodes[1].attributes["kernel_shape"].value == [1, 3]
    assert graph.nodes[1].attributes["strides"].value == [1, 2]
    assert graph.nodes[1].attributes["dilations"].value == [1, 1]
    assert graph.nodes[2].op_type == "Squeeze"


def test_layout_fold_constants():
    import struct
    from onnx9000.core.ir import Attribute

    graph = Graph("test")

    w_dw_data = struct.pack(f"<{27}f", *([1.0] * 27))
    graph.tensors["W_dw"] = Tensor(
        "W_dw", shape=(3, 1, 3, 3), dtype="float32", is_initializer=True, data=w_dw_data
    )
    graph.nodes.append(
        Node(
            "Conv",
            ["X1", "W_dw"],
            ["Y1"],
            attributes={"group": Attribute("group", "INT", 3)},
            name="conv_dw",
        )
    )

    w_ct_data = struct.pack(f"<{3 * 64 * 9}f", *([1.0] * (3 * 64 * 9)))
    graph.tensors["W_ct"] = Tensor(
        "W_ct", shape=(3, 64, 3, 3), dtype="float32", is_initializer=True, data=w_ct_data
    )
    graph.nodes.append(Node("ConvTranspose", ["X2", "W_ct"], ["Y2"], name="conv_t"))

    w_gemm_data = struct.pack(f"<{200}f", *([1.0] * 200))
    graph.tensors["W_gemm"] = Tensor(
        "W_gemm", shape=(10, 20), dtype="float32", is_initializer=True, data=w_gemm_data
    )
    graph.nodes.append(Node("Gemm", ["X3", "W_gemm"], ["Y3"], name="gemm1"))

    opt = LayoutOptimizer(graph, False)
    opt.optimize()

    assert graph.tensors["W_dw"].shape == (1, 3, 3, 3)
    assert graph.tensors["W_ct"].shape == (64, 3, 3, 3)
    assert graph.tensors["W_gemm"].shape == (20, 10)
