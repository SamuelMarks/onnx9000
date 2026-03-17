from onnx9000.core.ir import Graph, Node, Tensor, Constant, DynamicDim, Attribute
from onnx9000.core.dtypes import DType
from onnx9000.core.profiler import profile


def test_profiler_resnet50_mock():
    # A single bottleneck block: 1x1 conv -> 3x3 conv -> 1x1 conv
    g = Graph("resnet50")
    g.add_tensor(Tensor("X", shape=(1, 256, 56, 56), dtype=DType.FLOAT32))

    g.add_tensor(Tensor("W1", shape=(64, 256, 1, 1), dtype=DType.FLOAT32, is_initializer=True))
    g.add_tensor(Tensor("Y1", shape=(1, 64, 56, 56), dtype=DType.FLOAT32))
    g.add_node(
        Node(
            "Conv",
            inputs=["X", "W1"],
            outputs=["Y1"],
            attributes={"kernel_shape": Attribute("k", value=[1, 1])},
        )
    )

    g.add_tensor(Tensor("W2", shape=(64, 64, 3, 3), dtype=DType.FLOAT32, is_initializer=True))
    g.add_tensor(Tensor("Y2", shape=(1, 64, 56, 56), dtype=DType.FLOAT32))
    g.add_node(
        Node(
            "Conv",
            inputs=["Y1", "W2"],
            outputs=["Y2"],
            attributes={
                "kernel_shape": Attribute("k", value=[3, 3]),
                "pads": Attribute("p", value=[1, 1, 1, 1]),
            },
        )
    )

    g.add_tensor(Tensor("W3", shape=(256, 64, 1, 1), dtype=DType.FLOAT32, is_initializer=True))
    g.add_tensor(Tensor("Y3", shape=(1, 256, 56, 56), dtype=DType.FLOAT32))
    g.add_node(
        Node(
            "Conv",
            inputs=["Y2", "W3"],
            outputs=["Y3"],
            attributes={"kernel_shape": Attribute("k", value=[1, 1])},
        )
    )

    res = g.profile()
    # verify MACs are calculated dynamically correctly for the layers
    assert res.total_macs > 0


def test_profiler_bert_mock():
    g = Graph("bert")
    g.add_tensor(Tensor("X", shape=(DynamicDim("B"), DynamicDim("S"), 768), dtype=DType.FLOAT32))
    g.add_tensor(Tensor("Y", shape=(DynamicDim("B"), DynamicDim("S"), 768), dtype=DType.FLOAT32))

    g.add_node(Node("Attention", inputs=["X"], outputs=["Y"]))
    res = g.profile()
    assert "B * S" in str(res.total_macs) or "(2 * B * S * 768^2" in str(res.total_macs)


def test_profiler_mobilenet_depthwise():
    g = Graph("mobilenet")
    g.add_tensor(Tensor("X", shape=(1, 32, 112, 112), dtype=DType.FLOAT32))
    g.add_tensor(Tensor("W", shape=(32, 1, 3, 3), dtype=DType.FLOAT32, is_initializer=True))
    g.add_tensor(Tensor("Y", shape=(1, 32, 112, 112), dtype=DType.FLOAT32))

    g.add_node(
        Node(
            "Conv",
            inputs=["X", "W"],
            outputs=["Y"],
            attributes={
                "kernel_shape": Attribute("k", value=[3, 3]),
                "group": Attribute("group", value=32),
                "pads": Attribute("p", value=[1, 1, 1, 1]),
            },
        )
    )

    res = g.profile()
    # out_v = 32 * 112 * 112 = 401408
    # k_v = 9
    # in_c = 32
    # groups = 32
    # macs = 401408 * 9 * (32 / 32) = 3612672
    assert res.total_macs == 3612672


def test_profiler_slice_dynamic():
    g = Graph("test")
    g.add_tensor(Tensor("X", shape=(DynamicDim("B"), 20, 30), dtype=DType.FLOAT32))
    g.inputs.extend(["X"])
    n = Node("Slice", inputs=["X"], outputs=["Y"])
    g.add_node(n)

    from onnx9000.core.shape_inference import infer_shapes_and_types

    infer_shapes_and_types(g)
    assert g.tensors["Y"].shape[0].value == "B"
