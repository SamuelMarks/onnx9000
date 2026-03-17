from onnx9000.core.ir import Graph, Node, Tensor, DynamicDim, Attribute
from onnx9000.core.dtypes import DType
from onnx9000.core.shape_inference import infer_shapes_and_types


def test_infer_shapes_conv_transpose():
    g = Graph("test")
    g.add_tensor(Tensor("X", shape=(1, 64, 112, 112), dtype=DType.FLOAT32))
    g.add_tensor(Tensor("W", shape=(64, 32, 3, 3), dtype=DType.FLOAT32))
    g.inputs.extend(["X", "W"])
    n = Node(
        "ConvTranspose",
        inputs=["X", "W"],
        outputs=["Y"],
        attributes={
            "kernel_shape": Attribute("kernel_shape", value=[3, 3]),
            "strides": Attribute("strides", value=[2, 2]),
            "pads": Attribute("pads", value=[1, 1, 1, 1]),
        },
    )
    g.add_node(n)
    infer_shapes_and_types(g)
    # Output shape: (112 - 1)*2 - 2*1 + (3 - 1)*1 + 1 = 111*2 - 2 + 2 + 1 = 222 - 2 + 3 = 223
    assert g.tensors["Y"].shape == (1, 32, 223, 223)


def test_infer_shapes_gather():
    g = Graph("test")
    g.add_tensor(Tensor("X", shape=(10, 20, 30), dtype=DType.FLOAT32))
    g.add_tensor(Tensor("indices", shape=(5, 5), dtype=DType.INT64))
    g.inputs.extend(["X", "indices"])
    n = Node(
        "Gather",
        inputs=["X", "indices"],
        outputs=["Y"],
        attributes={"axis": Attribute("axis", value=1)},
    )
    g.add_node(n)
    infer_shapes_and_types(g)
    assert g.tensors["Y"].shape == (10, 5, 5, 30)


def test_infer_shapes_slice():
    g = Graph("test")
    g.add_tensor(Tensor("X", shape=(10, 20, 30), dtype=DType.FLOAT32))
    starts_t = Tensor("starts", shape=(2,), dtype=DType.INT64)
    starts_t.values = [2, 0]
    ends_t = Tensor("ends", shape=(2,), dtype=DType.INT64)
    ends_t.values = [8, 15]
    axes_t = Tensor("axes", shape=(2,), dtype=DType.INT64)
    axes_t.values = [0, 2]

    g.add_tensor(starts_t)
    g.add_tensor(ends_t)
    g.add_tensor(axes_t)
    g.inputs.extend(["X", "starts", "ends", "axes"])
    n = Node("Slice", inputs=["X", "starts", "ends", "axes"], outputs=["Y"])
    g.add_node(n)
    infer_shapes_and_types(g)
    assert g.tensors["Y"].shape == (6, 20, 15)


def test_infer_shapes_concat():
    g = Graph("test")
    g.add_tensor(Tensor("A", shape=(10, 20), dtype=DType.FLOAT32))
    g.add_tensor(Tensor("B", shape=(10, 30), dtype=DType.FLOAT32))
    g.inputs.extend(["A", "B"])
    n = Node(
        "Concat", inputs=["A", "B"], outputs=["Y"], attributes={"axis": Attribute("axis", value=1)}
    )
    g.add_node(n)
    infer_shapes_and_types(g)
    assert g.tensors["Y"].shape == (10, 50)


def test_infer_shapes_split():
    g = Graph("test")
    g.add_tensor(Tensor("X", shape=(10, 50), dtype=DType.FLOAT32))
    g.inputs.extend(["X"])
    n = Node(
        "Split",
        inputs=["X"],
        outputs=["Y1", "Y2"],
        attributes={
            "axis": Attribute("axis", value=1),
            "split": Attribute("split", value=[20, 30]),
        },
    )
    g.add_node(n)
    infer_shapes_and_types(g)
    assert g.tensors["Y1"].shape == (10, 20)
    assert g.tensors["Y2"].shape == (10, 30)


def test_infer_shapes_tile():
    g = Graph("test")
    g.add_tensor(Tensor("X", shape=(10, 20), dtype=DType.FLOAT32))
    repeats_t = Tensor("repeats", shape=(2,), dtype=DType.INT64)
    repeats_t.values = [2, 3]
    g.add_tensor(repeats_t)
    g.inputs.extend(["X", "repeats"])
    n = Node("Tile", inputs=["X", "repeats"], outputs=["Y"])
    g.add_node(n)
    infer_shapes_and_types(g)
    assert g.tensors["Y"].shape == (20, 60)


def test_infer_shapes_pad():
    g = Graph("test")
    g.add_tensor(Tensor("X", shape=(10, 20), dtype=DType.FLOAT32))
    pads_t = Tensor("pads", shape=(4,), dtype=DType.INT64)
    pads_t.values = [1, 2, 3, 4]
    g.add_tensor(pads_t)
    g.inputs.extend(["X", "pads"])
    n = Node("Pad", inputs=["X", "pads"], outputs=["Y"])
    g.add_node(n)
    infer_shapes_and_types(g)
    assert g.tensors["Y"].shape == (14, 26)


def test_infer_shapes_topk():
    g = Graph("test")
    g.add_tensor(Tensor("X", shape=(10, 20), dtype=DType.FLOAT32))
    k_t = Tensor("K", shape=(1,), dtype=DType.INT64)
    k_t.values = [5]
    g.add_tensor(k_t)
    g.inputs.extend(["X", "K"])
    n = Node(
        "TopK",
        inputs=["X", "K"],
        outputs=["Values", "Indices"],
        attributes={"axis": Attribute("axis", value=-1)},
    )
    g.add_node(n)
    infer_shapes_and_types(g)
    assert g.tensors["Values"].shape == (10, 5)
    assert g.tensors["Indices"].shape == (10, 5)
