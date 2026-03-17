from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.core.shape_inference import infer_shapes_and_types


def test_infer_shapes_and_types_add() -> None:
    g = Graph("test")
    g.add_tensor(Tensor("A", shape=(10, 20), dtype=DType.FLOAT32))
    g.add_tensor(Tensor("B", shape=(10, 20), dtype=DType.FLOAT32))
    g.inputs.append("A")
    g.inputs.append("B")
    n = Node("Add", inputs=["A", "B"], outputs=["Y"], attributes={})
    g.add_node(n)
    infer_shapes_and_types(g)
    assert "Y" in g.tensors
    assert g.tensors["Y"].shape == (10, 20)
    assert g.tensors["Y"].dtype == DType.FLOAT32


def test_infer_shapes_and_types_matmul() -> None:
    g = Graph("test")
    g.add_tensor(Tensor("A", shape=(10, 20), dtype=DType.FLOAT32))
    g.add_tensor(Tensor("B", shape=(20, 30), dtype=DType.FLOAT32))
    g.inputs.append("A")
    g.inputs.append("B")
    n = Node("MatMul", inputs=["A", "B"], outputs=["Y"], attributes={})
    g.add_node(n)
    infer_shapes_and_types(g)
    assert "Y" in g.tensors
    assert g.tensors["Y"].shape == (10, 30)


def test_infer_shapes_and_types_relu() -> None:
    g = Graph("test")
    g.add_tensor(Tensor("A", shape=(10, 20), dtype=DType.FLOAT32))
    g.inputs.append("A")
    n = Node("Relu", inputs=["A"], outputs=["Y"], attributes={})
    g.add_node(n)
    infer_shapes_and_types(g)
    assert "Y" in g.tensors
    assert g.tensors["Y"].shape == (10, 20)


from onnx9000.core.ir import DynamicDim, Attribute


def test_infer_shapes_symbolic_add():
    g = Graph("test")
    g.add_tensor(Tensor("A", shape=(10, DynamicDim("seq_len")), dtype=DType.FLOAT32))
    g.add_tensor(Tensor("B", shape=(10, 1), dtype=DType.FLOAT32))
    g.inputs.extend(["A", "B"])
    n = Node("Add", inputs=["A", "B"], outputs=["Y"], attributes={})
    g.add_node(n)
    infer_shapes_and_types(g)
    assert g.tensors["Y"].shape[0] == 10
    assert g.tensors["Y"].shape[1] == DynamicDim("seq_len")


def test_infer_shapes_conv():
    g = Graph("test_conv")
    g.add_tensor(Tensor("X", shape=(1, 3, 224, 224), dtype=DType.FLOAT32))
    g.add_tensor(Tensor("W", shape=(64, 3, 7, 7), dtype=DType.FLOAT32))
    g.inputs.extend(["X", "W"])
    n = Node(
        "Conv",
        inputs=["X", "W"],
        outputs=["Y"],
        attributes={
            "kernel_shape": Attribute("kernel_shape", value=[7, 7]),
            "strides": Attribute("strides", value=[2, 2]),
            "pads": Attribute("pads", value=[3, 3, 3, 3]),
        },
    )
    g.add_node(n)
    infer_shapes_and_types(g)
    assert g.tensors["Y"].shape == (1, 64, 112, 112)


def test_infer_shapes_reshape():
    g = Graph("test_reshape")
    g.add_tensor(Tensor("X", shape=(1, 10, 20), dtype=DType.FLOAT32))
    # use values array directly in tensor
    shape_t = Tensor("shape", shape=(2,), dtype=DType.INT64)
    shape_t.values = [1, -1]  # Custom attribute mock trick
    g.add_tensor(shape_t)
    g.inputs.extend(["X", "shape"])
    n = Node("Reshape", inputs=["X", "shape"], outputs=["Y"], attributes={})
    g.add_node(n)
    infer_shapes_and_types(g)
    assert g.tensors["Y"].shape == (1, 200)


def test_infer_shapes_matmul_symbolic():
    g = Graph("test_matmul")
    g.add_tensor(Tensor("A", shape=(DynamicDim("B"), 10, 20), dtype=DType.FLOAT32))
    g.add_tensor(Tensor("B", shape=(10, 20, 30), dtype=DType.FLOAT32))
    g.inputs.extend(["A", "B"])
    n = Node("MatMul", inputs=["A", "B"], outputs=["Y"], attributes={})
    g.add_node(n)
    infer_shapes_and_types(g)
    assert g.tensors["Y"].shape == (DynamicDim("max(B, 10)"), 10, 30)
