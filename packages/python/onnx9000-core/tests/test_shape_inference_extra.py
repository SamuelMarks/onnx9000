import numpy as np
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor, ValueInfo
from onnx9000.core.shape_inference import infer_shapes_and_types


def test_shape_inference_reshape_tensor_data():
    g = Graph("g")
    g.inputs.append(ValueInfo("x", DType.FLOAT32, (10, 20)))

    shape_tensor = Tensor("shape", DType.INT64, (2,))
    shape_tensor.data = np.array([20, 10], dtype=np.int64).tobytes()
    g.tensors["shape"] = shape_tensor
    g.add_tensor(shape_tensor)

    n = Node("Reshape", ["x", "shape"], ["y"])
    g.add_node(n)

    infer_shapes_and_types(g)
