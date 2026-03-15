import numpy as np
import pytest
from onnx9000.core.ir import Graph, Node, Tensor, DynamicDim
from onnx9000.core.dtypes import DType
from onnx9000.backends.cpu.executor import Executor


def create_simple_graph():
    graph = Graph("test_graph")
    graph.inputs = ["X"]
    graph.outputs = ["Y"]
    node = Node("Relu", ["X"], ["Y"], {})
    graph.add_node(node)
    x_tensor = Tensor("X", (2, 2), DType.FLOAT32)
    y_tensor = Tensor("Y", (2, 2), DType.FLOAT32)
    graph.add_tensor(x_tensor)
    graph.add_tensor(y_tensor)
    return graph


def create_complex_graph():
    graph = Graph("complex")
    graph.inputs = ["X"]
    graph.outputs = ["Z"]
    graph.initializers = ["W"]
    node1 = Node("MatMul", ["X", "W"], ["Y"], {})
    node2 = Node("Relu", ["Y"], ["Z"], {})
    graph.add_node(node1)
    graph.add_node(node2)
    w_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    w_tensor = Tensor("W", (2, 2), DType.FLOAT32, is_initializer=True, data=w_data)
    y_tensor = Tensor("Y", (2, 2), DType.FLOAT32)
    z_tensor = Tensor("Z", (2, 2), DType.FLOAT32)
    x_tensor = Tensor("X", (2, 2), DType.FLOAT32)
    graph.add_tensor(w_tensor)
    graph.add_tensor(x_tensor)
    graph.add_tensor(y_tensor)
    graph.add_tensor(z_tensor)
    return graph


def test_executor_simple():
    graph = create_simple_graph()
    executor = Executor(graph)
    inputs = {"X": np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float32)}
    results = executor.run(inputs)
    assert "Y" in results
    np.testing.assert_array_equal(
        results["Y"], np.array([[0.0, 2.0], [0.0, 4.0]], dtype=np.float32)
    )


def test_executor_complex():
    graph = create_complex_graph()
    executor = Executor(graph, use_threadpool=True)
    inputs = {"X": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)}
    results = executor.run(inputs)
    assert "Z" in results
    expected_y = np.matmul(
        inputs["X"], np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    )
    expected_z = np.maximum(0, expected_y)
    np.testing.assert_array_equal(results["Z"], expected_z)


def test_executor_unsupported_op():
    graph = Graph("test")
    node = Node("UnknownOp", ["A"], ["B"], {})
    graph.add_node(node)
    executor = Executor(graph)
    with pytest.raises(RuntimeError):
        executor.run({"A": np.array([1])})


def test_executor_dynamic_dim():
    graph = Graph("test")
    graph.inputs = ["X"]
    graph.outputs = ["Y"]
    node = Node("Relu", ["X"], ["Y"], {})
    graph.add_node(node)
    y_tensor = Tensor("Y", (DynamicDim("N"), 2), DType.FLOAT32)
    graph.add_tensor(y_tensor)
    executor = Executor(graph)
    inputs = {"X": np.array([[-1.0, 2.0], [-3.0, 4.0], [1.0, -1.0]], dtype=np.float32)}
    results = executor.run(inputs)
    assert "Y" in results
    np.testing.assert_array_equal(
        results["Y"], np.array([[0.0, 2.0], [0.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    )


def test_executor_missing_tensor_shape():
    graph = Graph("test")
    graph.inputs = ["X"]
    graph.outputs = ["Y"]
    node = Node("Relu", ["X"], ["Y"], {})
    graph.add_node(node)
    executor = Executor(graph)
    inputs = {"X": np.array([[-1.0, 2.0]], dtype=np.float32)}
    results = executor.run(inputs)
    np.testing.assert_array_equal(
        results["Y"], np.array([[0.0, 2.0]], dtype=np.float32)
    )


def test_executor_constant_output():
    graph = Graph("test")
    graph.outputs = ["W"]
    w_data = np.array([1.0], dtype=np.float32)
    w_tensor = Tensor("W", (1,), DType.FLOAT32, is_initializer=True, data=w_data)
    graph.initializers = ["W"]
    graph.add_tensor(w_tensor)
    executor = Executor(graph)
    results = executor.run({})
    np.testing.assert_array_equal(results["W"], w_data)
