import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from onnx9000.core.ir import Graph, Node, Tensor, DynamicDim
from onnx9000.core.dtypes import DType
from onnx9000.backends.apple.executor import Dispatcher
import onnx9000.backends.apple.executor as exe
import onnx9000.backends.apple.bindings as bind


def test_apple_bindings():
    assert isinstance(bind.is_accelerate_available(), bool)
    assert isinstance(bind.is_metal_available(), bool)
    assert isinstance(bind.is_mps_available(), bool)
    bind.nsstring("hello")
    bind.mtl_create_system_default_device()


def create_math_graph():
    graph = Graph("math")
    graph.inputs = ["A", "B"]
    graph.outputs = ["C"]
    node = Node("Add", ["A", "B"], ["C"], {})
    graph.add_node(node)
    a_tensor = Tensor("A", (2, 2), DType.FLOAT32)
    b_tensor = Tensor("B", (2, 2), DType.FLOAT32)
    c_tensor = Tensor("C", (2, 2), DType.FLOAT32)
    graph.add_tensor(a_tensor)
    graph.add_tensor(b_tensor)
    graph.add_tensor(c_tensor)
    return graph


def create_matmul_graph():
    graph = Graph("matmul")
    graph.inputs = ["A", "B"]
    graph.outputs = ["C"]
    node = Node("MatMul", ["A", "B"], ["C"], {})
    graph.add_node(node)
    a_tensor = Tensor("A", (2, 2), DType.FLOAT32)
    b_tensor = Tensor("B", (2, 2), DType.FLOAT32)
    c_tensor = Tensor("C", (2, 2), DType.FLOAT32)
    graph.add_tensor(a_tensor)
    graph.add_tensor(b_tensor)
    graph.add_tensor(c_tensor)
    return graph


def test_apple_executor_fallback():
    graph = create_math_graph()
    with patch.object(exe, "is_accelerate_available", return_value=False):
        dispatcher = Dispatcher(graph)
        inputs = {
            "A": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            "B": np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
        }
        results = dispatcher.run(inputs)
        np.testing.assert_array_equal(
            results["C"], np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
        )
        graph.nodes[0].op_type = "Sub"
        results = dispatcher.run(inputs)
        np.testing.assert_array_equal(
            results["C"], np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
        )
        graph.nodes[0].op_type = "Mul"
        results = dispatcher.run(inputs)
        np.testing.assert_array_equal(
            results["C"], np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        )


def test_apple_executor_matmul_fallback():
    graph = create_matmul_graph()
    with patch.object(exe, "is_accelerate_available", return_value=False):
        dispatcher = Dispatcher(graph)
        inputs = {
            "A": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            "B": np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32),
        }
        results = dispatcher.run(inputs)
        np.testing.assert_array_equal(
            results["C"], np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)
        )


@patch.object(exe, "is_accelerate_available", return_value=True)
def test_apple_executor_accelerate_mocked(mock_accel):
    graph = create_math_graph()
    mock_lib = MagicMock()
    with patch.object(exe, "_accelerate_lib", mock_lib):
        dispatcher = Dispatcher(graph)
        inputs = {
            "A": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            "B": np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
        }
        dispatcher.run(inputs)
        mock_lib.vDSP_vadd.assert_called()
        graph.nodes[0].op_type = "Sub"
        dispatcher.run(inputs)
        mock_lib.vDSP_vsub.assert_called()
        graph.nodes[0].op_type = "Mul"
        dispatcher.run(inputs)
        mock_lib.vDSP_vmul.assert_called()
        graph.nodes[0].op_type = "MatMul"
        dispatcher.run(inputs)
        mock_lib.cblas_sgemm.assert_called()


def test_apple_executor_unsupported():
    graph = Graph("unsupported")
    graph.add_node(Node("UnknownOp", ["A"], ["B"], {}))
    graph.add_tensor(Tensor("A", (1,), DType.FLOAT32))
    graph.add_tensor(Tensor("B", (1,), DType.FLOAT32))
    dispatcher = Dispatcher(graph)
    inputs = {"A": np.array([1.0], dtype=np.float32)}
    with pytest.raises(RuntimeError):
        dispatcher.run(inputs)


def test_apple_executor_init_with_initializers():
    graph = Graph("init")
    graph.add_node(Node("Add", ["A", "B"], ["C"], {}))
    graph.outputs = ["C"]
    w_data = np.array([1.0], dtype=np.float32)
    graph.add_tensor(Tensor("A", (1,), DType.FLOAT32, is_initializer=True, data=w_data))
    graph.initializers = ["A"]
    graph.add_tensor(Tensor("B", (1,), DType.FLOAT32))
    graph.add_tensor(Tensor("C", (1,), DType.FLOAT32))
    dispatcher = Dispatcher(graph)
    inputs = {"B": np.array([2.0], dtype=np.float32)}
    with patch.object(exe, "is_accelerate_available", return_value=False):
        results = dispatcher.run(inputs)
        np.testing.assert_array_equal(results["C"], np.array([3.0], dtype=np.float32))


def test_apple_bindings_foundation_missing():
    with patch.object(bind, "_foundation_lib", None):
        res = bind.nsstring("hello")
        assert res.value is None


def test_apple_bindings_metal_attr_error():
    mock_lib = MagicMock()
    del mock_lib.MTLCreateSystemDefaultDevice
    with (
        patch.object(bind, "is_metal_available", return_value=True),
        patch.object(bind, "_metal_lib", mock_lib),
    ):
        res = bind.mtl_create_system_default_device()
        assert res is None


def test_apple_bindings_import_exceptions():
    import ctypes
    import importlib

    original_cdll = ctypes.CDLL

    def mock_cdll(name, *args, **kwargs):
        if name and isinstance(name, str) and "/System/Library/Frameworks" in name:
            raise Exception("Mocked exception for " + name)
        return original_cdll(name, *args, **kwargs)

    with patch("ctypes.CDLL", side_effect=mock_cdll):
        importlib.reload(bind)
        assert bind._accelerate_lib is None
        assert bind._metal_lib is None
        assert bind._mps_lib is None
        assert bind._foundation_lib is None
    importlib.reload(bind)


def test_apple_bindings_mtl_create_system_default_device_unavailable():
    with patch.object(bind, "is_metal_available", return_value=False):
        assert bind.mtl_create_system_default_device() is None
