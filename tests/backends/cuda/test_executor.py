import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from onnx9000.core.ir import Graph, Node, Tensor, DynamicDim
from onnx9000.core.dtypes import DType
from onnx9000.backends.cuda.executor import Dispatcher
import onnx9000.backends.cuda.executor as exe


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


def test_dispatcher_init_no_cuda():
    graph = create_simple_graph()
    dispatcher = Dispatcher(graph)
    assert not dispatcher.initialized
    inputs = {"X": np.array([[1.0, -1.0], [2.0, -2.0]], dtype=np.float32)}
    results = dispatcher.run(inputs)
    assert "Y" in results


@patch.object(exe, "is_cuda_available", return_value=True)
@patch.object(exe, "is_cublas_available", return_value=True)
@patch.object(exe, "is_cudnn_available", return_value=True)
@patch.object(
    exe,
    "_cuda_lib",
    MagicMock(
        cuInit=MagicMock(return_value=0),
        cuStreamCreate=MagicMock(return_value=0),
        cuStreamDestroy_v2=MagicMock(return_value=0),
    ),
)
@patch.object(
    exe,
    "_cublas_lib",
    MagicMock(
        cublasCreate_v2=MagicMock(return_value=0),
        cublasSetStream_v2=MagicMock(return_value=0),
        cublasDestroy_v2=MagicMock(return_value=0),
        cublasSgemm_v2=MagicMock(return_value=0),
    ),
)
@patch.object(
    exe,
    "_cudnn_lib",
    MagicMock(
        cudnnCreate=MagicMock(return_value=0), cudnnDestroy=MagicMock(return_value=0)
    ),
)
def test_dispatcher_init_with_cuda(*args):
    graph = create_simple_graph()
    with (
        patch.object(exe.CUDAMemoryPlanner, "build_arena") as mock_build,
        patch.object(exe.CUDAMemoryPlanner, "set_tensor") as mock_set,
        patch.object(
            exe.CUDAMemoryPlanner, "get_host_tensor", return_value=np.zeros((2, 2))
        ),
    ):
        dispatcher = Dispatcher(graph)
        assert dispatcher.initialized
        inputs = {"X": np.array([[1.0, -1.0], [2.0, -2.0]], dtype=np.float32)}
        results = dispatcher.run(inputs)
        assert "Y" in results


@patch.object(exe, "is_cuda_available", return_value=True)
@patch.object(exe, "is_cublas_available", return_value=True)
@patch.object(exe, "is_cudnn_available", return_value=True)
@patch.object(
    exe,
    "_cuda_lib",
    MagicMock(
        cuInit=MagicMock(return_value=0), cuStreamCreate=MagicMock(return_value=0)
    ),
)
@patch.object(
    exe,
    "_cublas_lib",
    MagicMock(
        cublasCreate_v2=MagicMock(return_value=0),
        cublasSetStream_v2=MagicMock(return_value=0),
        cublasSgemm_v2=MagicMock(return_value=0),
    ),
)
@patch.object(exe, "_cudnn_lib", MagicMock(cudnnCreate=MagicMock(return_value=0)))
def test_dispatcher_matmul(*args):
    graph = create_matmul_graph()
    with (
        patch.object(exe.CUDAMemoryPlanner, "build_arena"),
        patch.object(exe.CUDAMemoryPlanner, "set_tensor"),
        patch.object(
            exe.CUDAMemoryPlanner, "get_tensor_ptr", return_value=exe.CUdeviceptr(0)
        ),
        patch.object(
            exe.CUDAMemoryPlanner, "get_host_tensor", return_value=np.zeros((2, 2))
        ),
    ):
        dispatcher = Dispatcher(graph)
        assert dispatcher.initialized
        inputs = {
            "A": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            "B": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        }
        dispatcher.planner.tensors_shape_dtype["A"] = (2, 2), np.dtype("float32")
        dispatcher.planner.tensors_shape_dtype["B"] = (2, 2), np.dtype("float32")
        results = dispatcher.run(inputs)
        assert "C" in results
        exe._cublas_lib.cublasSgemm_v2.assert_called()


@patch.object(exe, "is_cuda_available", return_value=True)
@patch.object(exe, "is_cublas_available", return_value=False)
@patch.object(exe, "is_cudnn_available", return_value=False)
@patch.object(
    exe,
    "_cuda_lib",
    MagicMock(
        cuInit=MagicMock(return_value=0), cuStreamCreate=MagicMock(return_value=0)
    ),
)
def test_dispatcher_matmul_no_cublas(*args):
    graph = create_matmul_graph()
    with patch.object(exe.CUDAMemoryPlanner, "build_arena"):
        dispatcher = Dispatcher(graph)
        assert dispatcher.initialized
        node = graph.nodes[0]
        with pytest.raises(RuntimeError, match="cuBLAS is required"):
            dispatcher._execute_matmul(node)


@patch.object(exe, "is_cuda_available", return_value=True)
@patch.object(exe, "is_cublas_available", return_value=False)
@patch.object(exe, "is_cudnn_available", return_value=False)
@patch.object(
    exe,
    "_cuda_lib",
    MagicMock(
        cuInit=MagicMock(return_value=0), cuStreamCreate=MagicMock(return_value=0)
    ),
)
def test_dispatcher_unsupported_op(*args):
    graph = Graph("unsupported")
    graph.add_node(Node("UnknownOp", ["A"], ["B"], {}))
    with patch.object(exe.CUDAMemoryPlanner, "build_arena"):
        graph.add_tensor(Tensor("A", (1,), DType.FLOAT32))
        graph.add_tensor(Tensor("B", (1,), DType.FLOAT32))
        dispatcher = Dispatcher(graph)
        inputs = {"A": np.array([1.0], dtype=np.float32)}
        with pytest.raises(RuntimeError):
            dispatcher.run(inputs)


@patch.object(exe, "is_cuda_available", return_value=True)
@patch.object(exe, "is_cublas_available", return_value=False)
@patch.object(exe, "is_cudnn_available", return_value=False)
@patch.object(
    exe,
    "_cuda_lib",
    MagicMock(
        cuInit=MagicMock(return_value=0), cuStreamCreate=MagicMock(return_value=0)
    ),
)
def test_dispatcher_dynamic_dim(*args):
    graph = Graph("dynamic")
    graph.add_node(Node("Relu", ["X"], ["Y"], {}))
    graph.add_tensor(Tensor("Y", (DynamicDim("N"), 2), DType.FLOAT32))
    graph.outputs = ["Y"]
    with patch.object(exe.CUDAMemoryPlanner, "build_arena"):
        dispatcher = Dispatcher(graph)
        assert dispatcher.initialized


def test_dispatcher_del_exceptions():
    graph = create_simple_graph()
    dispatcher = Dispatcher(graph)
    dispatcher.initialized = True
    with (
        patch.object(exe, "is_cublas_available", return_value=True),
        patch.object(exe, "is_cudnn_available", return_value=True),
        patch.object(exe, "is_cuda_available", return_value=True),
        patch.object(
            exe,
            "_cublas_lib",
            MagicMock(cublasDestroy_v2=MagicMock(side_effect=Exception)),
        ),
        patch.object(
            exe, "_cudnn_lib", MagicMock(cudnnDestroy=MagicMock(side_effect=Exception))
        ),
        patch.object(
            exe,
            "_cuda_lib",
            MagicMock(cuStreamDestroy_v2=MagicMock(side_effect=Exception)),
        ),
    ):
        dispatcher.cublas_handle = 1
        dispatcher.cudnn_handle = 1
        dispatcher.stream = 1
        dispatcher.__del__()


@patch.object(exe, "is_cuda_available", return_value=True)
@patch.object(exe, "is_cublas_available", return_value=True)
@patch.object(exe, "is_cudnn_available", return_value=True)
@patch.object(
    exe,
    "_cuda_lib",
    MagicMock(
        cuInit=MagicMock(return_value=0),
        cuStreamCreate=MagicMock(return_value=0),
        cuModuleLoadData=MagicMock(return_value=0),
        cuModuleGetFunction=MagicMock(return_value=0),
        cuLaunchKernel=MagicMock(return_value=0),
        cuModuleUnload=MagicMock(return_value=0),
    ),
)
@patch.object(
    exe,
    "_cublas_lib",
    MagicMock(
        cublasCreate_v2=MagicMock(return_value=0),
        cublasSetStream_v2=MagicMock(return_value=0),
    ),
)
@patch.object(exe, "_cudnn_lib", MagicMock(cudnnCreate=MagicMock(return_value=0)))
def test_dispatcher_elementwise(*args):
    graph = Graph("math")
    graph.inputs = ["A", "B"]
    graph.outputs = ["C"]
    node = Node("Add", ["A", "B"], ["C"], {})
    graph.add_node(node)
    with (
        patch.object(exe.CUDAMemoryPlanner, "build_arena"),
        patch.object(exe.CUDAMemoryPlanner, "set_tensor"),
        patch.object(
            exe.CUDAMemoryPlanner, "get_tensor_ptr", return_value=exe.CUdeviceptr(0)
        ),
        patch.object(
            exe.CUDAMemoryPlanner, "get_host_tensor", return_value=np.zeros((2, 2))
        ),
    ):
        dispatcher = Dispatcher(graph)
        assert dispatcher.initialized
        inputs = {
            "A": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            "B": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        }
        dispatcher.planner.tensors_shape_dtype["A"] = (2, 2), np.dtype("float32")
        dispatcher.planner.tensors_shape_dtype["B"] = (2, 2), np.dtype("float32")
        dispatcher.planner.tensors_shape_dtype["C"] = (2, 2), np.dtype("float32")
        with patch.object(exe.CUDACompiler, "compile_kernel", return_value=b"fake ptx"):
            results = dispatcher.run(inputs)
            assert "C" in results
            exe._cuda_lib.cuLaunchKernel.assert_called()
        with patch.object(exe.CUDACompiler, "compile_kernel", return_value=b""):
            results = dispatcher.run(inputs)
            assert "C" in results
