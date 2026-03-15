import numpy as np
import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.core.dtypes import DType
from onnx9000.backends.rocm.executor import Dispatcher
import onnx9000.backends.rocm.executor as exe
import onnx9000.backends.rocm.bindings as bind
from onnx9000.backends.rocm.compiler import ROCmCompiler


def test_bindings_availability():
    assert isinstance(bind.is_hip_available(), bool)
    assert isinstance(bind.is_rocblas_available(), bool)
    assert isinstance(bind.is_miopen_available(), bool)


def test_bindings_errors():
    bind.check_hip_error(0)
    with pytest.raises(RuntimeError):
        bind.check_hip_error(1)
    bind.check_rocblas_error(0)
    with pytest.raises(RuntimeError):
        bind.check_rocblas_error(1)
    bind.check_miopen_error(0)
    with pytest.raises(RuntimeError):
        bind.check_miopen_error(1)


def test_register_apis():
    mock_lib = MagicMock()
    bind._register_hip_api(mock_lib)
    assert hasattr(mock_lib.hipMalloc, "argtypes")
    mock_rocblas = MagicMock()
    bind._register_rocblas_api(mock_rocblas)
    assert hasattr(mock_rocblas.rocblas_create_handle, "argtypes")
    mock_miopen = MagicMock()
    bind._register_miopen_api(mock_miopen)
    assert hasattr(mock_miopen.miopenCreate, "argtypes")


def test_compiler_success():
    with patch("subprocess.run") as mock_run, tempfile.TemporaryDirectory() as tmp_dir:

        def fake_run(*args, **kwargs):
            out_file = args[0][args[0].index("-o") + 1]
            with open(out_file, "wb") as f:
                f.write(b"fake hsaco content")
            return MagicMock(returncode=0)

        mock_run.side_effect = fake_run
        result = ROCmCompiler.compile_kernel(
            "fake code", "test_kernel", cache_dir=tmp_dir
        )
        assert result == b"fake hsaco content"
        mock_run.reset_mock()
        result_cached = ROCmCompiler.compile_kernel(
            "fake code", "test_kernel", cache_dir=tmp_dir
        )
        assert result_cached == b"fake hsaco content"
        mock_run.assert_not_called()


def test_compiler_not_found():
    with patch("subprocess.run", side_effect=FileNotFoundError):
        result = ROCmCompiler.compile_kernel("fake code", "test_kernel2")
        assert result == b""


def test_compiler_failure():
    import subprocess

    error = subprocess.CalledProcessError(1, ["hipcc"], stderr=b"syntax error")
    with patch("subprocess.run", side_effect=error):
        with pytest.raises(RuntimeError, match="ROCm Kernel compilation failed"):
            ROCmCompiler.compile_kernel("fake code", "test_kernel3")


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


def test_executor_fallback():
    graph = create_simple_graph()
    dispatcher = Dispatcher(graph)
    assert not dispatcher.initialized
    inputs = {"X": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)}
    results = dispatcher.run(inputs)
    assert "Y" in results


@patch.object(exe, "is_hip_available", return_value=True)
@patch.object(exe, "is_rocblas_available", return_value=True)
@patch.object(exe, "is_miopen_available", return_value=True)
@patch.object(
    exe,
    "_hip_lib",
    MagicMock(
        hipStreamCreate=MagicMock(return_value=0),
        hipStreamDestroy=MagicMock(return_value=0),
    ),
)
@patch.object(
    exe,
    "_rocblas_lib",
    MagicMock(
        rocblas_create_handle=MagicMock(return_value=0),
        rocblas_set_stream=MagicMock(return_value=0),
        rocblas_destroy_handle=MagicMock(return_value=0),
    ),
)
@patch.object(
    exe,
    "_miopen_lib",
    MagicMock(
        miopenCreate=MagicMock(return_value=0), miopenDestroy=MagicMock(return_value=0)
    ),
)
def test_executor_with_rocm(*args):
    graph = create_simple_graph()
    dispatcher = Dispatcher(graph)
    assert dispatcher.initialized
    graph = create_matmul_graph()
    dispatcher = Dispatcher(graph)
    inputs = {
        "A": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "B": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    }
    graph.nodes[0].op_type = "Add"
    dispatcher.run(inputs)
    graph.nodes[0].op_type = "Sub"
    dispatcher.run(inputs)
    graph.nodes[0].op_type = "UnknownOp"
    with pytest.raises(RuntimeError):
        dispatcher.run(inputs)
    dispatcher.__del__()


@patch.object(exe, "is_hip_available", return_value=True)
@patch.object(exe, "is_rocblas_available", return_value=False)
@patch.object(exe, "is_miopen_available", return_value=False)
@patch.object(exe, "_hip_lib", MagicMock(hipStreamCreate=MagicMock(return_value=0)))
def test_executor_matmul_missing_rocblas(*args):
    graph = create_matmul_graph()
    dispatcher = Dispatcher(graph)
    with pytest.raises(RuntimeError, match="rocBLAS is required"):
        dispatcher._execute_matmul(graph.nodes[0])


def test_executor_del_exceptions():
    graph = create_simple_graph()
    dispatcher = Dispatcher(graph)
    dispatcher.initialized = True
    with (
        patch.object(exe, "is_hip_available", return_value=True),
        patch.object(exe, "is_rocblas_available", return_value=True),
        patch.object(exe, "is_miopen_available", return_value=True),
        patch.object(
            exe,
            "_hip_lib",
            MagicMock(hipStreamDestroy=MagicMock(side_effect=Exception)),
        ),
        patch.object(
            exe,
            "_rocblas_lib",
            MagicMock(rocblas_destroy_handle=MagicMock(side_effect=Exception)),
        ),
        patch.object(
            exe,
            "_miopen_lib",
            MagicMock(miopenDestroy=MagicMock(side_effect=Exception)),
        ),
    ):
        dispatcher.rocblas_handle = 1
        dispatcher.miopen_handle = 1
        dispatcher.stream = 1
        dispatcher.__del__()


def test_executor_init_with_initializers():
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
    results = dispatcher.run(inputs)
    assert "C" in results


@patch.object(exe, "is_hip_available", return_value=True)
@patch.object(exe, "is_rocblas_available", return_value=True)
@patch.object(exe, "is_miopen_available", return_value=True)
@patch.object(exe, "_hip_lib", MagicMock(hipStreamCreate=MagicMock(return_value=0)))
@patch.object(
    exe,
    "_rocblas_lib",
    MagicMock(
        rocblas_create_handle=MagicMock(return_value=0),
        rocblas_set_stream=MagicMock(return_value=0),
    ),
)
@patch.object(exe, "_miopen_lib", MagicMock(miopenCreate=MagicMock(return_value=0)))
def test_executor_matmul_success(*args):
    graph = create_matmul_graph()
    dispatcher = Dispatcher(graph)
    inputs = {
        "A": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "B": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    }
    dispatcher.run(inputs)
