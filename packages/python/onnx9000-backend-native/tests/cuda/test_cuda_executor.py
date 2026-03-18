"""Tests the cuda executor module functionality."""

import ctypes
from unittest.mock import MagicMock, patch

import numpy as np
import onnx9000.backends.cuda.executor as executor
import pytest
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor


def test_cuda_executor_fallback() -> None:
    """Tests the cuda executor fallback functionality."""
    g = Graph("test")
    g.inputs.append(Tensor("A", (1, 2), DType.FLOAT32))
    g.outputs.append(Tensor("B", (1, 2), DType.FLOAT32))
    g.add_node(Node("Relu", ["A"], ["B"]))

    with (
        patch("onnx9000.backends.cuda.executor.is_cuda_available", return_value=False),
        patch("onnx9000.backends.memory.cuda_arena.is_cuda_available", return_value=False),
    ):
        dispatcher = executor.Dispatcher(g)

    assert dispatcher.initialized is False

    # Run
    inp = np.array([[1.0, -1.0]], dtype=np.float32)
    with patch.object(dispatcher.cpu_fallback, "execute") as mock_exec:
        mock_exec.return_value = {"B": np.array([[1.0, 0.0]], dtype=np.float32)}
        out = dispatcher.run({"A": inp})

    assert "B" in out
    np.testing.assert_allclose(out["B"], np.array([[1.0, 0.0]], dtype=np.float32))


def test_cuda_executor_matmul_cublas() -> None:
    """Tests the cuda executor matmul cublas functionality."""
    g = Graph("test")
    g.inputs.append(Tensor("A", (2, 2), DType.FLOAT32))
    g.inputs.append(Tensor("B", (2, 2), DType.FLOAT32))
    g.outputs.append(Tensor("C", (2, 2), DType.FLOAT32))
    g.add_node(Node("MatMul", ["A", "B"], ["C"]))

    with (
        patch("onnx9000.backends.cuda.executor.is_cuda_available", return_value=True),
        patch("onnx9000.backends.memory.cuda_arena.is_cuda_available", return_value=True),
        patch("onnx9000.backends.cuda.executor.is_cublas_available", return_value=True),
        patch("onnx9000.backends.cuda.executor.is_cudnn_available", return_value=True),
        patch(
            "onnx9000.backends.cuda.executor._cuda_lib",
            MagicMock(
                **{
                    "cuInit.return_value": 0,
                    "cuStreamCreate.return_value": 0,
                    "cuModuleLoadData.return_value": 0,
                    "cuModuleGetFunction.return_value": 0,
                    "cuLaunchKernel.return_value": 0,
                }
            ),
        ),
        patch(
            "onnx9000.backends.memory.cuda_arena._cuda_lib",
            MagicMock(
                **{
                    "cuMemAlloc.return_value": 0,
                    "cuMemAlloc_v2.return_value": 0,
                    "cuMemFree.return_value": 0,
                    "cuMemcpyHtoD_v2.return_value": 0,
                    "cuMemcpyDtoH_v2.return_value": 0,
                }
            ),
        ),
        patch(
            "onnx9000.backends.cuda.executor._cublas_lib",
            MagicMock(
                **{
                    "cublasCreate_v2.return_value": 0,
                    "cublasSetStream_v2.return_value": 0,
                    "cublasSgemm_v2.return_value": 0,
                }
            ),
        ) as mock_cublas,
        patch(
            "onnx9000.backends.cuda.executor._cudnn_lib",
            MagicMock(**{"cudnnCreate.return_value": 0}),
        ),
    ):
        # mock planner ptr so tests don't crash
        with patch(
            "onnx9000.backends.memory.cuda_arena.CUDAMemoryPlanner.get_tensor_ptr",
            return_value=ctypes.c_void_p(123),
        ):
            dispatcher = executor.Dispatcher(g)

            inp_A = np.ones((2, 2), dtype=np.float32)
            inp_B = np.ones((2, 2), dtype=np.float32)
            with patch(
                "onnx9000.backends.memory.cuda_arena.CUDAMemoryPlanner.get_host_tensor",
                return_value=np.zeros((2, 2), dtype=np.float32).tobytes(),
            ):
                dispatcher.run({"A": inp_A, "B": inp_B})

            assert mock_cublas.cublasSgemm_v2.called


def test_cuda_executor_matmul_no_cublas() -> None:
    """Tests the cuda executor matmul no cublas functionality."""
    g = Graph("test")
    g.inputs.append(Tensor("A", (2, 2), DType.FLOAT32))
    g.inputs.append(Tensor("B", (2, 2), DType.FLOAT32))
    g.outputs.append(Tensor("C", (2, 2), DType.FLOAT32))
    g.add_node(Node("MatMul", ["A", "B"], ["C"]))

    with (
        patch("onnx9000.backends.cuda.executor.is_cuda_available", return_value=True),
        patch("onnx9000.backends.memory.cuda_arena.is_cuda_available", return_value=True),
        patch("onnx9000.backends.cuda.executor.is_cublas_available", return_value=False),
        patch(
            "onnx9000.backends.cuda.executor._cuda_lib",
            MagicMock(**{"cuInit.return_value": 0, "cuStreamCreate.return_value": 0}),
        ),
        patch(
            "onnx9000.backends.memory.cuda_arena._cuda_lib",
            MagicMock(
                **{
                    "cuMemAlloc.return_value": 0,
                    "cuMemAlloc_v2.return_value": 0,
                    "cuMemFree.return_value": 0,
                    "cuMemcpyHtoD_v2.return_value": 0,
                    "cuMemcpyDtoH_v2.return_value": 0,
                }
            ),
        ),
    ):
        dispatcher = executor.Dispatcher(g)

        with pytest.raises(RuntimeError, match="cuBLAS is required"):
            dispatcher.run(
                {"A": np.ones((2, 2), dtype=np.float32), "B": np.ones((2, 2), dtype=np.float32)}
            )


def test_cuda_executor_elementwise() -> None:
    """Tests the cuda executor elementwise functionality."""
    for op_type in ["Add", "Sub", "Mul"]:
        g = Graph("test")
        g.inputs.append(Tensor("A", (2, 2), DType.FLOAT32))
        g.inputs.append(Tensor("B", (2, 2), DType.FLOAT32))
        g.outputs.append(Tensor("C", (2, 2), DType.FLOAT32))
        g.add_node(Node(op_type, ["A", "B"], ["C"]))

        with (
            patch("onnx9000.backends.cuda.executor.is_cuda_available", return_value=True),
            patch("onnx9000.backends.memory.cuda_arena.is_cuda_available", return_value=True),
            patch(
                "onnx9000.backends.cuda.executor._cuda_lib",
                MagicMock(
                    **{
                        "cuInit.return_value": 0,
                        "cuStreamCreate.return_value": 0,
                        "cuModuleLoadData.return_value": 0,
                        "cuModuleGetFunction.return_value": 0,
                        "cuLaunchKernel.return_value": 0,
                    }
                ),
            ) as mock_cuda,
            patch(
                "onnx9000.backends.memory.cuda_arena._cuda_lib",
                MagicMock(
                    **{
                        "cuMemAlloc.return_value": 0,
                        "cuMemAlloc_v2.return_value": 0,
                        "cuMemFree.return_value": 0,
                        "cuMemcpyHtoD_v2.return_value": 0,
                        "cuMemcpyDtoH_v2.return_value": 0,
                    }
                ),
            ),
            patch(
                "onnx9000.backends.cuda.executor.CUDACompiler.compile_kernel", return_value=b"ptx"
            ),
            patch(
                "onnx9000.backends.memory.cuda_arena.CUDAMemoryPlanner.get_tensor_ptr",
                return_value=ctypes.c_void_p(123),
            ),
        ):
            dispatcher = executor.Dispatcher(g)
            inp_A = np.ones((2, 2), dtype=np.float32)
            inp_B = np.ones((2, 2), dtype=np.float32)
            with patch(
                "onnx9000.backends.memory.cuda_arena.CUDAMemoryPlanner.get_host_tensor",
                return_value=np.zeros((2, 2), dtype=np.float32).tobytes(),
            ):
                dispatcher.run({"A": inp_A, "B": inp_B})

            assert mock_cuda.cuLaunchKernel.called


def test_cuda_executor_elementwise_no_ptx() -> None:
    """Tests the cuda executor elementwise no ptx functionality."""
    g = Graph("test")
    g.inputs.append(Tensor("A", (2, 2), DType.FLOAT32))
    g.inputs.append(Tensor("B", (2, 2), DType.FLOAT32))
    g.outputs.append(Tensor("C", (2, 2), DType.FLOAT32))
    g.add_node(Node("Add", ["A", "B"], ["C"]))

    with (
        patch("onnx9000.backends.cuda.executor.is_cuda_available", return_value=True),
        patch("onnx9000.backends.memory.cuda_arena.is_cuda_available", return_value=True),
        patch(
            "onnx9000.backends.cuda.executor._cuda_lib",
            MagicMock(
                **{
                    "cuInit.return_value": 0,
                    "cuStreamCreate.return_value": 0,
                    "cuModuleLoadData.return_value": 0,
                    "cuModuleGetFunction.return_value": 0,
                    "cuLaunchKernel.return_value": 0,
                }
            ),
        ),
        patch(
            "onnx9000.backends.memory.cuda_arena._cuda_lib",
            MagicMock(
                **{
                    "cuMemAlloc.return_value": 0,
                    "cuMemAlloc_v2.return_value": 0,
                    "cuMemFree.return_value": 0,
                    "cuMemcpyHtoD_v2.return_value": 0,
                    "cuMemcpyDtoH_v2.return_value": 0,
                }
            ),
        ),
        patch("onnx9000.backends.cuda.executor.CUDACompiler.compile_kernel", return_value=b""),
    ):
        dispatcher = executor.Dispatcher(g)
        inp_A = np.ones((2, 2), dtype=np.float32)
        inp_B = np.ones((2, 2), dtype=np.float32)
        with patch.object(
            dispatcher.cpu_fallback,
            "execute",
            return_value={"C": np.ones((2, 2), dtype=np.float32)},
        ) as mock_exec:
            with patch(
                "onnx9000.backends.memory.cuda_arena.CUDAMemoryPlanner.get_host_tensor",
                return_value=np.zeros((2, 2), dtype=np.float32).tobytes(),
            ):
                dispatcher.run({"A": inp_A, "B": inp_B})
            assert mock_exec.called


def test_cuda_executor_init_memory_dynamic_and_init() -> None:
    """Tests the cuda executor init memory dynamic and init functionality."""
    g = Graph("test")
    # Using dynamic shape for output
    t_out = Tensor("out_t", ("N", 2), DType.FLOAT32)
    t_init = Tensor(
        "init_t", (2,), DType.FLOAT32, data=np.array([1.0, 2.0], dtype=np.float32).tobytes()
    )
    g.tensors["out_t"] = t_out
    g.tensors["init_t"] = t_init
    g.initializers.append("init_t")
    g.outputs.append("out_t")
    g.add_node(Node("Dummy", [], ["out_t"]))

    with (
        patch("onnx9000.backends.cuda.executor.is_cuda_available", return_value=False),
        patch("onnx9000.backends.memory.cuda_arena.is_cuda_available", return_value=False),
    ):
        dispatcher = executor.Dispatcher(g)

    assert dispatcher.planner._cpu_fallback_tensors["init_t"] is not None


def test_cuda_executor_del_handles_errors() -> None:
    """Tests the cuda executor del handles errors functionality."""
    g = Graph("test")
    with (
        patch("onnx9000.backends.cuda.executor.is_cuda_available", return_value=True),
        patch("onnx9000.backends.cuda.executor.is_cublas_available", return_value=True),
        patch("onnx9000.backends.cuda.executor.is_cudnn_available", return_value=True),
        patch(
            "onnx9000.backends.cuda.executor._cuda_lib",
            MagicMock(**{"cuInit.return_value": 0, "cuStreamCreate.return_value": 0}),
        ) as mock_cuda,
        patch(
            "onnx9000.backends.cuda.executor._cublas_lib",
            MagicMock(**{"cublasCreate_v2.return_value": 0, "cublasSetStream_v2.return_value": 0}),
        ) as mock_cublas,
        patch(
            "onnx9000.backends.cuda.executor._cudnn_lib",
            MagicMock(**{"cudnnCreate.return_value": 0}),
        ) as mock_cudnn,
    ):
        dispatcher = executor.Dispatcher(g)

        dispatcher.cublas_handle = MagicMock()
        dispatcher.cudnn_handle = MagicMock()
        dispatcher.stream = MagicMock()
        dispatcher.initialized = True

        mock_cublas.cublasDestroy_v2.side_effect = Exception("cublas error")
        mock_cudnn.cudnnDestroy.side_effect = Exception("cudnn error")
        mock_cuda.cuStreamDestroy_v2.side_effect = Exception("cuda error")

        dispatcher.__del__()


def test_ignore_this5() -> None:
    """Executes the ignore this5 operation."""
    g = Graph("test")
    with (
        patch("onnx9000.backends.cuda.executor.is_cuda_available", return_value=True),
        patch("onnx9000.backends.memory.cuda_arena.is_cuda_available", return_value=True),
        patch("onnx9000.backends.cuda.executor.is_cublas_available", return_value=True),
        patch("onnx9000.backends.cuda.executor.is_cudnn_available", return_value=True),
        patch(
            "onnx9000.backends.cuda.executor._cuda_lib",
            MagicMock(
                **{
                    "cuInit.return_value": 0,
                    "cuStreamCreate.return_value": 0,
                    "cuModuleLoadData.return_value": 0,
                    "cuModuleGetFunction.return_value": 0,
                    "cuLaunchKernel.return_value": 0,
                }
            ),
        ) as mock_cuda,
        patch(
            "onnx9000.backends.memory.cuda_arena._cuda_lib",
            MagicMock(
                **{
                    "cuMemAlloc.return_value": 0,
                    "cuMemAlloc_v2.return_value": 0,
                    "cuMemFree.return_value": 0,
                    "cuMemcpyHtoD_v2.return_value": 0,
                    "cuMemcpyDtoH_v2.return_value": 0,
                }
            ),
        ),
        patch(
            "onnx9000.backends.cuda.executor._cublas_lib",
            MagicMock(
                **{
                    "cublasCreate_v2.return_value": 0,
                    "cublasSetStream_v2.return_value": 0,
                    "cublasSgemm_v2.return_value": 0,
                }
            ),
        ) as mock_cublas,
        patch(
            "onnx9000.backends.cuda.executor._cudnn_lib",
            MagicMock(**{"cudnnCreate.return_value": 0}),
        ) as mock_cudnn,
    ):
        dispatcher = executor.Dispatcher(g)

        mock_cublas.cublasDestroy_v2.side_effect = Exception("cublas error")
        mock_cudnn.cudnnDestroy.side_effect = Exception("cudnn error")
        mock_cuda.cuStreamDestroy_v2.side_effect = Exception("cuda error")

        # Test del paths when exceptions are thrown
        dispatcher.__del__()
        # Assertions are implicitly that __del__ doesn't crash


def test_cuda_executor_matmul_cublas_static_alloc() -> None:
    """Tests the cuda executor matmul cublas static alloc functionality."""
    g = Graph("test")
    g.inputs.append(Tensor("A", (2, 2), DType.FLOAT32))
    g.inputs.append(Tensor("B", (2, 2), DType.FLOAT32))
    g.outputs.append(Tensor("C", (2, 2), DType.FLOAT32))
    g.add_node(Node("MatMul", ["A", "B"], ["C"]))
    g.tensors["A"] = Tensor("A", (2, 2), DType.FLOAT32)
    g.tensors["B"] = Tensor("B", (2, 2), DType.FLOAT32)
    g.tensors["C"] = Tensor("C", (2, 2), DType.FLOAT32)

    with (
        patch("onnx9000.backends.cuda.executor.is_cuda_available", return_value=True),
        patch("onnx9000.backends.memory.cuda_arena.is_cuda_available", return_value=True),
        patch("onnx9000.backends.cuda.executor.is_cublas_available", return_value=True),
        patch("onnx9000.backends.cuda.executor.is_cudnn_available", return_value=True),
        patch(
            "onnx9000.backends.cuda.executor._cuda_lib",
            MagicMock(
                **{
                    "cuInit.return_value": 0,
                    "cuStreamCreate.return_value": 0,
                    "cuModuleLoadData.return_value": 0,
                    "cuModuleGetFunction.return_value": 0,
                    "cuLaunchKernel.return_value": 0,
                }
            ),
        ),
        patch(
            "onnx9000.backends.memory.cuda_arena._cuda_lib",
            MagicMock(
                **{
                    "cuMemAlloc.return_value": 0,
                    "cuMemAlloc_v2.return_value": 0,
                    "cuMemFree.return_value": 0,
                    "cuMemcpyHtoD_v2.return_value": 0,
                    "cuMemcpyDtoH_v2.return_value": 0,
                }
            ),
        ),
        patch(
            "onnx9000.backends.cuda.executor._cublas_lib",
            MagicMock(
                **{
                    "cublasCreate_v2.return_value": 0,
                    "cublasSetStream_v2.return_value": 0,
                    "cublasSgemm_v2.return_value": 0,
                }
            ),
        ),
        patch(
            "onnx9000.backends.cuda.executor._cudnn_lib",
            MagicMock(**{"cudnnCreate.return_value": 0}),
        ),
        patch(
            "onnx9000.backends.memory.cuda_arena.CUDAMemoryPlanner.get_tensor_ptr",
            return_value=ctypes.c_void_p(123),
        ),
        patch(
            "onnx9000.backends.memory.cuda_arena.CUDAMemoryPlanner.get_host_tensor",
            return_value=np.zeros((2, 2), dtype=np.float32).tobytes(),
        ),
    ):
        dispatcher = executor.Dispatcher(g)
        assert "C" in dispatcher.planner.offsets
        inp_A = np.ones((2, 2), dtype=np.float32)
        inp_B = np.ones((2, 2), dtype=np.float32)
        dispatcher.run({"A": inp_A, "B": inp_B})


def test_cuda_executor_execute_node_fallback() -> None:
    """Tests the cuda executor execute node fallback functionality."""
    g = Graph("test")
    g.inputs.append(Tensor("A", (2, 2), DType.FLOAT32))
    g.outputs.append(Tensor("B", (2, 2), DType.FLOAT32))
    g.add_node(Node("Relu", ["A"], ["B"]))
    g.tensors["A"] = Tensor("A", (2, 2), DType.FLOAT32)
    g.tensors["B"] = Tensor("B", (2, 2), DType.FLOAT32)

    with (
        patch("onnx9000.backends.cuda.executor.is_cuda_available", return_value=True),
        patch("onnx9000.backends.memory.cuda_arena.is_cuda_available", return_value=True),
        patch("onnx9000.backends.cuda.executor.is_cublas_available", return_value=True),
        patch("onnx9000.backends.cuda.executor.is_cudnn_available", return_value=True),
        patch(
            "onnx9000.backends.cuda.executor._cuda_lib",
            MagicMock(**{"cuInit.return_value": 0, "cuStreamCreate.return_value": 0}),
        ),
        patch(
            "onnx9000.backends.memory.cuda_arena._cuda_lib",
            MagicMock(
                **{
                    "cuMemAlloc.return_value": 0,
                    "cuMemAlloc_v2.return_value": 0,
                    "cuMemFree.return_value": 0,
                    "cuMemcpyHtoD_v2.return_value": 0,
                    "cuMemcpyDtoH_v2.return_value": 0,
                }
            ),
        ),
        patch(
            "onnx9000.backends.cuda.executor._cublas_lib",
            MagicMock(**{"cublasCreate_v2.return_value": 0, "cublasSetStream_v2.return_value": 0}),
        ),
        patch(
            "onnx9000.backends.cuda.executor._cudnn_lib",
            MagicMock(**{"cudnnCreate.return_value": 0}),
        ),
        patch(
            "onnx9000.backends.memory.cuda_arena.CUDAMemoryPlanner.get_tensor_ptr",
            return_value=ctypes.c_void_p(123),
        ),
        patch(
            "onnx9000.backends.memory.cuda_arena.CUDAMemoryPlanner.get_host_tensor",
            return_value=np.zeros((2, 2), dtype=np.float32).tobytes(),
        ),
    ):
        dispatcher = executor.Dispatcher(g)
        inp_A = np.ones((2, 2), dtype=np.float32)

        with patch.object(
            dispatcher.cpu_fallback,
            "execute",
            return_value={"B": np.ones((2, 2), dtype=np.float32)},
        ) as mock_exec:
            dispatcher.run({"A": inp_A})
            assert mock_exec.called
