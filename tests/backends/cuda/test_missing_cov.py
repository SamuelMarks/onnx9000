import numpy as np
from unittest.mock import patch
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.core.dtypes import DType


def test_executor_initializer():
    from onnx9000.backends.cuda.executor import Dispatcher

    graph = Graph("test")
    t = Tensor(
        "init",
        (2,),
        DType.FLOAT32,
        is_initializer=True,
        data=np.array([1.0, 2.0], dtype=np.float32),
    )
    graph.tensors["init"] = t
    graph.initializers.append("init")
    with (
        patch("onnx9000.backends.cuda.executor.is_cuda_available", return_value=True),
        patch(
            "onnx9000.backends.cuda.executor.is_cublas_available", return_value=False
        ),
        patch("onnx9000.backends.cuda.executor.is_cudnn_available", return_value=False),
        patch("onnx9000.backends.cuda.executor._cuda_lib") as mock_cuda,
        patch("onnx9000.backends.cuda.memory.is_cuda_available", return_value=True),
        patch("onnx9000.backends.cuda.memory._cuda_lib") as mock_mem_cuda,
    ):
        mock_cuda.cuInit.return_value = 0
        mock_cuda.cuStreamCreate.return_value = 0

        def mock_alloc(ptr_ref, size):
            ptr_ref._obj.value = 1234
            return 0

        mock_mem_cuda.cuMemAlloc_v2.side_effect = mock_alloc
        mock_mem_cuda.cuMemcpyHtoD_v2.return_value = 0
        dispatcher = Dispatcher(graph)
        assert "init" in dispatcher.planner.dynamic_allocations


def test_memory_set_tensor_dynamic_larger():
    from onnx9000.backends.cuda.memory import CUDAMemoryPlanner
    import ctypes

    planner = CUDAMemoryPlanner()
    with (
        patch("onnx9000.backends.cuda.memory.is_cuda_available", return_value=True),
        patch("onnx9000.backends.cuda.memory._cuda_lib") as mock_cuda,
    ):

        def mock_alloc(ptr_ref, size):
            ptr_ref._obj.value = 1234
            return 0

        mock_cuda.cuMemAlloc_v2.side_effect = mock_alloc
        mock_cuda.cuMemcpyHtoD_v2.return_value = 0
        mock_cuda.cuMemFree_v2.return_value = 0
        data1 = np.array([1.0], dtype=np.float32)
        planner.set_tensor("dyn_tensor", data1)
        data2 = np.array([1.0, 2.0], dtype=np.float32)
        planner.set_tensor("dyn_tensor", data2)
        mock_cuda.cuMemFree_v2.assert_called()


def test_memory_cleanup_arena():
    from onnx9000.backends.cuda.memory import CUDAMemoryPlanner

    planner = CUDAMemoryPlanner()
    with (
        patch("onnx9000.backends.cuda.memory.is_cuda_available", return_value=True),
        patch("onnx9000.backends.cuda.memory._cuda_lib") as mock_cuda,
    ):

        def mock_alloc(ptr_ref, size):
            ptr_ref._obj.value = 1234
            return 0

        mock_cuda.cuMemAlloc_v2.side_effect = mock_alloc
        mock_cuda.cuMemFree_v2.return_value = 0
        planner.allocate_static("static_tensor", 100, (25,), np.dtype("float32"))
        planner.build_arena()
        assert planner.arena_ptr.value != 0
        planner.cleanup()
        assert planner.arena_ptr.value == 0
        mock_cuda.cuMemFree_v2.assert_called()
