import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from onnx9000.backends.cuda.memory import CUDAMemoryPlanner
import onnx9000.backends.cuda.memory as mem


def get_mock_cuda_lib():
    mock_lib = MagicMock()
    mock_lib.cuMemAlloc_v2.return_value = 0
    mock_lib.cuMemFree_v2.return_value = 0
    mock_lib.cuMemcpyHtoD_v2.return_value = 0
    mock_lib.cuMemcpyDtoH_v2.return_value = 0
    return mock_lib


@patch.object(mem, "is_cuda_available", return_value=True)
@patch.object(mem, "_cuda_lib", get_mock_cuda_lib())
def test_cuda_memory_planner_static(mock_avail):
    planner = CUDAMemoryPlanner()
    planner.allocate_static("a", 16, (4,), np.dtype("float32"))
    planner.build_arena()
    assert planner.current_offset == 16
    mem._cuda_lib.cuMemAlloc_v2.assert_called()
    ptr = planner.get_tensor_ptr("a")
    assert ptr is not None
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    planner.set_tensor("a", data)
    mem._cuda_lib.cuMemcpyHtoD_v2.assert_called()
    out_data = planner.get_host_tensor("a")
    assert out_data.shape == (4,)
    mem._cuda_lib.cuMemcpyDtoH_v2.assert_called()


@patch.object(mem, "is_cuda_available", return_value=True)
@patch.object(mem, "_cuda_lib", get_mock_cuda_lib())
def test_cuda_memory_planner_dynamic(mock_avail):
    planner = CUDAMemoryPlanner()
    planner.allocate_static("a", 4, (1,), np.dtype("float32"))
    planner.build_arena()
    data = np.array([1.0, 2.0], dtype=np.float32)
    planner.set_tensor("a", data)
    assert "a" in planner.dynamic_allocations
    ptr = planner.get_tensor_ptr("a")
    assert ptr is not None
    data2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    planner.set_tensor("a", data2)
    out_data = planner.get_host_tensor("a")
    assert out_data.dtype == np.float32
    planner.cleanup()


@patch.object(mem, "is_cuda_available", return_value=True)
@patch.object(mem, "_cuda_lib", get_mock_cuda_lib())
def test_cuda_memory_planner_unallocated(mock_avail):
    planner = CUDAMemoryPlanner()
    data = np.array([1.0, 2.0], dtype=np.float32)
    planner.set_tensor("b", data)
    assert "b" in planner.dynamic_allocations
    with pytest.raises(RuntimeError):
        planner.get_tensor_ptr("c")
    with pytest.raises(RuntimeError):
        planner.get_host_tensor("c")


def test_cuda_memory_planner_no_cuda():
    planner = CUDAMemoryPlanner()
    planner.build_arena()
    planner.set_tensor("a", np.array([1.0]))
    planner.cleanup()


@patch.object(mem, "is_cuda_available", return_value=True)
@patch.object(mem, "_cuda_lib", get_mock_cuda_lib())
def test_cuda_memory_planner_dynamic_shrink(mock_avail):
    planner = CUDAMemoryPlanner()
    data = np.array([1.0, 2.0], dtype=np.float32)
    planner.set_tensor("c", data)
    assert "c" in planner.dynamic_allocations
    data_small = np.array([1.0], dtype=np.float32)
    planner.set_tensor("c", data_small)


def test_cuda_memory_del_exception():
    planner = CUDAMemoryPlanner()
    with patch.object(planner, "cleanup", side_effect=Exception):
        planner.__del__()
