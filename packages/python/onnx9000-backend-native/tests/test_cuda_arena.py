"""Tests the cuda arena module functionality."""

from typing import NoReturn
from unittest.mock import MagicMock

import numpy as np
import onnx9000.backends.memory.cuda_arena as cuda_arena_mod
import pytest
from onnx9000.backends.memory.cuda_arena import CUDAMemoryPlanner


@pytest.fixture
def mock_cuda(monkeypatch):
    """Execute the mock cuda operation."""
    monkeypatch.setattr(cuda_arena_mod, "is_cuda_available", lambda: True)
    mock_lib = MagicMock()

    def mock_alloc(ptr_ref, size) -> int:
        """Test the mock alloc functionality."""
        ptr_ref._obj.value = 1000
        return 0

    mock_lib.cuMemAlloc_v2.side_effect = mock_alloc
    mock_lib.cuMemcpyHtoD_v2.return_value = 0
    mock_lib.cuMemcpyDtoH_v2.return_value = 0
    mock_lib.cuMemFree_v2.return_value = 0
    monkeypatch.setattr(cuda_arena_mod, "_cuda_lib", mock_lib)
    monkeypatch.setattr(cuda_arena_mod, "check_cuda_error", lambda x: None)
    return mock_lib


def test_cuda_arena_no_cuda(monkeypatch) -> None:
    """Tests the cuda arena no cuda functionality."""
    monkeypatch.setattr(cuda_arena_mod, "is_cuda_available", lambda: False)
    planner = CUDAMemoryPlanner()
    planner.allocate_static("A", 1024, (2, 2), np.float32)
    planner.build_arena()
    arr = np.array([1.0], dtype=np.float32)
    planner.set_tensor("A", arr)
    assert planner._cpu_fallback_tensors["A"] is arr
    out = planner.get_host_tensor("A")
    assert out is arr


def test_cuda_arena_with_cuda(mock_cuda) -> None:
    """Tests the cuda arena with cuda functionality."""
    planner = CUDAMemoryPlanner()
    planner.allocate_static("A", 8, (2,), np.float32)
    planner.build_arena()
    assert planner.arena_ptr.value == 1000
    arr = np.array([1.0, 2.0], dtype=np.float32)
    planner.set_tensor("A", arr)
    mock_cuda.cuMemcpyHtoD_v2.assert_called()
    ptr = planner.get_tensor_ptr("A")
    assert ptr.value == 1000
    out = planner.get_host_tensor("A")
    assert out.shape == (2,)
    arr_big = np.zeros(2000, dtype=np.uint8)
    planner.set_tensor("A", arr_big)
    assert "A" in planner.dynamic_allocations
    arr_bigger = np.zeros(3000, dtype=np.uint8)
    planner.set_tensor("A", arr_bigger)
    planner.set_tensor("A", arr_big)
    planner.set_tensor("B", arr)
    assert "B" in planner.dynamic_allocations
    planner.cleanup()
    mock_cuda.cuMemFree_v2.assert_called()


def test_cuda_arena_errors(mock_cuda) -> None:
    """Tests the cuda arena errors functionality."""
    planner = CUDAMemoryPlanner()
    planner.build_arena()
    with pytest.raises(RuntimeError):
        planner.get_tensor_ptr("MISSING")
    with pytest.raises(RuntimeError):
        planner.get_host_tensor("MISSING")


def test_cuda_arena_dynamic_reallocation_and_fetch(mock_cuda) -> None:
    """Tests the cuda arena dynamic reallocation and fetch functionality."""
    planner = CUDAMemoryPlanner()
    arr = np.array([1.0], dtype=np.float32)
    planner.set_tensor("B", arr)
    assert "B" in planner.dynamic_allocations
    ptr = planner.get_tensor_ptr("B")
    assert ptr.value == 1000
    out = planner.get_host_tensor("B")
    assert out.shape == (1,)
    arr_big = np.zeros(2000, dtype=np.uint8)
    planner.set_tensor("B", arr_big)

    class BrokenPlanner(CUDAMemoryPlanner):
        """Represents the BrokenPlanner class and its associated logic."""

        def cleanup(self) -> NoReturn:
            """Test the cleanup functionality."""
            raise Exception("broken")

    bp = BrokenPlanner()
    del bp


def test_cuda_arena_extra_methods(mock_cuda) -> None:
    """Tests the cuda arena extra methods functionality."""
    import ctypes

    planner = CUDAMemoryPlanner()
    ptr = planner.allocate_pinned(1024)
    assert ptr is not None
    mptr = planner.allocate_managed(1024)
    assert mptr is not None
    planner.synchronize_stream(ctypes.c_void_p(1234))
    arr = np.array([1.0], dtype=np.float32)
    planner.set_tensor("A", arr, stream=ctypes.c_void_p(1234))
    out = planner.get_host_tensor("A", stream=ctypes.c_void_p(1234))
    assert out.shape == (1,)


def test_cuda_arena_extra_methods_no_cuda(monkeypatch) -> None:
    """Tests the cuda arena extra methods no cuda functionality."""
    monkeypatch.setattr(cuda_arena_mod, "is_cuda_available", lambda: False)
    import ctypes

    planner = CUDAMemoryPlanner()
    planner.allocate_pinned(1024)
    planner.allocate_managed(1024)
    planner.synchronize_stream(ctypes.c_void_p(1234))
