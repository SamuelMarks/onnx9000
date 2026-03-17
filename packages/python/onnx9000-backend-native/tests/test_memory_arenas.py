"""Tests the memory arenas module functionality."""

import contextlib
import gc
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from onnx9000.backends.memory.cpu_arena import CPUMemoryPlanner
from onnx9000.backends.memory.metal_arena import MetalMemoryPlanner


def test_cpu_memory_planner() -> None:
    """Tests the cpu memory planner functionality."""
    planner = CPUMemoryPlanner()
    planner.allocate_static("A", 1024, (2, 2), "FLOAT32")
    planner.allocate_static("B", 1024, (2, 2), "FLOAT32")
    planner.build_arena()
    assert planner.arena_mmap is not None
    arr = np.array([1.0, 2.0], dtype=np.float32)
    planner.set_tensor("A", memoryview(arr.tobytes()), (2,), "FLOAT32")
    out = planner.get_host_tensor("A")
    assert len(out) == 1024
    ptr = planner.get_tensor_ptr("A")
    assert ptr is not None
    arr_big = np.zeros(2000, dtype=np.uint8)
    planner.set_tensor("A", memoryview(arr_big.tobytes()), (2000,), "UINT8")
    ptr_dyn = planner.get_tensor_ptr("A")
    assert ptr_dyn is not None
    assert len(planner.get_host_tensor("A")) == 2000
    planner.set_tensor("C", memoryview(arr.tobytes()), (2,), "FLOAT32")
    assert len(planner.get_host_tensor("C")) == 8
    with pytest.raises(RuntimeError):
        planner.get_tensor_ptr("UNKNOWN")
    with pytest.raises(RuntimeError):
        planner.get_host_tensor("UNKNOWN")
    gc.collect()
    with contextlib.suppress(BufferError):
        planner.cleanup()


def test_cpu_memory_planner_missing() -> None:
    """Tests the cpu memory planner missing functionality."""
    planner = CPUMemoryPlanner()
    planner.allocate_static("A", 1024, (2, 2), "FLOAT32")
    with pytest.raises(RuntimeError):
        planner.get_tensor_ptr("A")


def test_metal_arena() -> None:
    """Tests the metal arena functionality."""
    planner = MetalMemoryPlanner()
    planner.allocate_static("A", 1024, (2, 2), "FLOAT32")
    planner.allocate_static("B", 1024, (2, 2), "FLOAT32")
    planner.build_arena()
    assert planner.arena_ptr is not None
    arr = np.array([1.0, 2.0], dtype=np.float32)
    planner.set_tensor("A", memoryview(arr.tobytes()), (2,), "FLOAT32")
    out = planner.get_host_tensor("A")
    assert len(out) == 1024
    ptr = planner.get_tensor_ptr("A")
    assert ptr is not None
    arr_big = np.zeros(2000, dtype=np.uint8)
    planner.set_tensor("A", memoryview(arr_big.tobytes()), (2000,), "UINT8")
    ptr_dyn = planner.get_tensor_ptr("A")
    assert ptr_dyn is not None
    assert len(planner.get_host_tensor("A")) == 2000
    planner.set_tensor("C", memoryview(arr.tobytes()), (2,), "FLOAT32")
    assert len(planner.get_host_tensor("C")) == 8
    with pytest.raises(RuntimeError):
        planner.get_tensor_ptr("UNKNOWN")
    with pytest.raises(RuntimeError):
        planner.get_host_tensor("UNKNOWN")
    gc.collect()
    with contextlib.suppress(BufferError):
        planner.cleanup()


def test_metal_arena_missing() -> None:
    """Tests the metal arena missing functionality."""
    planner = MetalMemoryPlanner()
    planner.allocate_static("A", 1024, (2, 2), "FLOAT32")
    with pytest.raises(RuntimeError):
        planner.get_tensor_ptr("A")
    with pytest.raises(RuntimeError):
        planner.get_host_tensor("A")


def test_metal_arena_cleanup_dynamic() -> None:
    """Tests the metal arena cleanup dynamic functionality."""
    planner = MetalMemoryPlanner()
    arr = np.array([1.0], dtype=np.float32)
    planner.set_tensor("A", memoryview(arr.tobytes()), (1,), "FLOAT32")
    planner.set_tensor("B", memoryview(arr.tobytes()), (1,), "FLOAT32")
    gc.collect()
    with contextlib.suppress(BufferError):
        planner.cleanup()


def test_cpu_memory_planner_edge_cases() -> None:
    """Tests the cpu memory planner edge cases functionality."""
    from unittest.mock import patch

    from onnx9000.backends.memory.cpu_arena import CPUMemoryPlanner

    planner = CPUMemoryPlanner()
    mmap0 = planner._allocate_mmap(0)
    assert len(mmap0) == 4096
    with patch("platform.system", return_value="Windows"):
        mmap_win = planner._allocate_mmap(100)
        assert len(mmap_win) == 4096
    with patch("platform.system", return_value="Linux"), patch("ctypes.CDLL") as mock_cdll:
        mock_libc = MagicMock()
        mock_cdll.return_value = mock_libc
        planner._allocate_mmap(3 * 1024 * 1024)
        assert mock_cdll.call_count >= 1
    with patch("platform.system", return_value="Linux"), patch("ctypes.CDLL") as mock_cdll:
        mock_libc = MagicMock()
        mock_libc.madvise.side_effect = Exception("fake error")
        mock_cdll.return_value = mock_libc
        planner._allocate_mmap(3 * 1024 * 1024)
    with patch("platform.system", return_value="Darwin"), patch("ctypes.CDLL") as mock_cdll:
        mock_libc = MagicMock()
        mock_libc.mlock.side_effect = Exception("fake mlock error")
        mock_cdll.return_value = mock_libc
        planner._allocate_mmap(4096)
    with patch("mmap.mmap", side_effect=OSError("fake oom")):
        import pytest

        with pytest.raises(MemoryError):
            planner._allocate_mmap(4096)
    planner.set_tensor("D", memoryview(b"1234"), (4,), "UINT8")
    planner.add_ref("D")
    planner.add_ref("D")
    planner.release_ref("D")
    planner.release_ref("D")
    planner.release_ref("D")
    planner.release_ref("UNKNOWN")


def test_cpu_memory_planner_ref_counting() -> None:
    """Tests the cpu memory planner ref counting functionality."""
    from onnx9000.backends.memory.cpu_arena import CPUMemoryPlanner

    planner = CPUMemoryPlanner()
    planner.allocate_static("D", 4, (4,), "UINT8")
    planner.build_arena()
    planner.add_ref("D")
    import numpy as np

    planner.set_tensor("D", memoryview(np.zeros(10, dtype=np.uint8).tobytes()), (10,), "UINT8")
    planner.release_ref("D")
    planner.release_ref("D")
    assert "D" not in planner.dynamic_allocations
    assert "D" not in planner.dynamic_sizes
    planner.cleanup()
    planner.__del__()


def test_cpu_memory_planner_cleanup_dynamic() -> None:
    """Tests the cpu memory planner cleanup dynamic functionality."""
    from onnx9000.backends.memory.cpu_arena import CPUMemoryPlanner

    planner = CPUMemoryPlanner()
    import numpy as np

    planner.set_tensor("D", memoryview(np.zeros(10, dtype=np.uint8).tobytes()), (10,), "UINT8")
    planner.cleanup()


def test_cpu_arena_reallocate_dynamic() -> None:
    """Tests the cpu arena reallocate dynamic functionality."""
    from onnx9000.backends.memory.cpu_arena import CPUMemoryPlanner

    planner = CPUMemoryPlanner()
    planner.allocate_dynamic("dyn", 100, (10,), "float32")
    # trigger reallocation
    planner.allocate_dynamic("dyn", 200, (20,), "float32")
    assert planner.dynamic_sizes["dyn"] == 200


def test_cuda_arena_reallocate_dynamic() -> None:
    """Tests the cuda arena reallocate dynamic functionality."""
    from onnx9000.backends.memory.cuda_arena import CUDAMemoryPlanner

    with patch("onnx9000.backends.memory.cuda_arena.is_cuda_available", return_value=True):
        with patch("onnx9000.backends.memory.cuda_arena._cuda_lib") as mock_lib:
            mock_lib.cuMemAlloc.return_value = 0
            mock_lib.cuMemFree.return_value = 0
            planner = CUDAMemoryPlanner()
            planner.allocate_dynamic("dyn", 100, (10,), "float32")
            # trigger reallocation
            planner.allocate_dynamic("dyn", 200, (20,), "float32")
            assert mock_lib.cuMemFree.called
