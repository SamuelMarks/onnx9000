import numpy as np
import pytest
from onnx9000.backends.cpu.memory import MemoryPlanner


def test_memory_planner():
    planner = MemoryPlanner()
    planner.allocate_static("a", 16, (4,), np.dtype("float32"))
    planner.allocate_static("b", 8, (2,), np.dtype("float32"))
    planner.build_arena()
    assert planner.arena.nbytes == 24
    a_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    planner.set_tensor("a", a_data)
    b_data = np.array([5.0, 6.0], dtype=np.float32)
    planner.set_tensor("b", b_data)
    out_a = planner.get_tensor("a")
    np.testing.assert_array_equal(out_a, a_data)
    out_b = planner.get_tensor("b")
    np.testing.assert_array_equal(out_b, b_data)


def test_memory_planner_fallback():
    planner = MemoryPlanner()
    planner.allocate_static("a", 16, (4,), np.dtype("float32"))
    planner.build_arena()
    a_data_large = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    planner.set_tensor("a", a_data_large)
    out_a = planner.get_tensor("a")
    np.testing.assert_array_equal(out_a, a_data_large)


def test_memory_planner_unallocated():
    planner = MemoryPlanner()
    planner.build_arena()
    c_data = np.array([1, 2, 3], dtype=np.int32)
    planner.set_tensor("c", c_data)
    out_c = planner.get_tensor("c")
    np.testing.assert_array_equal(out_c, c_data)


def test_memory_planner_out_of_bounds():
    planner = MemoryPlanner()
    planner.offsets["d"] = 10, 20
    planner.tensors_shape_dtype["d"] = (5,), np.dtype("float32")
    planner.arena = np.empty((10,), dtype=np.uint8)
    with pytest.raises(RuntimeError):
        planner.get_tensor("d")


def test_memory_planner_reallocate():
    planner = MemoryPlanner()
    planner.allocate_static("a", 4, (1,), np.dtype("float32"))
    planner.build_arena()
    planner._reallocate_arena(8)
    assert planner.arena.nbytes == 8


def test_memory_planner_shape_mismatch():
    planner = MemoryPlanner()
    planner.allocate_static("a", 16, (2, 2), np.dtype("float32"))
    planner.build_arena()
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    planner.set_tensor("a", data)
    assert "a" in planner.dynamic_tensors
