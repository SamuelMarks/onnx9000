"""Comprehensive unit tests for onnx9000.arena and onnx9000_arena."""

import onnx9000_arena
import pytest
from onnx9000.arena import ArenaAllocationError, ArenaAllocator, plan
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import DynamicDim, Graph, Node, Tensor


def test_plan_valid() -> None:
    """Verify that plan returns the expected summary string for a valid model identifier."""
    assert plan("test_model") == "[Arena] planner processed test_model"
    assert onnx9000_arena.plan("m2") == "[Arena] planner processed m2"


def test_plan_invalid() -> None:
    """Verify that plan raises ArenaAllocationError when given an empty model string."""
    with pytest.raises(ArenaAllocationError, match="Invalid model string"):
        plan("")


def test_allocator_validation() -> None:
    """Verify that ArenaAllocator validates positive page_size and alignment arguments."""
    with pytest.raises(ArenaAllocationError, match="page_size must be positive"):
        ArenaAllocator(page_size=0)
    with pytest.raises(ArenaAllocationError, match="alignment must be positive"):
        ArenaAllocator(alignment=-4)


def test_allocator_dtype_and_shapes() -> None:
    """Verify element counting and byte-width resolution across static, symbolic, and string dtypes."""
    allocator = ArenaAllocator(page_size=65536, alignment=16)
    assert allocator.get_wasm_pages() == 0

    assert ArenaAllocator._dtype_itemsize(DType.FLOAT32) == 4
    assert ArenaAllocator._dtype_itemsize("float16") == 2
    assert ArenaAllocator._dtype_itemsize("unknown_type") == 4
    assert ArenaAllocator._dtype_itemsize(None) == 4

    assert ArenaAllocator._num_elements(None) == 1
    assert ArenaAllocator._num_elements(()) == 1
    assert ArenaAllocator._num_elements((2, DynamicDim("N"), DynamicDim(3), -1)) == 6


def test_allocator_allocate_graph() -> None:
    """Verify full graph allocation including registered tensors, node Tensor outputs, and unindexed outputs."""
    graph = Graph("arena_test")
    t_a = Tensor("A", shape=(4, 4), dtype=DType.FLOAT32)  # 16 * 4 = 64 bytes
    t_b = Tensor("B", shape=(3,), dtype=DType.INT8)  # 3 * 1 = 3 -> aligned to 16 bytes
    t_c = Tensor("C", shape=(2, 2), dtype=DType.FLOAT64)  # 4 * 8 = 32 bytes
    graph.add_tensor(t_a)
    graph.add_tensor(t_b)

    node1 = Node("Add", inputs=["A", "B"], outputs=[t_c, "D_unregistered", "A"])
    graph.add_node(node1)
    graph.add_tensor(t_c)

    node2 = Node(
        "Relu", inputs=["C"], outputs=[Tensor("E_tensor_only", shape=(1,), dtype=DType.FLOAT32)]
    )
    graph.add_node(node2)

    allocator = ArenaAllocator(page_size=64, alignment=16)
    offsets, total_bytes = allocator.allocate(graph)

    assert offsets["A"] == 0
    assert offsets["B"] == 64
    assert offsets["C"] == 80
    assert offsets["D_unregistered"] == 112
    assert offsets["E_tensor_only"] == 128
    assert total_bytes == 144

    # Re-allocating an already allocated tensor returns the existing offset
    assert allocator.allocate_tensor(t_a) == 0
    # 144 bytes with 64-byte pages requires 3 pages
    assert allocator.get_wasm_pages() == 3
