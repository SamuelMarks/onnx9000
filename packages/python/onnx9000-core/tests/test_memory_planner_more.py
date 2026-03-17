"""Tests the memory planner more module functionality."""

import pytest
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.core.memory_planner import ArenaSimulator, MemoryBlock, simulate_memory_plan


def test_memory_block_repr():
    """Tests the memory block repr functionality."""
    b = MemoryBlock(0, 100)
    assert repr(b) == "Block(offset=0, size=100, free=True, tensor=None)"


def test_arena_align_exact():
    """Tests the arena align exact functionality."""
    arena = ArenaSimulator(alignment=256)
    assert arena._align(256) == 256


def test_allocate_first_fit_reuse():
    """Tests the allocate first fit reuse functionality."""
    arena = ArenaSimulator(alignment=256)
    arena.allocate_first_fit("t1", 200)  # uses 256
    arena.allocate_first_fit("t2", 300)  # uses 512
    arena.free("t1")
    # reuse t1 block
    arena.allocate_first_fit("t3", 100)
    assert arena.tensor_offsets["t3"] == 0


def test_allocate_best_fit():
    """Tests the allocate best fit functionality."""
    arena = ArenaSimulator(alignment=256)
    arena.allocate_best_fit("t1", 200)  # uses 256
    arena.allocate_best_fit("t2", 600)  # uses 768
    arena.allocate_best_fit("t3", 200)  # uses 256
    arena.free("t1")
    arena.free("t3")
    # now we have a 256 block at 0 and a 256 block at 1024
    # best fit for 100 should pick the first one
    arena.allocate_best_fit("t4", 100)
    assert arena.tensor_offsets["t4"] == 0
    # or just normal allocation
    arena.allocate_best_fit("t5", 1000)


def test_free_merge():
    """Tests the free merge functionality."""
    arena = ArenaSimulator(alignment=256)
    arena.allocate_first_fit("t1", 200)  # uses 256
    arena.allocate_first_fit("t2", 200)  # uses 256
    arena.allocate_first_fit("t3", 200)  # uses 256
    arena.free("t1")
    arena.free("t2")  # should merge with t1
    assert len(arena.blocks) == 2  # one big free, one used


def test_calculate_fragmentation():
    """Tests the calculate fragmentation functionality."""
    arena = ArenaSimulator(alignment=256)
    assert arena.calculate_fragmentation() == 0.0
    arena.allocate_first_fit("t1", 200)
    arena.allocate_first_fit("t2", 200)
    arena.free("t1")
    frag = arena.calculate_fragmentation()
    assert frag > 0


def test_simulate_best_fit():
    """Tests the simulate best fit functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("in1", [10, 10], "float32"))
    g.inputs.append("in1")
    g.add_node(Node("Identity", ["in1"], ["mid1"]))
    g.add_tensor(Tensor("mid1", [10, 10], "float32"))
    g.add_node(Node("Relu", ["mid1"], ["out1"]))
    g.add_tensor(Tensor("out1", [10, 10], "float32"))

    # fake the input already in lifetimes to hit line 135
    g.add_node(Node("Identity", ["in1"], ["in1"]))

    plan = simulate_memory_plan(g, strategy="best_fit")
    assert plan.peak_memory > 0


def test_block_splitting():
    """Tests the block splitting functionality."""
    arena = ArenaSimulator(alignment=256)
    arena.allocate_first_fit("t1", 1024)
    arena.free("t1")
    # Allocate something smaller to split the block using first_fit
    arena.allocate_first_fit("t2", 512)
    assert len(arena.blocks) == 2

    arena2 = ArenaSimulator(alignment=256)
    arena2.allocate_best_fit("t1", 1024)
    arena2.free("t1")
    # Allocate something smaller to split the block using best_fit
    arena2.allocate_best_fit("t2", 512)
    assert len(arena2.blocks) == 2


def test_simulate_in_place_and_view():
    """Tests the simulate in place and view functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("in1", [10, 10], "float32"))
    g.inputs.append("in1")
    g.add_node(Node("Identity", ["in1"], ["mid1"]))
    g.add_tensor(Tensor("mid1", [10, 10], "float32"))
    g.add_node(Node("Relu", ["mid1"], ["out1"]))
    g.add_tensor(Tensor("out1", [10, 10], "float32"))

    g.add_node(Node("Reshape", ["out1"], ["out2"]))
    g.add_tensor(Tensor("out2", [100], "float32"))

    plan = simulate_memory_plan(g, strategy="first_fit")

    # check if offsets match for inplace and view
    assert plan.tensor_offsets.get("out1") == plan.tensor_offsets.get("mid1")
    assert plan.tensor_offsets.get("out2") == plan.tensor_offsets.get("out1")


def test_simulate_empty_input():
    """Tests the simulate empty input functionality."""
    g = Graph("g")
    g.inputs.append("unused")
    simulate_memory_plan(g)
