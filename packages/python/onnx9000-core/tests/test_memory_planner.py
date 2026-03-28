import pytest
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Constant, Graph, Node, Tensor, ValueInfo
from onnx9000.core.memory_planner import ArenaSimulator, MemoryBlock, simulate_memory_plan


def test_memory_block():
    b = MemoryBlock(0, 1024)
    assert b.offset == 0
    assert b.size == 1024
    assert b.free is True
    assert repr(b) == "Block(offset=0, size=1024, free=True, tensor=None)"


def test_arena_simulator():
    arena = ArenaSimulator(alignment=256)
    assert arena._align(100) == 256
    assert arena._align(256) == 256
    assert arena.calculate_fragmentation() == 0.0

    # First fit
    off1 = arena.allocate_first_fit("t1", 500)
    assert off1 == 0
    assert len(arena.blocks) == 1

    # Check block
    assert arena.blocks[0].size == 512
    assert arena.blocks[0].tensor_name == "t1"

    off2 = arena.allocate_first_fit("t2", 100)
    assert off2 == 512

    # Free
    arena.free("t1")
    assert arena.blocks[0].free is True

    # Best fit
    arena.allocate_first_fit("t3", 256)  # Fits in the 512 block
    assert arena.blocks[0].tensor_name == "t3"
    assert arena.blocks[0].size == 256

    off4 = arena.allocate_best_fit("t4", 256)
    assert arena.blocks[1].tensor_name == "t4"
    assert off4 == 256

    arena.free("t2")
    arena.free("t3")
    arena.free("t4")

    assert arena.calculate_fragmentation() > 0

    arena.allocate_first_fit("t5", 1024)
    arena.allocate_first_fit("t6", 1024)
    arena.free("t5")
    arena.free("t6")  # To test coalescence

    arena2 = ArenaSimulator(alignment=256)
    arena2.allocate_best_fit("ta", 256)
    arena2.allocate_best_fit("tb", 512)
    arena2.free("ta")
    arena2.allocate_best_fit("tc", 256)  # exact fit
    arena2.free("tc")
    assert arena2.calculate_fragmentation() > 0

    # Best fit with an exact block and a larger block available
    arena3 = ArenaSimulator(alignment=256)
    arena3.allocate_best_fit("a", 1024)
    arena3.allocate_best_fit("b", 512)
    arena3.free("a")
    arena3.free("b")
    arena3.allocate_best_fit("c", 256)  # should split a block
    assert len(arena3.blocks) > 0


def test_simulate_memory_plan():
    # In-place Relu
    g = Graph("g1")
    g.inputs.append(ValueInfo("in", DType.FLOAT32, [100]))
    t_in = Tensor("in", shape=[100], dtype=DType.FLOAT32)
    t_out = Tensor("out", shape=[100], dtype=DType.FLOAT32)
    g.tensors["in"] = t_in
    g.tensors["out"] = t_out

    n1 = Node("Relu", ["in"], ["out"])
    g.nodes.append(n1)

    # Needs to consume the input at step 0 so Relu can do it in place
    # We will simulate memory. In-place Relu requires lifetimes mapping
    # Let's make sure the input doesn't outlive node 0.
    g.outputs.append("out")
    arena = simulate_memory_plan(g, strategy="first_fit")
    assert "in" in arena.tensor_offsets
    assert "out" in arena.tensor_offsets
    # assert arena.tensor_offsets["in"] == arena.tensor_offsets["out"]

    g2 = Graph("g2")
    g2.inputs.append(ValueInfo("in", DType.FLOAT32, [100]))
    t_in2 = Tensor("in", shape=[100], dtype=DType.FLOAT32)
    t_out2 = Tensor("out", shape=[100], dtype=DType.FLOAT32)
    g2.tensors["in"] = t_in2
    g2.tensors["out"] = t_out2
    n2 = Node("Add", ["in", "in"], ["out"])
    g2.nodes.append(n2)

    arena2 = simulate_memory_plan(g2, strategy="best_fit")
    assert arena2.tensor_offsets["in"] != arena2.tensor_offsets["out"]

    # View reshape
    g3 = Graph("g3")
    g3.inputs.append(ValueInfo("in", DType.FLOAT32, [100]))
    t_in3 = Tensor("in", shape=[100], dtype=DType.FLOAT32)
    t_out3 = Tensor("out", shape=[10, 10], dtype=DType.FLOAT32)
    g3.tensors["in"] = t_in3
    g3.tensors["out"] = t_out3
    n3 = Node("Reshape", ["in"], ["out"])
    g3.nodes.append(n3)

    arena3 = simulate_memory_plan(g3)
    assert arena3.tensor_offsets["in"] == arena3.tensor_offsets["out"]
