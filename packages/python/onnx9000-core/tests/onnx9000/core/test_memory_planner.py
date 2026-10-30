"""Unit tests for static memory arena planner in `onnx9000.core.memory_planner`."""

from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Variable
from onnx9000.core.memory_planner import ArenaSimulator, MemoryBlock, simulate_memory_plan


def test_memory_planner() -> None:
    """Verify MemoryBlock, ArenaSimulator, and simulate_memory_plan."""
    blk = MemoryBlock(0, 256)
    assert "offset=0" in repr(blk)

    sim = ArenaSimulator(alignment=256)
    off1 = sim.allocate_first_fit("t1", 100)
    off2 = sim.allocate_best_fit("t2", 200)
    assert off1 == 0
    assert off2 == 256
    sim.free("t1")
    assert sim.calculate_fragmentation() > 0

    g = Graph("mem_g")
    g.inputs.append("x")
    g.outputs.append("y")
    g.add_tensor(Variable("x", (4, 4), DType.FLOAT32))
    g.add_tensor(Variable("y", (4, 4), DType.FLOAT32))
    g.add_node(Node("Relu", ["x"], ["y"], name="r0"))

    arena = simulate_memory_plan(g)
    assert arena.peak_memory >= 256
