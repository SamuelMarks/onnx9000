import gc
from onnx9000.backends.cpu.ops import OP_REGISTRY
from onnx9000.backends.memory.cpu_arena import CPUMemoryPlanner
from onnx9000.backends.memory.metal_arena import MetalMemoryPlanner


def test_cpu_memory_planner():
    planner = CPUMemoryPlanner()
    planner.allocate_static("A", 128, (2, 2), "float32")
    planner.build_arena()
    assert planner.arena_mmap is not None
    assert len(planner.arena_mmap) >= 128
    ptr = planner.get_tensor_ptr("A")
    assert ptr is not None
    view = memoryview(b"A" * 128)
    planner.set_tensor("A", view, (2, 2), "float32")
    res = planner.get_host_tensor("A")
    assert len(res) == 128
    planner.set_tensor("B", memoryview(b"B"), (1,), "float32")
    assert planner.get_host_tensor("B").tobytes() == b"B"


def test_metal_memory_planner():
    planner = MetalMemoryPlanner()
    planner.allocate_static("A", 128, (2, 2), "float32")
    planner.build_arena()
    assert planner.arena_ptr is not None
    ptr = planner.get_tensor_ptr("A")
    assert ptr is not None
    view = memoryview(b"A" * 128)
    planner.set_tensor("A", view, (2, 2), "float32")
    res = planner.get_host_tensor("A")
    assert len(res) == 128
    planner.set_tensor("B", memoryview(b"B"), (1,), "float32")
    assert planner.get_host_tensor("B").tobytes() == b"B"


def test_cpu_ops_registry():
    import numpy as np

    inputs = [np.array([1, 2]), np.array([3, 4])]
    add_fn = OP_REGISTRY.get("Add")
    if add_fn:
        res = add_fn(inputs, {})
        assert np.array_equal(res[0], np.array([4, 6]))
    relu_fn = OP_REGISTRY.get("Relu")
    if relu_fn:
        res = relu_fn([np.array([-1, 1])], {})
        assert np.array_equal(res[0], np.array([0, 1]))
    sub_fn = OP_REGISTRY.get("Sub")
    if sub_fn:
        res = sub_fn(inputs, {})
        assert np.array_equal(res[0], np.array([-2, -2]))


def test_cpu_executor():
    import numpy as np
    from onnx9000.backends.cpu.executor import CPUExecutionProvider
    from onnx9000.core.dtypes import DType
    from onnx9000.core.execution import ExecutionContext, SessionOptions
    from onnx9000.core.ir import Graph, Node, Tensor

    ep = CPUExecutionProvider({})
    g = Graph("test")
    g.inputs.append(Tensor("A", (2,), DType.FLOAT32))
    g.outputs.append(Tensor("B", (2,), DType.FLOAT32))
    g.add_node(Node("Add", inputs=["A", "A"], outputs=["B"], attributes={}))
    assert ep.get_supported_nodes(g) == ["Add"]
    ep.allocate_tensors([])
    ctx = ExecutionContext(SessionOptions())
    t = Tensor("A", (2,), DType.FLOAT32, data=np.array([1, 2], dtype=np.float32))
    res = ep.execute(g, ctx, {"A": t})
    assert "B" in res
    assert np.array_equal(res["B"].data, np.array([2, 4], dtype=np.float32))
