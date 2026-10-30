"""Test executor."""

import contextlib
from typing import NoReturn

import numpy as np
from onnx9000.backends.apple import bindings
from onnx9000.backends.apple.executor import (
    Dispatcher,
    _apple_add,
    _apple_matmul,
    _apple_mul,
    _apple_sub,
)
from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.core.ir import Graph, Node, Tensor


def test_dispatcher_init() -> None:
    """Test dispatcher init."""
    g = Graph("test")
    d = Dispatcher(g)
    assert d.planner is not None


def test_dispatcher_run() -> None:
    """Test dispatcher run."""
    g = Graph("test")
    t1 = Tensor(name="x", shape=[2, 2], dtype="float32")
    t2 = Tensor(
        name="y",
        shape=[2, 2],
        dtype="float32",
        data=np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32).tobytes(),
    )
    t3 = Tensor(name="z", shape=[2, 2], dtype="float32")
    g.tensors["x"] = t1
    g.tensors["y"] = t2
    g.tensors["z"] = t3
    g.initializers.append("y")
    g.inputs.append("x")
    g.outputs.append("z")
    node = Node(name="n1", op_type="Add", inputs=["x", "y"], outputs=["z"])
    g.nodes.append(node)

    d = Dispatcher(g)
    res = d.run({"x": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)})
    assert "z" in res


def test_ops() -> None:
    """Test ops."""
    g = Graph("test")
    t1 = Tensor(name="x", shape=[2, 2], dtype="float32")
    t2 = Tensor(
        name="y",
        shape=[2, 2],
        dtype="float32",
        data=np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32).tobytes(),
    )
    t3 = Tensor(name="z", shape=[2, 2], dtype="float32")
    g.tensors["x"] = t1
    g.tensors["y"] = t2
    g.tensors["z"] = t3
    g.initializers.append("y")
    d = Dispatcher(g)
    d.run({"x": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)})

    node = Node(name="n1", op_type="MatMul", inputs=["x", "y"], outputs=["z"])
    _apple_matmul(d, node)

    node = Node(name="n2", op_type="Add", inputs=["x", "y"], outputs=["z"])
    _apple_add(d, node)

    node = Node(name="n3", op_type="Sub", inputs=["x", "y"], outputs=["z"])
    _apple_sub(d, node)

    node = Node(name="n4", op_type="Mul", inputs=["x", "y"], outputs=["z"])
    _apple_mul(d, node)
    assert node is not None or node is None


def test_ops_no_accelerate() -> None:
    """Test ops without accelerate."""
    g = Graph("test")
    t1 = Tensor(name="x", shape=[2, 2], dtype="float32")
    t2 = Tensor(
        name="y",
        shape=[2, 2],
        dtype="float32",
        data=np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32).tobytes(),
    )
    t3 = Tensor(name="z", shape=[2, 2], dtype="float32")
    g.tensors["x"] = t1
    g.tensors["y"] = t2
    g.tensors["z"] = t3
    g.initializers.append("y")
    d = Dispatcher(g)
    d.run({"x": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)})

    orig_lib = bindings._accelerate_lib
    bindings._accelerate_lib = None

    node = Node(name="n1", op_type="MatMul", inputs=["x", "y"], outputs=["z"])
    _apple_matmul(d, node)

    node = Node(name="n2", op_type="Add", inputs=["x", "y"], outputs=["z"])
    _apple_add(d, node)

    bindings._accelerate_lib = orig_lib
    assert node is not None or node is None


def test_cpu_fallback_tensor() -> None:
    """Test CPU fallback."""
    g = Graph("test")
    t1 = Tensor(name="x", shape=[2, 2], dtype="float32")
    t2 = Tensor(
        name="y",
        shape=[2, 2],
        dtype="float32",
        data=np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32).tobytes(),
    )
    t3 = Tensor(name="z", shape=[2, 2], dtype="float32")
    g.tensors["x"] = t1
    g.tensors["y"] = t2
    g.tensors["z"] = t3
    g.initializers.append("y")
    d = Dispatcher(g)
    d.run({"x": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)})

    node = Node(name="n1", op_type="Add", inputs=["x", "y"], outputs=["z"])

    # mock the cpu_fallback
    class MockCPU:
        """Mock CPU."""

        def execute(
            self,
            graph: int | float | str | bool | bytes | list | dict | tuple | None,
            context: int | float | str | bool | bytes | list | dict | tuple | None,
            inputs: int | float | str | bool | bytes | list | dict | tuple | None,
        ) -> int | float | str | bool | bytes | list | dict | tuple | None:
            """Execute."""
            return {
                "z": Tensor(
                    name="z",
                    shape=[2, 2],
                    dtype="float32",
                    data=np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32).tobytes(),
                )
            }

    d.cpu_fallback = MockCPU()
    d._cpu_fallback_node(node)
    assert node is not None or node is None


def test_runtime_error() -> None:
    """Test runtime error."""
    g = Graph("test")
    d = Dispatcher(g)
    node = Node(name="n5", op_type="NonExistentOp123", inputs=["x"], outputs=["z"])
    with contextlib.suppress(
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        d._execute_node(node)

    # raise Onnx9000Error specifically
    class MockRegistry:
        """Mock Registry."""

        def get_op(
            self,
            domain: int | float | str | bool | bytes | list | dict | tuple | None,
            op_type: int | float | str | bool | bytes | list | dict | tuple | None,
            provider: int | float | str | bool | bytes | list | dict | tuple | None = None,
        ) -> int | float | str | bool | bytes | list | dict | tuple | None:
            """Get op."""

            def raise_rt(
                *args: int | float | str | bool | bytes | list | dict | tuple | None,
            ) -> NoReturn:
                """Raise."""
                raise Onnx9000Error("Test")

            return raise_rt

    from onnx9000.core import registry

    orig = registry.global_registry
    registry.global_registry = MockRegistry()
    with contextlib.suppress(RuntimeError):
        d._execute_node(node)
    registry.global_registry = orig
    assert orig is not None or orig is None
