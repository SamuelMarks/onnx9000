import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.core.dtypes import DType
import onnx9000.backends.apple.executor as executor


def test_apple_executor_fallback():
    g = Graph("test")
    g.inputs.append(Tensor("A", (1, 2), DType.FLOAT32))
    g.outputs.append(Tensor("B", (1, 2), DType.FLOAT32))
    g.add_node(Node("Relu", ["A"], ["B"]))

    with patch("onnx9000.backends.apple.executor.mtl_create_system_default_device") as mock_device:
        mock_device.return_value = "fake_device"
        dispatcher = executor.Dispatcher(g)

    assert dispatcher.initialized is True

    # Run
    inp = np.array([[1.0, -1.0]], dtype=np.float32)
    with patch.object(dispatcher.cpu_fallback, "execute") as mock_exec:
        mock_exec.return_value = {"B": np.array([[1.0, 0.0]], dtype=np.float32)}
        out = dispatcher.run({"A": inp})

    assert "B" in out
    np.testing.assert_allclose(out["B"], np.array([[1.0, 0.0]], dtype=np.float32))


def test_apple_executor_matmul_accelerate():
    g = Graph("test")
    g.inputs.append(Tensor("A", (2, 2), DType.FLOAT32))
    g.inputs.append(Tensor("B", (2, 2), DType.FLOAT32))
    g.outputs.append(Tensor("C", (2, 2), DType.FLOAT32))
    g.add_node(Node("MatMul", ["A", "B"], ["C"]))

    with patch("onnx9000.backends.apple.executor.is_accelerate_available", return_value=True):
        with patch("onnx9000.backends.apple.executor._accelerate_lib") as mock_accel:
            dispatcher = executor.Dispatcher(g)

            inp_A = np.ones((2, 2), dtype=np.float32)
            inp_B = np.ones((2, 2), dtype=np.float32)
            out = dispatcher.run({"A": inp_A, "B": inp_B})

            assert mock_accel.cblas_sgemm.called
            assert "C" in out


def test_apple_executor_elementwise_accelerate():
    for op_type, func_name in [("Add", "vDSP_vadd"), ("Sub", "vDSP_vsub"), ("Mul", "vDSP_vmul")]:
        g = Graph("test")
        g.inputs.append(Tensor("A", (2, 2), DType.FLOAT32))
        g.inputs.append(Tensor("B", (2, 2), DType.FLOAT32))
        g.outputs.append(Tensor("C", (2, 2), DType.FLOAT32))
        g.add_node(Node(op_type, ["A", "B"], ["C"]))

        with patch("onnx9000.backends.apple.executor.is_accelerate_available", return_value=True):
            with patch("onnx9000.backends.apple.executor._accelerate_lib") as mock_accel:
                dispatcher = executor.Dispatcher(g)

                inp_A = np.ones((2, 2), dtype=np.float32)
                inp_B = np.ones((2, 2), dtype=np.float32)
                out = dispatcher.run({"A": inp_A, "B": inp_B})

                assert getattr(mock_accel, func_name).called
                assert "C" in out


def test_apple_executor_matmul_no_accelerate():
    g = Graph("test")
    g.inputs.append(Tensor("A", (2, 2), DType.FLOAT32))
    g.inputs.append(Tensor("B", (2, 2), DType.FLOAT32))
    g.outputs.append(Tensor("C", (2, 2), DType.FLOAT32))
    g.add_node(Node("MatMul", ["A", "B"], ["C"]))

    with patch("onnx9000.backends.apple.executor.is_accelerate_available", return_value=False):
        dispatcher = executor.Dispatcher(g)

        inp_A = np.ones((2, 2), dtype=np.float32)
        inp_B = np.ones((2, 2), dtype=np.float32)
        with patch.object(
            dispatcher.cpu_fallback,
            "execute",
            return_value={"C": np.ones((2, 2), dtype=np.float32)},
        ) as mock_exec:
            out = dispatcher.run({"A": inp_A, "B": inp_B})
            assert mock_exec.called


def test_apple_executor_elementwise_no_accelerate():
    g = Graph("test")
    g.inputs.append(Tensor("A", (2, 2), DType.FLOAT32))
    g.inputs.append(Tensor("B", (2, 2), DType.FLOAT32))
    g.outputs.append(Tensor("C", (2, 2), DType.FLOAT32))
    g.add_node(Node("Add", ["A", "B"], ["C"]))

    with patch("onnx9000.backends.apple.executor.is_accelerate_available", return_value=False):
        dispatcher = executor.Dispatcher(g)

        inp_A = np.ones((2, 2), dtype=np.float32)
        inp_B = np.ones((2, 2), dtype=np.float32)
        with patch.object(
            dispatcher.cpu_fallback,
            "execute",
            return_value={"C": np.ones((2, 2), dtype=np.float32)},
        ) as mock_exec:
            out = dispatcher.run({"A": inp_A, "B": inp_B})
            assert mock_exec.called


def test_apple_executor_init_memory():
    g = Graph("test")
    t_out = Tensor("out_t", (2,), DType.FLOAT32)
    t_init = Tensor(
        "init_t", (2,), DType.FLOAT32, data=np.array([1.0, 2.0], dtype=np.float32).tobytes()
    )
    g.tensors["out_t"] = t_out
    g.tensors["init_t"] = t_init
    g.initializers.append("init_t")
    g.add_node(Node("Dummy", [], ["out_t"]))

    with patch(
        "onnx9000.backends.apple.executor.mtl_create_system_default_device",
        return_value="fake_device",
    ):
        dispatcher = executor.Dispatcher(g)

    assert "out_t" in dispatcher.planner.offsets
    assert "init_t" in dispatcher.planner.dynamic_allocations
