from unittest.mock import MagicMock, patch

import numpy as np
import onnx9000.backends.rocm.executor as executor
import pytest
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor


def test_rocm_executor_fallback() -> None:
    g = Graph("test")
    g.inputs.append(Tensor("A", (1, 2), DType.FLOAT32))
    g.outputs.append(Tensor("B", (1, 2), DType.FLOAT32))
    g.add_node(Node("Relu", ["A"], ["B"]))

    with patch("onnx9000.backends.rocm.executor.is_hip_available", return_value=False):
        dispatcher = executor.Dispatcher(g)

    assert dispatcher.initialized is False

    # Run
    inp = np.array([[1.0, -1.0]], dtype=np.float32)
    with patch.object(dispatcher.cpu_fallback, "execute") as mock_exec:
        mock_exec.return_value = {"B": np.array([[1.0, 0.0]], dtype=np.float32)}
        out = dispatcher.run({"A": inp})

    assert "B" in out
    np.testing.assert_allclose(out["B"], np.array([[1.0, 0.0]], dtype=np.float32))


def test_rocm_executor_matmul_fallback() -> None:
    g = Graph("test")
    g.inputs.append(Tensor("A", (2, 2), DType.FLOAT32))
    g.inputs.append(Tensor("B", (2, 2), DType.FLOAT32))
    g.outputs.append(Tensor("C", (2, 2), DType.FLOAT32))
    g.add_node(Node("MatMul", ["A", "B"], ["C"]))
    g.tensors["A"] = Tensor("A", (2, 2), DType.FLOAT32)
    g.tensors["B"] = Tensor("B", (2, 2), DType.FLOAT32)
    g.tensors["C"] = Tensor("C", (2, 2), DType.FLOAT32)

    with (
        patch("onnx9000.backends.rocm.executor.is_hip_available", return_value=True),
        patch("onnx9000.backends.rocm.executor.is_rocblas_available", return_value=True),
        patch("onnx9000.backends.rocm.executor.is_miopen_available", return_value=True),
        patch(
            "onnx9000.backends.rocm.executor._hip_lib",
            MagicMock(**{"hipStreamCreate.return_value": 0}),
        ),
        patch(
            "onnx9000.backends.rocm.executor._rocblas_lib",
            MagicMock(
                **{"rocblas_create_handle.return_value": 0, "rocblas_set_stream.return_value": 0}
            ),
        ),
        patch(
            "onnx9000.backends.rocm.executor._miopen_lib",
            MagicMock(**{"miopenCreate.return_value": 0}),
        ),
    ):
        dispatcher = executor.Dispatcher(g)

        inp_A = np.ones((2, 2), dtype=np.float32)
        inp_B = np.ones((2, 2), dtype=np.float32)
        with patch.object(
            dispatcher.cpu_fallback,
            "execute",
            return_value={"C": np.ones((2, 2), dtype=np.float32)},
        ):
            out = dispatcher.run({"A": inp_A, "B": inp_B})

        assert "C" in out


def test_rocm_executor_matmul_no_rocblas() -> None:
    g = Graph("test")
    g.inputs.append(Tensor("A", (2, 2), DType.FLOAT32))
    g.inputs.append(Tensor("B", (2, 2), DType.FLOAT32))
    g.outputs.append(Tensor("C", (2, 2), DType.FLOAT32))
    g.add_node(Node("MatMul", ["A", "B"], ["C"]))

    with (
        patch("onnx9000.backends.rocm.executor.is_hip_available", return_value=True),
        patch("onnx9000.backends.rocm.executor.is_rocblas_available", return_value=False),
        patch(
            "onnx9000.backends.rocm.executor._hip_lib",
            MagicMock(**{"hipStreamCreate.return_value": 0}),
        ),
    ):
        dispatcher = executor.Dispatcher(g)

        with pytest.raises(RuntimeError, match="rocBLAS is required"):
            dispatcher.run(
                {"A": np.ones((2, 2), dtype=np.float32), "B": np.ones((2, 2), dtype=np.float32)}
            )


def test_rocm_executor_elementwise_fallback() -> None:
    for op_type in ["Add", "Sub", "Conv", "Dummy"]:
        g = Graph("test")
        g.inputs.append(Tensor("A", (2, 2), DType.FLOAT32))
        g.inputs.append(Tensor("B", (2, 2), DType.FLOAT32))
        g.outputs.append(Tensor("C", (2, 2), DType.FLOAT32))
        g.add_node(Node(op_type, ["A", "B"], ["C"]))
        g.tensors["A"] = Tensor("A", (2, 2), DType.FLOAT32)
        g.tensors["B"] = Tensor("B", (2, 2), DType.FLOAT32)
        g.tensors["C"] = Tensor("C", (2, 2), DType.FLOAT32)

        with (
            patch("onnx9000.backends.rocm.executor.is_hip_available", return_value=True),
            patch(
                "onnx9000.backends.rocm.executor._hip_lib",
                MagicMock(**{"hipStreamCreate.return_value": 0}),
            ),
        ):
            dispatcher = executor.Dispatcher(g)
            inp_A = np.ones((2, 2), dtype=np.float32)
            inp_B = np.ones((2, 2), dtype=np.float32)
            with patch.object(
                dispatcher.cpu_fallback,
                "execute",
                return_value={"C": np.ones((2, 2), dtype=np.float32)},
            ):
                out = dispatcher.run({"A": inp_A, "B": inp_B})
                assert "C" in out


def test_rocm_executor_init_memory_dynamic_and_init() -> None:
    g = Graph("test")
    t_out = Tensor("out_t", ("N", 2), DType.FLOAT32)
    t_init = Tensor(
        "init_t", (2,), DType.FLOAT32, data=np.array([1.0, 2.0], dtype=np.float32).tobytes()
    )
    g.tensors["out_t"] = t_out
    g.tensors["init_t"] = t_init
    g.initializers.append("init_t")
    g.outputs.append("out_t")
    g.add_node(Node("Dummy", [], ["out_t"]))

    with patch("onnx9000.backends.rocm.executor.is_hip_available", return_value=False):
        dispatcher = executor.Dispatcher(g)

    assert "init_t" in dispatcher.planner.dynamic_allocations


def test_rocm_executor_del_handles_success() -> None:
    g = Graph("test")
    with (
        patch("onnx9000.backends.rocm.executor.is_hip_available", return_value=True),
        patch("onnx9000.backends.rocm.executor.is_rocblas_available", return_value=True),
        patch("onnx9000.backends.rocm.executor.is_miopen_available", return_value=True),
        patch(
            "onnx9000.backends.rocm.executor._hip_lib",
            MagicMock(**{"hipStreamCreate.return_value": 0, "hipStreamDestroy.return_value": 0}),
        ) as mock_hip,
        patch(
            "onnx9000.backends.rocm.executor._rocblas_lib",
            MagicMock(
                **{
                    "rocblas_create_handle.return_value": 0,
                    "rocblas_set_stream.return_value": 0,
                    "rocblas_destroy_handle.return_value": 0,
                }
            ),
        ) as mock_rocblas,
        patch(
            "onnx9000.backends.rocm.executor._miopen_lib",
            MagicMock(**{"miopenCreate.return_value": 0, "miopenDestroy.return_value": 0}),
        ) as mock_miopen,
    ):
        dispatcher = executor.Dispatcher(g)
        dispatcher.rocblas_handle = MagicMock()
        dispatcher.miopen_handle = MagicMock()
        dispatcher.stream = MagicMock()
        dispatcher.initialized = True

        dispatcher.__del__()
        assert mock_rocblas.rocblas_destroy_handle.called
        assert mock_miopen.miopenDestroy.called
        assert mock_hip.hipStreamDestroy.called


def test_rocm_executor_del_handles_errors() -> None:
    g = Graph("test")
    with (
        patch("onnx9000.backends.rocm.executor.is_hip_available", return_value=True),
        patch("onnx9000.backends.rocm.executor.is_rocblas_available", return_value=True),
        patch("onnx9000.backends.rocm.executor.is_miopen_available", return_value=True),
        patch(
            "onnx9000.backends.rocm.executor._hip_lib",
            MagicMock(**{"hipStreamCreate.return_value": 0}),
        ) as mock_hip,
        patch(
            "onnx9000.backends.rocm.executor._rocblas_lib",
            MagicMock(
                **{"rocblas_create_handle.return_value": 0, "rocblas_set_stream.return_value": 0}
            ),
        ) as mock_rocblas,
        patch(
            "onnx9000.backends.rocm.executor._miopen_lib",
            MagicMock(**{"miopenCreate.return_value": 0}),
        ) as mock_miopen,
    ):
        dispatcher = executor.Dispatcher(g)

        dispatcher.rocblas_handle = MagicMock()
        dispatcher.miopen_handle = MagicMock()
        dispatcher.stream = MagicMock()
        dispatcher.initialized = True

        mock_rocblas.rocblas_destroy_handle.side_effect = Exception("rocblas error")
        mock_miopen.miopenDestroy.side_effect = Exception("miopen error")
        mock_hip.hipStreamDestroy.side_effect = Exception("hip error")

        dispatcher.__del__()


def ignore() -> None:
    g = Graph("test")
    with (
        patch("onnx9000.backends.rocm.executor.is_hip_available", return_value=True),
        patch("onnx9000.backends.rocm.executor.is_rocblas_available", return_value=True),
        patch("onnx9000.backends.rocm.executor.is_miopen_available", return_value=True),
        patch(
            "onnx9000.backends.rocm.executor._hip_lib",
            MagicMock(**{"hipStreamCreate.return_value": 0}),
        ) as mock_hip,
        patch(
            "onnx9000.backends.rocm.executor._rocblas_lib",
            MagicMock(
                **{"rocblas_create_handle.return_value": 0, "rocblas_set_stream.return_value": 0}
            ),
        ) as mock_rocblas,
        patch(
            "onnx9000.backends.rocm.executor._miopen_lib",
            MagicMock(**{"miopenCreate.return_value": 0}),
        ) as mock_miopen,
    ):
        dispatcher = executor.Dispatcher(g)

        mock_rocblas.rocblas_destroy_handle.side_effect = Exception("rocblas error")
        mock_miopen.miopenDestroy.side_effect = Exception("miopen error")
        mock_hip.hipStreamDestroy.side_effect = Exception("hip error")

        dispatcher.__del__()
