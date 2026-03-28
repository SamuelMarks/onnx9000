"""Tests the runner module functionality."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from onnx9000.backends.cpu.executor import CPUExecutionProvider
from onnx9000.backends.testing.runner import ONNXBackendTestRunner
from onnx9000.core import onnx_pb2
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.core.serializer import save


def _create_tensor_pb(name: str, arr: np.ndarray, file_path: str) -> None:
    """Test the create tensor pb functionality."""
    tensor_proto = onnx_pb2.TensorProto()
    tensor_proto.name = name
    tensor_proto.dims.extend(arr.shape)
    if arr.dtype == np.float32:
        tensor_proto.data_type = onnx_pb2.TensorProto.FLOAT
        tensor_proto.raw_data = arr.tobytes()
    elif arr.dtype == np.int64:
        tensor_proto.data_type = onnx_pb2.TensorProto.INT64
        tensor_proto.raw_data = arr.tobytes()
    elif arr.dtype == bool:
        tensor_proto.data_type = onnx_pb2.TensorProto.BOOL
        tensor_proto.raw_data = arr.tobytes()
    with open(file_path, "wb") as f:
        f.write(tensor_proto.SerializeToString())


def test_onnx_backend_test_runner() -> None:
    """Tests the onnx backend test runner functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        g = Graph("test_add")
        g.inputs.append(Tensor("A", (1,), DType.FLOAT32))
        g.inputs.append(Tensor("B", (2,), DType.FLOAT32))
        g.outputs.append(Tensor("C", (1,), DType.FLOAT32))
        g.add_node(Node("Add", inputs=["A", "B"], outputs=["C"], attributes={}))
        save(g, base_dir / "model.onnx")
        ds_dir = base_dir / "test_data_set_0"
        ds_dir.mkdir()
        A_data = np.array([1.0, 2.0], dtype=np.float32)
        B_data = np.array([3.0, 4.0], dtype=np.float32)
        C_data = np.array([4.0, 6.0], dtype=np.float32)
        _create_tensor_pb("A", A_data, str(ds_dir / "input_0.pb"))
        _create_tensor_pb("B", B_data, str(ds_dir / "input_1.pb"))
        _create_tensor_pb("C", C_data, str(ds_dir / "output_0.pb"))
        runner = ONNXBackendTestRunner(providers=[CPUExecutionProvider({})])
        (passed, msg) = runner.run_node_test(base_dir)
        assert passed, f"Test failed with message: {msg}"


def test_onnx_backend_test_runner_mismatch() -> None:
    """Tests the onnx backend test runner mismatch functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        g = Graph("test_add")
        g.inputs.append(Tensor("A", (1,), DType.FLOAT32))
        g.inputs.append(Tensor("B", (2,), DType.FLOAT32))
        g.outputs.append(Tensor("C", (1,), DType.FLOAT32))
        g.add_node(Node("Add", inputs=["A", "B"], outputs=["C"], attributes={}))
        save(g, base_dir / "model.onnx")
        ds_dir = base_dir / "test_data_set_0"
        ds_dir.mkdir()
        A_data = np.array([1.0, 2.0], dtype=np.float32)
        B_data = np.array([3.0, 4.0], dtype=np.float32)
        C_data = np.array([99.0, 99.0], dtype=np.float32)
        _create_tensor_pb("A", A_data, str(ds_dir / "input_0.pb"))
        _create_tensor_pb("B", B_data, str(ds_dir / "input_1.pb"))
        _create_tensor_pb("C", C_data, str(ds_dir / "output_0.pb"))
        runner = ONNXBackendTestRunner(providers=[CPUExecutionProvider({})])
        (passed, msg) = runner.run_node_test(base_dir)
        assert not passed
        assert "Numerical tolerance mismatch" in msg


def test_runner_skip_list() -> None:
    """Tests the runner skip list functionality."""
    runner = ONNXBackendTestRunner(providers=[])
    runner.set_skip_list(["test_skip_this"])
    assert "test_skip_this" in runner.skip_lists


def test_runner_empty_tensor() -> None:
    """Tests the runner empty tensor functionality."""
    runner = ONNXBackendTestRunner(providers=[])
    arr = runner._convert_tensor_to_numpy(Tensor("E", (1,), DType.FLOAT32, data=None))
    assert arr.shape == (1,)


def test_runner_invalid_dtype() -> None:
    """Tests the runner invalid dtype functionality."""
    runner = ONNXBackendTestRunner(providers=[])
    with pytest.raises(TypeError):
        runner._convert_tensor_to_numpy(Tensor("E", (1,), DType.STRING, data=b"hi"))


def test_runner_missing_model() -> None:
    """Tests the runner missing model functionality."""
    runner = ONNXBackendTestRunner(providers=[])
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        (passed, msg) = runner.run_node_test(base_dir)
        assert not passed
        assert "Model file not found" in msg


def test_runner_no_datasets() -> None:
    """Tests the runner no datasets functionality."""
    runner = ONNXBackendTestRunner(providers=[])
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        save(Graph("test"), base_dir / "model.onnx")
        (passed, msg) = runner.run_node_test(base_dir)
        assert not passed
        assert "No test data sets found" in msg


def test_runner_missing_output() -> None:
    """Tests the runner missing output functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        g = Graph("test_add")
        g.inputs.append(Tensor("A", (1,), DType.FLOAT32))
        g.inputs.append(Tensor("B", (2,), DType.FLOAT32))
        g.outputs.append(Tensor("C", (1,), DType.FLOAT32))
        g.add_node(Node("Add", inputs=["A", "B"], outputs=["C"], attributes={}))
        save(g, base_dir / "model.onnx")
        ds_dir = base_dir / "test_data_set_0"
        ds_dir.mkdir()
        A_data = np.array([1.0, 2.0], dtype=np.float32)
        B_data = np.array([3.0, 4.0], dtype=np.float32)
        C_data = np.array([4.0, 6.0], dtype=np.float32)
        D_data = np.array([4.0, 6.0], dtype=np.float32)
        _create_tensor_pb("A", A_data, str(ds_dir / "input_0.pb"))
        _create_tensor_pb("B", B_data, str(ds_dir / "input_1.pb"))
        _create_tensor_pb("C", C_data, str(ds_dir / "output_0.pb"))
        _create_tensor_pb("D", D_data, str(ds_dir / "output_1.pb"))
        runner = ONNXBackendTestRunner(providers=[CPUExecutionProvider({})])
        (passed, msg) = runner.run_node_test(base_dir)
        assert not passed
        assert "Missing output tensor at index 1" in msg


def test_runner_missing_outputs() -> None:
    """Tests the runner missing outputs functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        g = Graph("test_add")
        g.inputs.append(Tensor("A", (1,), DType.BOOL))
        g.outputs.append(Tensor("C", (1,), DType.BOOL))
        g.add_node(Node("Add", inputs=["A", "B"], outputs=["C"], attributes={}))
        save(g, base_dir / "model.onnx")
        ds_dir = base_dir / "test_data_set_0"
        ds_dir.mkdir()
        _create_tensor_pb("A", np.array([1.0], dtype=np.float32), str(ds_dir / "input_0.pb"))
        runner = ONNXBackendTestRunner(providers=[CPUExecutionProvider({})])
        (passed, msg) = runner.run_node_test(base_dir)
        assert passed


def test_runner_extra_inputs() -> None:
    """Tests the runner extra inputs functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        g = Graph("test_add")
        g.inputs.append(Tensor("A", (1,), DType.FLOAT32))
        g.outputs.append(Tensor("C", (1,), DType.FLOAT32))
        g.add_node(Node("Add", inputs=["A", "A"], outputs=["C"], attributes={}))
        save(g, base_dir / "model.onnx")
        ds_dir = base_dir / "test_data_set_0"
        ds_dir.mkdir()
        _create_tensor_pb("A", np.array([1.0], dtype=np.float32), str(ds_dir / "input_0.pb"))
        _create_tensor_pb("B", np.array([1.0], dtype=np.float32), str(ds_dir / "input_1.pb"))
        _create_tensor_pb("C", np.array([2.0], dtype=np.float32), str(ds_dir / "output_0.pb"))
        runner = ONNXBackendTestRunner(providers=[CPUExecutionProvider({})])
        (passed, msg) = runner.run_node_test(base_dir)
        assert passed


def test_runner_mismatches() -> None:
    """Tests the runner mismatches functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        g = Graph("test_add")
        g.inputs.append(Tensor("A", (1,), DType.FLOAT32))
        g.outputs.append(Tensor("C", (1,), DType.FLOAT32))
        g.add_node(Node("Add", inputs=["A", "A"], outputs=["C"], attributes={}))
        save(g, base_dir / "model.onnx")
        ds_dir = base_dir / "test_data_set_0"
        ds_dir.mkdir()
        _create_tensor_pb("A", np.array([1.0], dtype=np.float32), str(ds_dir / "input_0.pb"))
        _create_tensor_pb("C", np.array([2.0, 2.0], dtype=np.float32), str(ds_dir / "output_0.pb"))
        runner = ONNXBackendTestRunner(providers=[CPUExecutionProvider({})])
        (passed, msg) = runner.run_node_test(base_dir)
        assert not passed
        assert "Shape mismatch" in msg
        _create_tensor_pb("C", np.array([2], dtype=np.int64), str(ds_dir / "output_0.pb"))
        (passed, msg) = runner.run_node_test(base_dir)
        assert not passed
        assert "DType mismatch" in msg


def test_runner_bool_int() -> None:
    """Tests the runner bool int functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        ds_dir = base_dir / "test_data_set_0"
        ds_dir.mkdir()

        class MockEP(CPUExecutionProvider):
            """Represents the MockEP class and its associated logic."""

            def get_supported_nodes(self, g):
                """Test the get supported nodes functionality."""
                return ["MockOp"]

            def execute(self, g, ctx, inputs):
                """Test the execute functionality."""
                return {
                    "C": Tensor("C", (1,), DType.BOOL, data=np.array([True], dtype=bool).tobytes())
                }

        g = Graph("t")
        g.inputs.append(Tensor("A", (1,), DType.BOOL))
        g.outputs.append(Tensor("C", (1,), DType.BOOL))
        g.add_node(Node("MockOp", ["A"], ["C"], {}, name="N1"))
        save(g, base_dir / "model.onnx")

        _create_tensor_pb("A", np.array([True], dtype=bool), str(ds_dir / "input_0.pb"))
        _create_tensor_pb("C", np.array([False], dtype=bool), str(ds_dir / "output_0.pb"))

        runner = ONNXBackendTestRunner(providers=[MockEP({})])
        passed, msg = runner.run_node_test(base_dir)
        assert not passed
        assert "Boolean array values mismatch" in msg

        # test int mismatch
        class MockEP2(CPUExecutionProvider):
            """Represents the MockEP2 class and its associated logic."""

            def get_supported_nodes(self, g):
                """Test the get supported nodes functionality."""
                return ["MockOp"]

            def execute(self, g, ctx, inputs):
                """Test the execute functionality."""
                return {
                    "C": Tensor(
                        "C", (1,), DType.INT64, data=np.array([1], dtype=np.int64).tobytes()
                    )
                }

        g = Graph("t2")
        g.inputs.append(Tensor("A", (1,), DType.INT64))
        g.outputs.append(Tensor("C", (1,), DType.INT64))
        g.add_node(Node("MockOp", ["A"], ["C"], {}, name="N1"))
        save(g, base_dir / "model.onnx")

        _create_tensor_pb("A", np.array([1], dtype=np.int64), str(ds_dir / "input_0.pb"))
        _create_tensor_pb("C", np.array([2], dtype=np.int64), str(ds_dir / "output_0.pb"))

        runner = ONNXBackendTestRunner(providers=[MockEP2({})])
        passed, msg = runner.run_node_test(base_dir)
        assert not passed
        assert "Numerical tolerance mismatch" in msg

        # test int pass
        _create_tensor_pb("C", np.array([1], dtype=np.int64), str(ds_dir / "output_0.pb"))
        passed, msg = runner.run_node_test(base_dir)
        assert passed


def test_runner_exception() -> None:
    """Tests the runner exception functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        g = Graph("test_add")
        g.inputs.append(Tensor("A", (1,), DType.FLOAT32))
        g.outputs.append(Tensor("C", (1,), DType.FLOAT32))
        g.add_node(Node("UnknownNodeXYZ", inputs=["A"], outputs=["C"], attributes={}))
        save(g, base_dir / "model.onnx")
        ds_dir = base_dir / "test_data_set_0"
        ds_dir.mkdir()
        _create_tensor_pb("A", np.array([1.0], dtype=np.float32), str(ds_dir / "input_0.pb"))
        _create_tensor_pb("C", np.array([1.0], dtype=np.float32), str(ds_dir / "output_0.pb"))
        runner = ONNXBackendTestRunner(providers=[CPUExecutionProvider({})])
        (passed, msg) = runner.run_node_test(base_dir)
        assert not passed
        assert "Execution failed:" in msg
