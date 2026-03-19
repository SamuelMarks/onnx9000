"""Pure-Python Test Case Runner targeting standard ONNX `.pb` directories."""

import glob
import logging
import os
from pathlib import Path
from typing import Union

import numpy as np
from onnx9000.backends.session import InferenceSession, SessionOptions
from onnx9000.core.dtypes import DType
from onnx9000.core.execution import ExecutionProvider
from onnx9000.core.ir import Tensor
from onnx9000.core.parser.core import load_tensor

logger = logging.getLogger(__name__)


class ONNXBackendTestRunner:
    """Run official ONNX node tests against ONNX9000 InferenceSession."""

    def __init__(
        self, providers: list[ExecutionProvider], rtol: float = 0.001, atol: float = 1e-05
    ) -> None:
        """Initialize the test runner."""
        self.providers = providers
        self.rtol = rtol
        self.atol = atol
        self.skip_lists: list[str] = []

    def set_skip_list(self, skip_patterns: list[str]) -> None:
        """Set a list of regular expressions to skip tests."""
        self.skip_lists = skip_patterns

    def load_tensors(self, directory: Path, prefix: str) -> list[Tensor]:
        """Load all tensors from a directory with a specific prefix (e.g., input_*.pb)."""
        files = sorted(glob.glob(os.path.join(directory, f"{prefix}_*.pb")))
        tensors = []
        for file in files:
            tensor = load_tensor(file)
            tensors.append(tensor)
        return tensors

    def _convert_tensor_to_numpy(self, tensor: Tensor) -> np.ndarray:
        """Convert an ir.Tensor to a NumPy ndarray."""
        if tensor.data is None:
            return np.empty(tensor.shape, dtype=np.float32)
        dtype_mapping = {
            DType.FLOAT32: np.float32,
            DType.FLOAT64: np.float64,
            DType.INT32: np.int32,
            DType.INT64: np.int64,
            DType.INT8: np.int8,
            DType.UINT8: np.uint8,
            DType.INT16: np.int16,
            DType.UINT16: np.uint16,
            DType.BOOL: bool,
            DType.FLOAT16: np.float16,
        }
        np_dtype = dtype_mapping.get(tensor.dtype)
        if np_dtype is None:
            raise TypeError(f"Unsupported dtype for numpy conversion: {tensor.dtype}")
        arr = np.frombuffer(tensor.data, dtype=np_dtype)
        return arr.reshape(tensor.shape)

    def run_node_test(self, test_dir: Union[str, Path]) -> tuple[bool, str]:
        """Execute a specific node test directory and compare outputs."""
        test_dir = Path(test_dir)
        model_path = test_dir / "model.onnx"
        if not model_path.exists():
            return (False, f"Model file not found: {model_path}")
        test_data_sets = sorted(glob.glob(os.path.join(test_dir, "test_data_set_*")))
        if not test_data_sets:
            return (False, "No test data sets found.")
        try:
            options = SessionOptions()
            session = InferenceSession(model_path, providers=self.providers, options=options)
            for ds in test_data_sets:
                ds_path = Path(ds)
                inputs = self.load_tensors(ds_path, "input")
                expected_outputs = self.load_tensors(ds_path, "output")
                if not expected_outputs:
                    continue
                input_feed = {}
                graph_inputs = session.get_inputs()
                for i, inp in enumerate(inputs):
                    if i < len(graph_inputs):
                        input_feed[graph_inputs[i].name] = inp
                    else:
                        input_feed[inp.name] = inp
                outputs = session.run(output_names=None, input_feed=input_feed)
                graph_outputs = session.get_outputs()
                for i, expected_output in enumerate(expected_outputs):
                    out_name = (
                        graph_outputs[i].name if i < len(graph_outputs) else expected_output.name
                    )
                    if i >= len(outputs):
                        return (False, f"Missing output tensor at index {i}: {out_name}")
                    actual_tensor = outputs[i]
                    actual_arr = self._convert_tensor_to_numpy(actual_tensor)
                    expected_arr = self._convert_tensor_to_numpy(expected_output)
                    if actual_arr.shape != expected_arr.shape:
                        return (
                            False,
                            f"Shape mismatch: {actual_arr.shape} != {expected_arr.shape}",
                        )
                    if actual_arr.dtype != expected_arr.dtype:
                        return (
                            False,
                            f"DType mismatch: {actual_arr.dtype} != {expected_arr.dtype}",
                        )
                    if expected_arr.dtype == bool:
                        if not np.array_equal(actual_arr, expected_arr):
                            return (False, "Boolean array values mismatch")
                    elif np.issubdtype(expected_arr.dtype, np.floating):
                        np.testing.assert_allclose(
                            actual_arr, expected_arr, rtol=self.rtol, atol=self.atol
                        )
                    else:
                        np.testing.assert_array_equal(actual_arr, expected_arr)
            return (True, "Passed")
        except AssertionError as e:
            return (False, f"Numerical tolerance mismatch: {e}")
        except Exception as e:
            return (False, f"Execution failed: {e}")
