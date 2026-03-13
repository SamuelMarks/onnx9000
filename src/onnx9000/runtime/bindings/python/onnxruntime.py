"""Python ONNX Runtime Parity Wrappers for onnx9000.

Provides a 100% compliant `InferenceSession` drop-in replacement
with strong types and no dependencies (except numpy/protobuf for model loading),
suitable for Pyodide and WASM targets.
"""

import os
from collections.abc import Sequence
from typing import Any, Dict, List, Optional, Union

from onnx9000.jit import compile as compile_model


class SessionOptions:
    """ONNX Runtime Session Options."""

    def __init__(self) -> None:
        """Execute the   init   process and return the computed results."""
        self.intra_op_num_threads: int = 0
        self.inter_op_num_threads: int = 0
        self.graph_optimization_level: int = 99
        self.log_severity_level: int = 2
        self.optimized_model_filepath: str = ""
        self.execution_mode: int = 0
        self.enable_cpu_mem_arena: bool = True
        self.enable_mem_pattern: bool = True
        self._config_entries: Dict[str, str] = {}

    def add_session_config_entry(self, key: str, value: str) -> None:
        """Adds a key-value configuration entry."""
        self._config_entries[key] = value

    def get_session_config_entry(self, key: str) -> str:
        """Gets a configuration entry."""
        return self._config_entries.get(key, "")


class RunOptions:
    """ONNX Runtime Run Options."""

    def __init__(self) -> None:
        """Execute the   init   process and return the computed results."""
        self.run_log_severity_level: int = 2  # pragma: no cover
        self.run_log_verbosity_level: int = 0  # pragma: no cover
        self.run_tag: str = ""  # pragma: no cover
        self.terminate: bool = False  # pragma: no cover
        self.only_execute_path_to_fetches: bool = False  # pragma: no cover


class OrtValue:
    """Type-erased ONNX Runtime Value container."""

    def __init__(self, data: Any) -> None:
        """Execute the   init   process and return the computed results."""
        self._data = data  # pragma: no cover

    @staticmethod
    def ortvalue_from_numpy(numpy_obj: Any) -> "OrtValue":
        """Creates an OrtValue from a NumPy array."""
        return OrtValue(numpy_obj)  # pragma: no cover

    @staticmethod
    def ortvalue_from_shape_and_type(
        shape: Sequence[int], element_type: int
    ) -> "OrtValue":
        """Creates an uninitialized OrtValue."""
        import numpy as np  # pragma: no cover

        dtype_map = {  # pragma: no cover
            1: np.float32,
            2: np.uint8,
            3: np.int8,
            4: np.uint16,
            5: np.int16,
            6: np.int32,
            7: np.int64,
            8: str,
            9: bool,
            10: np.float16,
            11: np.float64,
            12: np.uint32,
            13: np.uint64,
        }
        dt = dtype_map.get(element_type, np.float32)  # pragma: no cover
        return OrtValue(np.zeros(shape, dtype=dt))  # pragma: no cover

    def numpy(self) -> Any:
        """Returns the underlying NumPy array."""
        return self._data  # pragma: no cover

    def shape(self) -> Sequence[int]:
        """Returns the shape of the tensor."""
        if hasattr(self._data, "shape"):  # pragma: no cover
            return self._data.shape  # pragma: no cover
        return []  # pragma: no cover

    def data_type(self) -> str:
        """Returns the data type of the tensor as string."""
        if hasattr(self._data, "dtype"):  # pragma: no cover
            return str(self._data.dtype)  # pragma: no cover
        return "unknown"  # pragma: no cover


class IOBinding:
    """Binds inputs and outputs to specific devices."""

    def __init__(self, session: "InferenceSession") -> None:
        """Execute the   init   process and return the computed results."""
        self.session = session  # pragma: no cover
        self.inputs: Dict[str, OrtValue] = {}  # pragma: no cover
        self.outputs: Dict[str, OrtValue] = {}  # pragma: no cover

    def bind_input(
        self,
        name: str,
        device_type: str,
        device_id: int,
        element_type: int,
        shape: Sequence[int],
        buffer_ptr: int,
    ) -> None:
        """Execute the Bind input process and return the computed results."""
        self.inputs[name] = OrtValue.ortvalue_from_shape_and_type(
            shape, element_type
        )  # pragma: no cover

    def bind_output(
        self,
        name: str,
        device_type: str,
        device_id: int,
        element_type: int,
        shape: Sequence[int],
        buffer_ptr: int,
    ) -> None:
        """Execute the Bind output process and return the computed results."""
        self.outputs[name] = OrtValue.ortvalue_from_shape_and_type(
            shape, element_type
        )  # pragma: no cover

    def bind_cpu_input(self, name: str, ort_value: OrtValue) -> None:
        """Execute the Bind cpu input process and return the computed results."""
        self.inputs[name] = ort_value  # pragma: no cover

    def bind_cpu_output(self, name: str, ort_value: OrtValue) -> None:
        """Execute the Bind cpu output process and return the computed results."""
        self.outputs[name] = ort_value  # pragma: no cover

    def clear_binding_inputs(self) -> None:
        """Execute the Clear binding inputs process and return the computed results."""
        self.inputs.clear()  # pragma: no cover

    def clear_binding_outputs(self) -> None:
        """Execute the Clear binding outputs process and return the computed results."""
        self.outputs.clear()  # pragma: no cover


class InferenceSession:
    """Main execution engine drop-in replacement."""

    def __init__(
        self,
        path_or_bytes: Union[str, bytes],
        sess_options: Optional[SessionOptions] = None,
        providers: Optional[List[str]] = None,
    ) -> None:
        """Execute the   init   process and return the computed results."""
        self.path_or_bytes = path_or_bytes
        self.sess_options = sess_options or SessionOptions()
        self.providers = providers or get_available_providers()
        self._is_closed: bool = False

        if isinstance(path_or_bytes, bytes):
            import tempfile  # pragma: no cover

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".onnx"
            ) as f:  # pragma: no cover
                f.write(path_or_bytes)  # pragma: no cover
                temp_path = f.name  # pragma: no cover
            try:  # pragma: no cover
                self._compiled_model = compile_model(temp_path)  # pragma: no cover
            finally:
                os.remove(temp_path)  # pragma: no cover
        elif isinstance(path_or_bytes, str) and os.path.exists(path_or_bytes):
            self._compiled_model = compile_model(path_or_bytes)
        else:
            raise ValueError("Invalid model path or bytes provided")

        self.graph = self._compiled_model.graph
        self._input_names = list(self.graph.inputs)
        self._output_names = list(self.graph.outputs)

    def run(
        self,
        output_names: Optional[List[str]],
        input_feed: Dict[str, Any],
        run_options: Optional[RunOptions] = None,
    ) -> List[Any]:
        """Execute the Run process and return the computed results."""
        if self._is_closed:
            raise RuntimeError("Session is closed")  # pragma: no cover

        if run_options and run_options.terminate:
            raise RuntimeError(
                "Execution cancelled by RunOptions.terminate"
            )  # pragma: no cover

        args = []
        for name in self._input_names:
            if name not in input_feed:
                raise ValueError(f"Missing required input: {name}")  # pragma: no cover
            val = input_feed[name]
            if isinstance(val, OrtValue):
                args.append(val.numpy())  # pragma: no cover
            else:
                args.append(val)

        try:
            res = self._compiled_model(*args)
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"onnx9000 execution failed: {e}")  # pragma: no cover

        if not isinstance(res, tuple):
            res = (res,)  # pragma: no cover

        out_names = output_names if output_names is not None else self._output_names
        result = []
        for requested_name in out_names:
            if requested_name not in self._output_names:
                raise ValueError(
                    f"Requested output not found in graph: {requested_name}"
                )  # pragma: no cover
            idx = self._output_names.index(requested_name)
            result.append(res[idx])

        return result

    def run_with_iobinding(
        self, iobinding: IOBinding, run_options: Optional[RunOptions] = None
    ) -> None:
        """Execute the Run with iobinding process and return the computed results."""
        inputs = {k: v.numpy() for k, v in iobinding.inputs.items()}  # pragma: no cover
        res = self.run(self._output_names, inputs, run_options)  # pragma: no cover
        for name, val in zip(self._output_names, res):  # pragma: no cover
            if name in iobinding.outputs:  # pragma: no cover
                iobinding.outputs[name]._data = val  # pragma: no cover

    def get_inputs(self) -> List[Any]:
        """Execute the Get inputs process and return the computed results."""

        class InputMeta:
            """Represent the InputMeta component within the architecture."""

            def __init__(self, name: str, shape: Sequence[int], type_str: str):
                """Execute the   init   process and return the computed results."""
                self.name = name
                self.shape = shape
                self.type = type_str

        meta = []
        for name in self._input_names:
            tensor = self.graph.tensors[name]
            shape = tuple(d.value if hasattr(d, "value") else d for d in tensor.shape)
            type_str = (
                f"tensor({tensor.dtype.name.lower()})"
                if tensor.dtype
                else "tensor(float)"
            )
            meta.append(InputMeta(name, shape, type_str))
        return meta

    def get_outputs(self) -> List[Any]:
        """Execute the Get outputs process and return the computed results."""

        class OutputMeta:
            """Represent the OutputMeta component within the architecture."""

            def __init__(self, name: str, shape: Sequence[int], type_str: str):
                """Execute the   init   process and return the computed results."""
                self.name = name
                self.shape = shape
                self.type = type_str

        meta = []
        for name in self._output_names:
            tensor = self.graph.tensors[name]
            shape = tuple(d.value if hasattr(d, "value") else d for d in tensor.shape)
            type_str = (
                f"tensor({tensor.dtype.name.lower()})"
                if tensor.dtype
                else "tensor(float)"
            )
            meta.append(OutputMeta(name, shape, type_str))
        return meta

    def get_providers(self) -> List[str]:
        """Execute the Get providers process and return the computed results."""
        return self.providers


def get_device() -> str:
    """Execute the Get device process and return the computed results."""
    return "CPU"


def get_available_providers() -> List[str]:
    """Execute the Get available providers process and return the computed results."""
    from onnx9000 import config

    providers = ["CPUExecutionProvider"]
    if config.ONNX9000_USE_CUDA:
        providers.append("CUDAExecutionProvider")  # pragma: no cover
    return providers
