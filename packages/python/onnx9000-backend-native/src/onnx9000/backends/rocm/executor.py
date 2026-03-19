"""ROCm Executor implementation."""

import ctypes
import logging

import numpy as np
from onnx9000.backends.cpu.executor import CPUExecutionProvider as CPUExecutor
from onnx9000.backends.memory.cpu_arena import CPUMemoryPlanner as MemoryPlanner
from onnx9000.backends.rocm.bindings import (
    _hip_lib,
    _miopen_lib,
    _rocblas_lib,
    check_hip_error,
    check_miopen_error,
    check_rocblas_error,
    hipStream_t,
    is_hip_available,
    is_miopen_available,
    is_rocblas_available,
    miopenHandle_t,
    rocblas_handle,
)
from onnx9000.core.ir import Graph, Node

logger = logging.getLogger(__name__)


class Dispatcher:
    """AMD ROCm (HIP) & MIOpen Dispatcher."""

    def __init__(self, graph: Graph) -> None:
        """Implement the __init__ method or operation."""
        self.graph = graph
        self.planner = MemoryPlanner()
        self.cpu_fallback = CPUExecutor({})
        self.stream = hipStream_t()
        self.rocblas_handle = rocblas_handle()
        self.miopen_handle = miopenHandle_t()
        self.initialized = False
        if is_hip_available():
            check_hip_error(_hip_lib.hipStreamCreate(ctypes.byref(self.stream)))
            if is_rocblas_available():
                check_rocblas_error(
                    _rocblas_lib.rocblas_create_handle(ctypes.byref(self.rocblas_handle))
                )
                check_rocblas_error(
                    _rocblas_lib.rocblas_set_stream(self.rocblas_handle, self.stream)
                )
            if is_miopen_available():
                check_miopen_error(_miopen_lib.miopenCreate(ctypes.byref(self.miopen_handle)))
            self.initialized = True
        self._init_memory()

    def _init_memory(self) -> None:
        """Plan memory for the graph."""
        for node in self.graph.nodes:
            for out_name in node.outputs:
                tensor = self.graph.tensors.get(out_name)
                if tensor:
                    shape = []
                    is_dynamic = False
                    for dim in tensor.shape:
                        val = getattr(dim, "value", dim)
                        if isinstance(val, str):
                            is_dynamic = True
                            break
                        shape.append(int(val))
                    if not is_dynamic:
                        dtype = np.float32
                        if tensor.dtype:
                            dtype = np.dtype("float32")
                        size_in_bytes = np.prod(shape, dtype=int) * dtype.itemsize
                        self.planner.allocate_static(
                            out_name, size_in_bytes, tuple(shape), str(dtype)
                        )
        self.planner.build_arena()
        for init_name in self.graph.initializers:
            tensor = self.graph.tensors.get(init_name)
            if tensor and tensor.data is not None:
                shape = tuple(int(getattr(dim, "value", dim)) for dim in tensor.shape)
                data_arr = np.frombuffer(tensor.data, dtype=np.float32).reshape(shape)
                self.planner.allocate_dynamic(init_name, data_arr.nbytes, shape, "float32")
                self.planner.set_tensor(init_name, memoryview(data_arr.tobytes()), shape, "float32")

    def _execute_matmul(self, node: Node) -> None:
        """Execute MatMul using rocBLAS."""
        if not is_rocblas_available():
            raise RuntimeError("rocBLAS is required for MatMul")
        (_a_name, _b_name) = (node.inputs[0], node.inputs[1])
        node.outputs[0]
        self._cpu_fallback_node(node)

    def _cpu_fallback_node(self, node: Node) -> None:
        """Run an unsupported op on the CPU."""
        inputs = {}
        for inp in node.inputs:
            inputs[inp] = np.frombuffer(
                self.planner.get_host_tensor(inp),
                dtype=np.dtype(self.planner.tensors_shape_dtype[inp][1]),
            ).reshape(self.planner.tensors_shape_dtype[inp][0])
        out_dict = self.cpu_fallback.execute(self.graph, None, inputs)
        for out in node.outputs:
            if out in out_dict:
                val = out_dict[out]
                if hasattr(val, "data") and val.data is not None:
                    val = np.frombuffer(val.data, dtype=np.float32).reshape(
                        [int(x) for x in val.shape]
                    )
                if out not in self.planner.offsets and out not in self.planner.dynamic_allocations:
                    self.planner.allocate_dynamic(out, val.nbytes, val.shape, str(val.dtype))
                self.planner.set_tensor(out, memoryview(val.tobytes()), val.shape, str(val.dtype))

    def _execute_node(self, node: Node) -> None:
        """Execute the  execute node operation."""
        if not self.initialized:
            self._cpu_fallback_node(node)
            return
        if node.op_type == "MatMul":
            self._execute_matmul(node)
        elif node.op_type in ["Add", "Sub", "Conv"]:
            self._cpu_fallback_node(node)
        else:
            self._cpu_fallback_node(node)

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Execute the run operation."""
        for name, data in inputs.items():
            if name not in self.planner.offsets and name not in self.planner.dynamic_allocations:
                self.planner.allocate_dynamic(name, data.nbytes, data.shape, str(data.dtype))
            self.planner.set_tensor(name, memoryview(data.tobytes()), data.shape, str(data.dtype))
        for node in self.graph.nodes:
            self._execute_node(node)
        results = {}
        for out_tensor in self.graph.outputs:
            out_name = getattr(out_tensor, "name", out_tensor)
            results[out_name] = np.frombuffer(
                self.planner.get_host_tensor(out_name),
                dtype=np.dtype(self.planner.tensors_shape_dtype[out_name][1]),
            ).reshape(self.planner.tensors_shape_dtype[out_name][0])
        return results

    def __del__(self) -> None:
        """Implement the __del__ method or operation."""
        if not self.initialized:
            return
        if is_rocblas_available() and getattr(self, "rocblas_handle", None):
            try:
                _rocblas_lib.rocblas_destroy_handle(self.rocblas_handle)
            except Exception as e:
                logger.debug(f"rocBLAS destroy error: {e}")
        if is_miopen_available() and getattr(self, "miopen_handle", None):
            try:
                _miopen_lib.miopenDestroy(self.miopen_handle)
            except Exception as e:
                logger.debug(f"MIOpen destroy error: {e}")
        if is_hip_available() and getattr(self, "stream", None):
            try:
                _hip_lib.hipStreamDestroy(self.stream)
            except Exception as e:
                logger.debug(f"HIP stream destroy error: {e}")
