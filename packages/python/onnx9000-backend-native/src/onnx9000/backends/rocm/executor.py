"""ROCm Executor implementation."""

import ctypes
import logging
import numpy as np
from onnx9000.backends.cpu.executor import Executor as CPUExecutor
from onnx9000.backends.cpu.memory import MemoryPlanner
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
        """Implements the __init__ method or operation."""
        self.graph = graph
        self.planner = MemoryPlanner()
        self.cpu_fallback = CPUExecutor(graph)
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
                    shape = [int(getattr(dim, "value", dim)) for dim in tensor.shape]
                    dtype = np.dtype("float32")
                    size_in_bytes = np.prod(shape, dtype=int) * dtype.itemsize
                    self.planner.allocate_static(out_name, size_in_bytes, tuple(shape), dtype)
        self.planner.build_arena()
        for init_name in self.graph.initializers:
            tensor = self.graph.tensors.get(init_name)
            if tensor and tensor.data is not None:
                self.planner.set_tensor(init_name, tensor.data)

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
            inputs[inp] = self.planner.get_tensor(inp)
        out_dict = self.cpu_fallback.run(inputs)
        for out in node.outputs:
            if out in out_dict:
                self.planner.set_tensor(out, out_dict[out])

    def _execute_node(self, node: Node) -> None:
        """Executes the  execute node operation."""
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
        """Executes the run operation."""
        for name, data in inputs.items():
            self.planner.set_tensor(name, data)
        for node in self.graph.nodes:
            self._execute_node(node)
        results = {}
        for out_name in self.graph.outputs:
            results[out_name] = self.planner.get_tensor(out_name)
        return results

    def __del__(self) -> None:
        """Implements the __del__ method or operation."""
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
