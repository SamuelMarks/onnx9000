"""Apple Executor implementation."""

import ctypes
import logging

import numpy as np
from onnx9000.backends.apple.bindings import (
    _accelerate_lib,
    is_accelerate_available,
    mtl_create_system_default_device,
)
from onnx9000.backends.cpu.executor import CPUExecutionProvider as CPUExecutor
from onnx9000.backends.memory.cpu_arena import CPUMemoryPlanner as MemoryPlanner
from onnx9000.core.ir import Graph, Node

logger = logging.getLogger(__name__)


class Dispatcher:
    """Apple Metal (MPS) & Accelerate Dispatcher."""

    def __init__(self, graph: Graph) -> None:
        """Initialize the dispatcher."""
        self.graph = graph
        self.planner = MemoryPlanner()
        self.cpu_fallback = CPUExecutor({})
        self.device = mtl_create_system_default_device()
        self.initialized = bool(self.device)
        self._init_memory()

    def _init_memory(self) -> None:
        """Plan memory for the graph."""
        for node in self.graph.nodes:
            for out_name in node.outputs:
                tensor = self.graph.tensors.get(out_name)
                if tensor:
                    shape = tuple(int(getattr(dim, "value", dim)) for dim in tensor.shape)
                    dtype = np.dtype("float32")
                    size_in_bytes = int(np.prod(shape)) * dtype.itemsize
                    self.planner.allocate_static(out_name, size_in_bytes, shape, str(dtype))
        self.planner.build_arena()
        for init_name in self.graph.initializers:
            tensor = self.graph.tensors.get(init_name)
            if tensor and tensor.data is not None:
                shape = tuple(int(getattr(dim, "value", dim)) for dim in tensor.shape)
                self.planner.allocate_dynamic(init_name, len(tensor.data), shape, "float32")
                self.planner.set_tensor(init_name, memoryview(tensor.data), shape, "float32")

    def _get_tensor(self, name: str) -> np.ndarray:
        """Executes the get tensor operation."""
        raw = self.planner.get_host_tensor(name)
        shape, dtype = self.planner.tensors_shape_dtype[name]
        return np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(shape)

    def _execute_matmul(self, node: Node) -> None:
        """Execute MatMul using Accelerate or fallback."""
        a_name, b_name = node.inputs[0], node.inputs[1]
        c_name = node.outputs[0]
        a_data = self._get_tensor(a_name)
        b_data = self._get_tensor(b_name)
        if is_accelerate_available():
            shape_a = a_data.shape
            shape_b = b_data.shape
            m = shape_a[0]
            k = shape_a[1]
            n = shape_b[1]
            c_data = np.zeros((m, n), dtype=np.float32)
            a_ptr = a_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            b_ptr = b_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            c_ptr = c_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            _accelerate_lib.cblas_sgemm(
                101, 111, 111, m, n, k, 1.0, a_ptr, k, b_ptr, n, 0.0, c_ptr, n
            )
            self._set_tensor_safe(c_name, c_data)
        else:
            self._cpu_fallback_node(node)

    def _execute_elementwise(self, node: Node, op_type: str) -> None:
        """Execute elementwise ops using Accelerate or fallback."""
        a_name, b_name = node.inputs[0], node.inputs[1]
        c_name = node.outputs[0]
        a_data = self._get_tensor(a_name)
        b_data = self._get_tensor(b_name)
        size = a_data.size
        c_data = np.zeros_like(a_data)
        a_ptr = a_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        b_ptr = b_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        c_ptr = c_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        if is_accelerate_available():
            if op_type == "Add":
                _accelerate_lib.vDSP_vadd(a_ptr, 1, b_ptr, 1, c_ptr, 1, size)
            elif op_type == "Sub":
                _accelerate_lib.vDSP_vsub(b_ptr, 1, a_ptr, 1, c_ptr, 1, size)
            elif op_type == "Mul":
                _accelerate_lib.vDSP_vmul(a_ptr, 1, b_ptr, 1, c_ptr, 1, size)
            self._set_tensor_safe(c_name, c_data)
        else:
            self._cpu_fallback_node(node)

    def _cpu_fallback_node(self, node: Node) -> None:
        """Fallback to CPU Execution Provider."""
        inputs = {}
        for inp in node.inputs:
            inputs[inp] = self._get_tensor(inp)
        out_dict = self.cpu_fallback.execute(self.graph, None, inputs)
        for out in node.outputs:
            if out in out_dict:
                val = out_dict[out]
                if hasattr(val, "data") and val.data is not None:
                    # it's a Tensor
                    import numpy as np

                    val = np.frombuffer(val.data, dtype=np.float32).reshape(
                        [int(x) for x in val.shape]
                    )
                self._set_tensor_safe(out, val)

    def _set_tensor_safe(self, name: str, data: np.ndarray) -> None:
        """Executes the set tensor safe operation."""
        if name not in self.planner.offsets and name not in self.planner.dynamic_allocations:
            self.planner.allocate_dynamic(name, data.nbytes, data.shape, str(data.dtype))
        self.planner.set_tensor(name, memoryview(data.tobytes()), data.shape, str(data.dtype))

    def _execute_node(self, node: Node) -> None:
        """Dispatch a single node."""
        if node.op_type == "MatMul":
            self._execute_matmul(node)
        elif node.op_type in ["Add", "Sub", "Mul"]:
            self._execute_elementwise(node, node.op_type)
        else:
            self._cpu_fallback_node(node)

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Execute the graph."""
        for name, data in inputs.items():
            self._set_tensor_safe(name, data)
        for node in self.graph.nodes:
            self._execute_node(node)
        results = {}
        for out_tensor in self.graph.outputs:
            name = getattr(out_tensor, "name", out_tensor)
            results[name] = self._get_tensor(name)
        return results
