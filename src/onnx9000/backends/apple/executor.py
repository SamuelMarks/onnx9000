"""Apple Executor implementation."""

import numpy as np
import logging
import ctypes

from onnx9000.core.ir import Graph, Node
from onnx9000.backends.apple.bindings import (
    is_accelerate_available,
    is_metal_available,
    is_mps_available,
    _accelerate_lib,
    _metal_lib,
    mtl_create_system_default_device,
    get_class,
    get_selector,
    nsstring,
)
from onnx9000.backends.cpu.memory import MemoryPlanner
from onnx9000.backends.cpu.executor import Executor as CPUExecutor

logger = logging.getLogger(__name__)


class Dispatcher:
    """Apple Metal (MPS) & Accelerate Dispatcher."""

    def __init__(self, graph: Graph) -> None:
        """Provides   init   functionality and verification."""
        self.graph = graph
        self.planner = MemoryPlanner()
        self.cpu_fallback = CPUExecutor(graph)
        self.device = mtl_create_system_default_device()
        self.initialized = False

        if self.device:
            self.initialized = True

        self._init_memory()

    def _init_memory(self) -> None:
        """Plan memory for the graph."""
        for node in self.graph.nodes:
            for out_name in node.outputs:
                tensor = self.graph.tensors.get(out_name)
                if tensor:
                    shape = [int(getattr(dim, "value", dim)) for dim in tensor.shape]
                    dtype = np.dtype("float32")  # Default float32
                    size_in_bytes = np.prod(shape, dtype=int) * dtype.itemsize
                    self.planner.allocate_static(
                        out_name, size_in_bytes, tuple(shape), dtype
                    )

        self.planner.build_arena()

        for init_name in self.graph.initializers:
            tensor = self.graph.tensors.get(init_name)
            if tensor and tensor.data is not None:
                self.planner.set_tensor(init_name, tensor.data)

    def _execute_matmul(self, node: Node) -> None:
        """Executes the  execute matmul operation."""
        a_name, b_name = node.inputs[0], node.inputs[1]
        c_name = node.outputs[0]

        a_data = self.planner.get_tensor(a_name)
        b_data = self.planner.get_tensor(b_name)

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

            self.planner.set_tensor(c_name, c_data)
        else:
            self._cpu_fallback_node(node)

    def _execute_elementwise(self, node: Node, op_type: str) -> None:
        """Executes the  execute elementwise operation."""
        a_name, b_name = node.inputs[0], node.inputs[1]
        c_name = node.outputs[0]

        a_data = self.planner.get_tensor(a_name)
        b_data = self.planner.get_tensor(b_name)

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
            self.planner.set_tensor(c_name, c_data)
        else:
            self._cpu_fallback_node(node)

    def _cpu_fallback_node(self, node: Node) -> None:
        """Executes the  cpu fallback node operation."""
        inputs = {}
        for inp in node.inputs:
            inputs[inp] = self.planner.get_tensor(inp)

        out_dict = self.cpu_fallback.run(inputs)
        for out in node.outputs:
            if out in out_dict:
                self.planner.set_tensor(out, out_dict[out])

    def _execute_node(self, node: Node) -> None:
        """Executes the  execute node operation."""
        if node.op_type == "MatMul":
            self._execute_matmul(node)
        elif node.op_type in ["Add", "Sub", "Mul"]:
            self._execute_elementwise(node, node.op_type)
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
