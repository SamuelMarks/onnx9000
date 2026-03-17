"""CUDA Executor implementation."""

import ctypes
import logging
import numpy as np
from onnx9000.backends.cpu.executor import CPUExecutionProvider as CPUExecutor
from onnx9000.backends.cuda.bindings import (
    CUstream,
    _cublas_lib,
    _cuda_lib,
    _cudnn_lib,
    check_cublas_error,
    check_cuda_error,
    check_cudnn_error,
    cublasHandle_t,
    cudnnHandle_t,
    is_cublas_available,
    is_cuda_available,
    is_cudnn_available,
)
from onnx9000.backends.cuda.compiler import CUDACompiler
from onnx9000.backends.memory.cuda_arena import CUDAMemoryPlanner
from onnx9000.core.ir import Graph, Node

logger = logging.getLogger(__name__)


class Dispatcher:
    """NVIDIA CUDA & cuBLAS Dispatcher."""

    def __init__(self, graph: Graph) -> None:
        """Initialize the dispatcher."""
        self.graph = graph
        self.planner = CUDAMemoryPlanner()
        self.cpu_fallback = CPUExecutor({})
        self.stream = CUstream()
        self.cublas_handle = cublasHandle_t()
        self.cudnn_handle = cudnnHandle_t()
        self.initialized = False
        if is_cuda_available():
            check_cuda_error(_cuda_lib.cuInit(0))
            check_cuda_error(_cuda_lib.cuStreamCreate(ctypes.byref(self.stream), 0))
            if is_cublas_available():
                check_cublas_error(_cublas_lib.cublasCreate_v2(ctypes.byref(self.cublas_handle)))
                check_cublas_error(_cublas_lib.cublasSetStream_v2(self.cublas_handle, self.stream))
            if is_cudnn_available():
                check_cudnn_error(_cudnn_lib.cudnnCreate(ctypes.byref(self.cudnn_handle)))
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
                self.planner.set_tensor(init_name, data_arr)

    def _set_tensor_safe(self, name: str, data: np.ndarray) -> None:
        if name not in self.planner.offsets and name not in self.planner.dynamic_allocations:
            self.planner.allocate_dynamic(name, data.nbytes, data.shape, str(data.dtype))
        self.planner.set_tensor(name, data)

    def _execute_matmul(self, node: Node) -> None:
        """Execute MatMul operation."""
        if not is_cublas_available():
            raise RuntimeError("cuBLAS is required for MatMul")
        a_name, b_name = node.inputs[0], node.inputs[1]
        c_name = node.outputs[0]
        shape_a = self.planner.tensors_shape_dtype[a_name][0]
        shape_b = self.planner.tensors_shape_dtype[b_name][0]
        m = shape_a[0]
        k = shape_a[1]
        n = shape_b[1]
        a_ptr = self.planner.get_tensor_ptr(a_name)
        b_ptr = self.planner.get_tensor_ptr(b_name)
        c_ptr = self.planner.get_tensor_ptr(c_name)
        alpha = ctypes.c_float(1.0)
        beta = ctypes.c_float(0.0)
        check_cublas_error(
            _cublas_lib.cublasSgemm_v2(
                self.cublas_handle,
                0,
                0,
                n,
                m,
                k,
                ctypes.byref(alpha),
                b_ptr,
                n,
                a_ptr,
                k,
                ctypes.byref(beta),
                c_ptr,
                n,
            )
        )

    def _execute_elementwise(self, node: Node) -> None:
        """Dynamically JIT compile and launch an elementwise kernel."""
        op_symbol = "+" if node.op_type == "Add" else "-" if node.op_type == "Sub" else "*"
        kernel_code = f'\n        extern "C" __global__ void elementwise_{node.op_type}(const float* A, const float* B, float* C, int N) {{\n            int idx = blockIdx.x * blockDim.x + threadIdx.x;\n            if (idx < N) {{\n                C[idx] = A[idx] {op_symbol} B[idx];\n            }}\n        }}\n        '
        ptx = CUDACompiler.compile_kernel(kernel_code, f"elementwise_{node.op_type}")
        if not ptx:
            self._cpu_fallback_node(node)
            return
        module = ctypes.c_void_p()
        check_cuda_error(_cuda_lib.cuModuleLoadData(ctypes.byref(module), ptx))
        func = ctypes.c_void_p()
        check_cuda_error(
            _cuda_lib.cuModuleGetFunction(
                ctypes.byref(func), module, f"elementwise_{node.op_type}".encode()
            )
        )
        a_name, b_name = node.inputs[0], node.inputs[1]
        c_name = node.outputs[0]
        a_ptr = self.planner.get_tensor_ptr(a_name)
        b_ptr = self.planner.get_tensor_ptr(b_name)
        c_ptr = self.planner.get_tensor_ptr(c_name)
        shape = self.planner.tensors_shape_dtype[a_name][0]
        n_elements = np.prod(shape, dtype=int)
        blocks, threads = CUDACompiler.calculate_grid_block(n_elements)
        a_ptr_arg = ctypes.c_void_p(a_ptr.value)
        b_ptr_arg = ctypes.c_void_p(b_ptr.value)
        c_ptr_arg = ctypes.c_void_p(c_ptr.value)
        n_arg = ctypes.c_int(n_elements)
        args = (ctypes.c_void_p * 4)(
            ctypes.cast(ctypes.pointer(a_ptr_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.pointer(b_ptr_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.pointer(c_ptr_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.pointer(n_arg), ctypes.c_void_p),
        )
        check_cuda_error(
            _cuda_lib.cuLaunchKernel(func, blocks, 1, 1, threads, 1, 1, 0, self.stream, args, None)
        )
        _cuda_lib.cuModuleUnload(module)

    def _cpu_fallback_node(self, node: Node) -> None:
        """Run an unsupported op on the CPU."""
        inputs = {}
        for inp in node.inputs:
            inputs[inp] = self.planner.get_host_tensor(inp)
        out_dict = self.cpu_fallback.execute(self.graph, None, inputs)
        for out in node.outputs:
            if out in out_dict:
                val = out_dict[out]
                if hasattr(val, "data") and val.data is not None:
                    val = np.frombuffer(val.data, dtype=np.float32).reshape(
                        [int(x) for x in val.shape]
                    )
                self._set_tensor_safe(out, val)

    def _execute_node(self, node: Node) -> None:
        """Dispatch a single node."""
        if not self.initialized:
            self._cpu_fallback_node(node)
            return
        if node.op_type == "MatMul":
            self._execute_matmul(node)
        elif node.op_type in ["Add", "Sub", "Mul"]:
            self._execute_elementwise(node)
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
            results[name] = self.planner.get_host_tensor(name)
        return results

    def __del__(self) -> None:
        """Cleanup resources."""
        if getattr(self, "initialized", False) is False:
            return
        if is_cublas_available() and getattr(self, "cublas_handle", None):
            try:
                _cublas_lib.cublasDestroy_v2(self.cublas_handle)
            except Exception as e:
                logger.debug(f"cuBLAS destroy error: {e}")
        if is_cudnn_available() and getattr(self, "cudnn_handle", None):
            try:
                _cudnn_lib.cudnnDestroy(self.cudnn_handle)
            except Exception as e:
                logger.debug(f"cuDNN destroy error: {e}")
        if is_cuda_available() and getattr(self, "stream", None):
            try:
                _cuda_lib.cuStreamDestroy_v2(self.stream)
            except Exception as e:
                logger.debug(f"CUDA stream destroy error: {e}")
