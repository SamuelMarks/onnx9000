"""CUDA Executor implementation."""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
import ctypes

from onnx9000.core.ir import Graph, Node
from onnx9000.backends.cuda.bindings import (
    is_cuda_available,
    is_cublas_available,
    is_cudnn_available,
    _cuda_lib,
    _cublas_lib,
    _cudnn_lib,
    CUstream,
    cublasHandle_t,
    cudnnHandle_t,
    check_cuda_error,
    check_cublas_error,
    check_cudnn_error,
    CUdeviceptr,
    cudnnTensorDescriptor_t,
    cudnnFilterDescriptor_t,
)
from onnx9000.backends.cuda.memory import CUDAMemoryPlanner
from onnx9000.backends.cpu.executor import Executor as CPUExecutor
from onnx9000.backends.cuda.compiler import CUDACompiler

logger = logging.getLogger(__name__)


class Dispatcher:
    """NVIDIA CUDA & cuBLAS Dispatcher."""

    def __init__(self, graph: Graph) -> None:
        """Provides   init   functionality and verification."""
        self.graph = graph
        self.planner = CUDAMemoryPlanner()
        self.cpu_fallback = CPUExecutor(graph)

        self.stream = CUstream()
        self.cublas_handle = cublasHandle_t()
        self.cudnn_handle = cudnnHandle_t()
        self.initialized = False

        if is_cuda_available():
            check_cuda_error(_cuda_lib.cuInit(0))
            check_cuda_error(_cuda_lib.cuStreamCreate(ctypes.byref(self.stream), 0))
            if is_cublas_available():
                check_cublas_error(
                    _cublas_lib.cublasCreate_v2(ctypes.byref(self.cublas_handle))
                )
                check_cublas_error(
                    _cublas_lib.cublasSetStream_v2(self.cublas_handle, self.stream)
                )
            if is_cudnn_available():
                check_cudnn_error(
                    _cudnn_lib.cudnnCreate(ctypes.byref(self.cudnn_handle))
                )
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
                        if hasattr(dim, "value") and isinstance(dim.value, str):
                            is_dynamic = True
                            break
                        shape.append(int(getattr(dim, "value", dim)))

                    if not is_dynamic:
                        dtype = np.float32
                        if tensor.dtype:
                            dtype = np.dtype("float32")
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
        op_symbol = (
            "+" if node.op_type == "Add" else ("-" if node.op_type == "Sub" else "*")
        )
        kernel_code = f"""
        extern "C" __global__ void elementwise_{node.op_type}(const float* A, const float* B, float* C, int N) {{
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < N) {{
                C[idx] = A[idx] {op_symbol} B[idx];
            }}
        }}
        """
        ptx = CUDACompiler.compile_kernel(kernel_code, f"elementwise_{node.op_type}")
        if not ptx:
            # CPU fallback if nvcc missing
            self._cpu_fallback_node(node)
            return

        module = ctypes.c_void_p()
        check_cuda_error(_cuda_lib.cuModuleLoadData(ctypes.byref(module), ptx))

        func = ctypes.c_void_p()
        check_cuda_error(
            _cuda_lib.cuModuleGetFunction(
                ctypes.byref(func),
                module,
                f"elementwise_{node.op_type}".encode("utf-8"),
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

        # Args
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
            _cuda_lib.cuLaunchKernel(
                func, blocks, 1, 1, threads, 1, 1, 0, self.stream, args, None
            )
        )
        _cuda_lib.cuModuleUnload(module)

    def _cpu_fallback_node(self, node: Node) -> None:
        """Run an unsupported op on the CPU."""
        inputs = {}
        for inp in node.inputs:
            inputs[inp] = self.planner.get_host_tensor(inp)

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
        elif node.op_type in ["Add", "Sub", "Mul"]:
            self._execute_elementwise(node)
        else:
            # Auto fallback for anything not natively wrapped
            self._cpu_fallback_node(node)

    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Executes the run operation."""
        for name, data in inputs.items():
            self.planner.set_tensor(name, data)

        for node in self.graph.nodes:
            self._execute_node(node)

        results = {}
        for out_name in self.graph.outputs:
            results[out_name] = self.planner.get_host_tensor(out_name)

        return results

    def __del__(self) -> None:
        """Provides   del   functionality and verification."""
        if not self.initialized:
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
