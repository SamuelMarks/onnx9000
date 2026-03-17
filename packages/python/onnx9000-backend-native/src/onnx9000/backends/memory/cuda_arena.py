"""CUDA Memory Planner and Pool."""

import ctypes
import logging
import time
import numpy as np
from onnx9000.backends.cuda.bindings import (
    CUdeviceptr,
    _cuda_lib,
    check_cuda_error,
    is_cuda_available,
)

logger = logging.getLogger(__name__)


class CUDAMemoryPlanner:
    """Manages memory allocations on the CUDA device."""

    def __init__(self) -> None:
        """Implements the __init__ method or operation."""
        self.allocations: dict[str, CUdeviceptr] = {}
        self.arena_ptr = CUdeviceptr(0)
        self.offsets: dict[str, tuple[int, int]] = {}
        self.current_offset: int = 0
        self.tensors_shape_dtype: dict[str, tuple[tuple[int, ...], np.dtype]] = {}
        self.dynamic_allocations: dict[str, tuple[CUdeviceptr, int]] = {}

    def allocate_static(
        self, name: str, size_in_bytes: int, shape: tuple[int, ...], dtype: np.dtype
    ) -> None:
        """Executes the allocate static operation."""
        aligned_size = size_in_bytes + 255 & ~255
        self.offsets[name] = (self.current_offset, aligned_size)
        self.current_offset += aligned_size
        self.tensors_shape_dtype[name] = (shape, dtype)

    def build_arena(self) -> None:
        """Executes the build arena operation."""
        if not is_cuda_available():
            logger.warning("CUDA not available, skipping arena build.")
            return
        if self.current_offset > 0:
            ptr = CUdeviceptr()
            res = _cuda_lib.cuMemAlloc_v2(ctypes.byref(ptr), self.current_offset)
            check_cuda_error(res)
            self.arena_ptr = ptr

    def allocate_pinned(self, size_bytes: int) -> ctypes.c_void_p:
        """Support explicit CUDA Pinned Memory allocation (`cudaMallocHost`) natively in Python."""
        ptr = ctypes.c_void_p()
        if is_cuda_available():
            res = _cuda_lib.cuMemHostAlloc(ctypes.byref(ptr), size_bytes, 2)
            check_cuda_error(res)
        return ptr

    def allocate_managed(self, size_bytes: int) -> CUdeviceptr:
        """Support explicit Unified Memory allocation (`cudaMallocManaged`) natively."""
        ptr = CUdeviceptr()
        if is_cuda_available():
            res = _cuda_lib.cuMemAllocManaged(ctypes.byref(ptr), size_bytes, 1)
            check_cuda_error(res)
        return ptr

    def get_tensor_ptr(self, name: str) -> CUdeviceptr:
        """Executes the get tensor ptr operation."""
        if name in self.dynamic_allocations:
            return self.dynamic_allocations[name][0]
        if name in self.offsets:
            (offset, _) = self.offsets[name]
            return CUdeviceptr(self.arena_ptr.value + offset)
        raise RuntimeError(f"Tensor {name} not found in CUDA memory planner")

    def set_tensor(self, name: str, host_data: np.ndarray, stream: ctypes.c_void_p = None) -> None:
        """Executes the set tensor operation."""
        if not is_cuda_available():
            if not hasattr(self, "_cpu_fallback_tensors"):
                self._cpu_fallback_tensors = {}
            self._cpu_fallback_tensors[name] = host_data
            return
        size_bytes = host_data.nbytes
        if name in self.offsets:
            (offset, size) = self.offsets[name]
            if size_bytes > size:
                ptr = CUdeviceptr()
                check_cuda_error(_cuda_lib.cuMemAlloc_v2(ctypes.byref(ptr), size_bytes))
                self.dynamic_allocations[name] = (ptr, size_bytes)
                dst_ptr = ptr
            else:
                dst_ptr = CUdeviceptr(self.arena_ptr.value + offset)
        elif name in self.dynamic_allocations:
            (ptr, size) = self.dynamic_allocations[name]
            if size_bytes > size:
                check_cuda_error(_cuda_lib.cuMemFree_v2(ptr))
                check_cuda_error(_cuda_lib.cuMemAlloc_v2(ctypes.byref(ptr), size_bytes))
                self.dynamic_allocations[name] = (ptr, size_bytes)
            dst_ptr = ptr
        else:
            ptr = CUdeviceptr()
            check_cuda_error(_cuda_lib.cuMemAlloc_v2(ctypes.byref(ptr), size_bytes))
            self.dynamic_allocations[name] = (ptr, size_bytes)
            dst_ptr = ptr
        contiguous_data = np.ascontiguousarray(host_data)
        src_ptr = contiguous_data.ctypes.data_as(ctypes.c_void_p)
        start = time.perf_counter()
        if stream:
            check_cuda_error(_cuda_lib.cuMemcpyHtoDAsync_v2(dst_ptr, src_ptr, size_bytes, stream))
        else:
            check_cuda_error(_cuda_lib.cuMemcpyHtoD_v2(dst_ptr, src_ptr, size_bytes))
        elapsed = time.perf_counter() - start
        bw = size_bytes / 1000000000.0 / elapsed if elapsed > 0 else 0
        logger.debug(
            f"Transfer CPU -> GPU implicitly: {size_bytes} bytes in {elapsed:.4f}s ({bw:.2f} GB/s)"
        )
        if name not in self.offsets:
            self.tensors_shape_dtype[name] = (host_data.shape, host_data.dtype)

    def get_host_tensor(self, name: str, stream: ctypes.c_void_p = None) -> np.ndarray:
        """Executes the get host tensor operation."""
        if (
            not is_cuda_available()
            and hasattr(self, "_cpu_fallback_tensors")
            and (name in self._cpu_fallback_tensors)
        ):
            return self._cpu_fallback_tensors[name]
        if name not in self.offsets and name not in self.dynamic_allocations:
            raise RuntimeError(f"Tensor {name} not found.")
        if name in self.dynamic_allocations:
            (ptr, size_bytes) = self.dynamic_allocations[name]
            (shape, dtype) = self.tensors_shape_dtype.get(
                name, ((size_bytes // 4,), np.dtype("float32"))
            )
        else:
            (offset, size_bytes) = self.offsets[name]
            ptr = CUdeviceptr(self.arena_ptr.value + offset)
            (shape, dtype) = self.tensors_shape_dtype[name]
        host_data = np.empty(shape, dtype=dtype)
        dst_ptr = host_data.ctypes.data_as(ctypes.c_void_p)
        if is_cuda_available():
            start = time.perf_counter()
            if stream:
                check_cuda_error(_cuda_lib.cuMemcpyDtoHAsync_v2(dst_ptr, ptr, size_bytes, stream))
            else:
                check_cuda_error(_cuda_lib.cuMemcpyDtoH_v2(dst_ptr, ptr, size_bytes))
            elapsed = time.perf_counter() - start
            bw = size_bytes / 1000000000.0 / elapsed if elapsed > 0 else 0
            logger.debug(
                f"Transfer GPU -> CPU implicitly: {size_bytes} bytes in {elapsed:.4f}s ({bw:.2f} GB/s)"
            )
        return host_data

    def synchronize_stream(self, stream: ctypes.c_void_p) -> None:
        """Provide explicit memory barrier primitives (`cudaStreamSynchronize`)"""
        if is_cuda_available() and stream:
            check_cuda_error(_cuda_lib.cuStreamSynchronize(stream))

    def cleanup(self) -> None:
        """Executes the cleanup operation."""
        if not is_cuda_available():
            return
        if self.arena_ptr.value != 0:
            _cuda_lib.cuMemFree_v2(self.arena_ptr)
            self.arena_ptr = CUdeviceptr(0)
        for ptr, _ in self.dynamic_allocations.values():
            _cuda_lib.cuMemFree_v2(ptr)
        self.dynamic_allocations.clear()

    def __del__(self) -> None:
        """Implements the __del__ method or operation."""
        try:
            self.cleanup()
        except Exception as e:
            logger.debug(f"Exception during CUDAMemoryPlanner cleanup: {e}")

    def allocate_dynamic(self, name: str, size: int, shape: tuple[int, ...], dtype: str) -> None:
        if not is_cuda_available():
            return
        if name in self.dynamic_allocations:
            _cuda_lib.cuMemFree(self.dynamic_allocations[name][0])
        ptr = CUdeviceptr(0)
        check_cuda_error(_cuda_lib.cuMemAlloc(ctypes.byref(ptr), size))
        self.dynamic_allocations[name] = (ptr, size)
        self.tensors_shape_dtype[name] = (shape, np.dtype(dtype))
