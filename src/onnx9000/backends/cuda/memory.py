"""CUDA Memory Planner and Pool."""

import ctypes
import numpy as np
from typing import Dict, Tuple, Optional
import logging

from onnx9000.backends.cuda.bindings import (
    is_cuda_available,
    _cuda_lib,
    CUdeviceptr,
    check_cuda_error,
)

logger = logging.getLogger(__name__)


class CUDAMemoryPlanner:
    """Manages memory allocations on the CUDA device."""

    def __init__(self) -> None:
        """Provides   init   functionality and verification."""
        self.allocations: Dict[str, CUdeviceptr] = {}
        self.arena_ptr = CUdeviceptr(0)
        self.offsets: Dict[str, Tuple[int, int]] = {}
        self.current_offset: int = 0
        self.tensors_shape_dtype: Dict[str, Tuple[Tuple[int, ...], np.dtype]] = {}
        self.dynamic_allocations: Dict[str, Tuple[CUdeviceptr, int]] = {}

    def allocate_static(
        self, name: str, size_in_bytes: int, shape: Tuple[int, ...], dtype: np.dtype
    ) -> None:
        """Executes the allocate static operation."""
        self.offsets[name] = (self.current_offset, size_in_bytes)
        self.current_offset += size_in_bytes
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

    def get_tensor_ptr(self, name: str) -> CUdeviceptr:
        """Executes the get tensor ptr operation."""
        if name in self.dynamic_allocations:
            return self.dynamic_allocations[name][0]

        if name in self.offsets:
            offset, _ = self.offsets[name]
            return CUdeviceptr(self.arena_ptr.value + offset)

        raise RuntimeError(f"Tensor {name} not found in CUDA memory planner")

    def set_tensor(self, name: str, host_data: np.ndarray) -> None:
        """Executes the set tensor operation."""
        if not is_cuda_available():
            # Fallback to pure CPU storage for tests without CUDA
            if not hasattr(self, "_cpu_fallback_tensors"):
                self._cpu_fallback_tensors = {}
            self._cpu_fallback_tensors[name] = host_data
            return

        size_bytes = host_data.nbytes
        if name in self.offsets:
            offset, size = self.offsets[name]
            if size_bytes > size:
                ptr = CUdeviceptr()
                check_cuda_error(_cuda_lib.cuMemAlloc_v2(ctypes.byref(ptr), size_bytes))
                self.dynamic_allocations[name] = (ptr, size_bytes)
                dst_ptr = ptr
            else:
                dst_ptr = CUdeviceptr(self.arena_ptr.value + offset)
        else:
            if name in self.dynamic_allocations:
                ptr, size = self.dynamic_allocations[name]
                if size_bytes > size:
                    check_cuda_error(_cuda_lib.cuMemFree_v2(ptr))
                    check_cuda_error(
                        _cuda_lib.cuMemAlloc_v2(ctypes.byref(ptr), size_bytes)
                    )
                    self.dynamic_allocations[name] = (ptr, size_bytes)
                dst_ptr = ptr
            else:
                ptr = CUdeviceptr()
                check_cuda_error(_cuda_lib.cuMemAlloc_v2(ctypes.byref(ptr), size_bytes))
                self.dynamic_allocations[name] = (ptr, size_bytes)
                dst_ptr = ptr

        contiguous_data = np.ascontiguousarray(host_data)
        src_ptr = contiguous_data.ctypes.data_as(ctypes.c_void_p)
        check_cuda_error(_cuda_lib.cuMemcpyHtoD_v2(dst_ptr, src_ptr, size_bytes))

        if name not in self.offsets:
            self.tensors_shape_dtype[name] = (host_data.shape, host_data.dtype)

    def get_host_tensor(self, name: str) -> np.ndarray:
        """Executes the get host tensor operation."""
        if (
            not is_cuda_available()
            and hasattr(self, "_cpu_fallback_tensors")
            and name in self._cpu_fallback_tensors
        ):
            return self._cpu_fallback_tensors[name]

        if name not in self.offsets and name not in self.dynamic_allocations:
            raise RuntimeError(f"Tensor {name} not found.")

        if name in self.dynamic_allocations:
            ptr, size_bytes = self.dynamic_allocations[name]
            # Assumes float32 for fallback if not registered statically
            shape, dtype = self.tensors_shape_dtype.get(
                name, ((size_bytes // 4,), np.dtype("float32"))
            )
        else:
            offset, size_bytes = self.offsets[name]
            ptr = CUdeviceptr(self.arena_ptr.value + offset)
            shape, dtype = self.tensors_shape_dtype[name]

        host_data = np.empty(shape, dtype=dtype)
        dst_ptr = host_data.ctypes.data_as(ctypes.c_void_p)

        if is_cuda_available():
            check_cuda_error(_cuda_lib.cuMemcpyDtoH_v2(dst_ptr, ptr, size_bytes))

        return host_data

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
        """Provides   del   functionality and verification."""
        try:
            self.cleanup()
        except Exception as e:
            logger.debug(f"Exception during CUDAMemoryPlanner cleanup: {e}")
