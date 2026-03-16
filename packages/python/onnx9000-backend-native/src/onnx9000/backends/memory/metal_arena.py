"""Apple Metal Performance Shaders (MPS) Memory Planner and Pool."""

import ctypes
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class MetalMemoryPlanner:
    """Manages memory allocations for the Apple Metal MPS backend."""

    def __init__(self) -> None:
        """Initialize the Metal memory planner."""
        self.allocations: dict[str, ctypes.c_void_p] = {}
        self.arena_ptr: Optional[ctypes.c_void_p] = None
        self.offsets: dict[str, tuple[int, int]] = {}
        self.current_offset: int = 0
        self.tensors_shape_dtype: dict[str, tuple[tuple[int, ...], str]] = {}
        self.dynamic_allocations: dict[str, tuple[ctypes.c_void_p, int]] = {}

    def allocate_static(
        self, name: str, size_in_bytes: int, shape: tuple[int, ...], dtype: str
    ) -> None:
        """Reserve a static block of memory in the Metal buffer."""
        self.offsets[name] = (self.current_offset, size_in_bytes)
        self.current_offset += size_in_bytes
        self.tensors_shape_dtype[name] = (shape, dtype)

    def build_arena(self) -> None:
        """Allocate the shared Metal buffer via ctypes to Metal framework."""
        if self.current_offset > 0:
            logger.info(f"Mocking Metal Buffer allocation: {self.current_offset} bytes")
            buf = bytearray(self.current_offset)
            buffer_info = (ctypes.c_char * len(buf)).from_buffer(buf)
            self._keep_alive = buf
            self.arena_ptr = ctypes.c_void_p(ctypes.addressof(buffer_info))

    def get_tensor_ptr(self, name: str) -> ctypes.c_void_p:
        """Retrieve the Metal buffer pointer for a given tensor."""
        if name in self.dynamic_allocations:
            return self.dynamic_allocations[name][0]
        if name in self.offsets and self.arena_ptr is not None:
            (offset, _) = self.offsets[name]
            addr = self.arena_ptr.value + offset if self.arena_ptr.value else 0
            return ctypes.c_void_p(addr)
        raise RuntimeError(f"Tensor {name} not found in Metal memory planner")

    def set_tensor(
        self, name: str, host_data: memoryview, shape: tuple[int, ...], dtype: str
    ) -> None:
        """Copy data from host memoryview to Metal Buffer."""
        size_bytes = len(host_data)
        if name not in self.tensors_shape_dtype:
            self.tensors_shape_dtype[name] = (shape, dtype)
        if name in self.offsets and self.arena_ptr is not None:
            (offset, size) = self.offsets[name]
            if size_bytes > size:
                buf = bytearray(host_data)
                buffer_info = (ctypes.c_char * len(buf)).from_buffer(buf)
                if not hasattr(self, "_dynamic_keep_alive"):
                    self._dynamic_keep_alive = []
                self._dynamic_keep_alive.append(buf)
                ptr = ctypes.addressof(buffer_info)
                self.dynamic_allocations[name] = (ctypes.c_void_p(ptr), size_bytes)
            else:
                dst_ptr = self.arena_ptr.value + offset if self.arena_ptr.value else 0
                ctypes.memmove(dst_ptr, host_data.tobytes(), size_bytes)
        else:
            buf = bytearray(host_data)
            buffer_info = (ctypes.c_char * len(buf)).from_buffer(buf)
            if not hasattr(self, "_dynamic_keep_alive"):
                self._dynamic_keep_alive = []
            self._dynamic_keep_alive.append(buf)
            ptr = ctypes.addressof(buffer_info)
            self.dynamic_allocations[name] = (ctypes.c_void_p(ptr), size_bytes)

    def get_host_tensor(self, name: str) -> memoryview:
        """Copy data from Metal Buffer back to host memoryview."""
        if name in self.dynamic_allocations:
            (ptr, size_bytes) = self.dynamic_allocations[name]
            buf = (ctypes.c_char * size_bytes).from_address(ptr.value if ptr.value else 0)
            return memoryview(buf)
        if name in self.offsets and self.arena_ptr is not None:
            (offset, size_bytes) = self.offsets[name]
            addr = self.arena_ptr.value + offset if self.arena_ptr.value else 0
            buf = (ctypes.c_char * size_bytes).from_address(addr)
            return memoryview(buf)
        raise RuntimeError(f"Tensor {name} not found.")

    def cleanup(self) -> None:
        """Free all explicitly allocated Metal buffers."""
        self.arena_ptr = None
        self.dynamic_allocations.clear()
        if hasattr(self, "_keep_alive"):
            del self._keep_alive
        if hasattr(self, "_dynamic_keep_alive"):
            self._dynamic_keep_alive.clear()

    def __del__(self) -> None:
        """Ensure cleanup on destruction."""
        self.cleanup()
