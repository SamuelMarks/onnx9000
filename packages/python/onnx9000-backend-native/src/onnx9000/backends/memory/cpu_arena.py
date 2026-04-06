"""CPU Memory Planner and Pool for generic host execution."""

import ctypes
import logging
import math
import mmap
import platform
import threading
from typing import Optional

logger = logging.getLogger(__name__)
MAP_ANONYMOUS = 32
MAP_PRIVATE = 2
MADV_HUGEPAGE = 14


class CPUMemoryPlanner:
    """Manages contiguous memory allocations on the CPU host."""

    def __init__(self) -> None:
        """Initialize the CPU memory planner."""
        self.arena_mmap: Optional[mmap.mmap] = None
        self.offsets: dict[str, tuple[int, int]] = {}
        self.current_offset: int = 0
        self.tensors_shape_dtype: dict[str, tuple[tuple[int, ...], str]] = {}
        self.dynamic_allocations: dict[str, mmap.mmap] = {}
        self.dynamic_sizes: dict[str, int] = {}
        self.alignment: int = 64
        self._lock = threading.RLock()
        self.ref_counts: dict[str, int] = {}
        self._mmap_ptrs: list[int] = []

    def allocate_static(
        self, name: str, size_in_bytes: int, shape: tuple[int, ...], dtype: str
    ) -> None:
        """Reserve a static block of memory in the arena."""
        with self._lock:
            aligned_offset = math.ceil(self.current_offset / self.alignment) * self.alignment
            self.offsets[name] = (aligned_offset, size_in_bytes)
            self.current_offset = aligned_offset + size_in_bytes
            self.tensors_shape_dtype[name] = (shape, dtype)
            self.ref_counts[name] = 1

    def _allocate_mmap(self, size: int) -> mmap.mmap:
        """Allocate memory explicitly using anonymous mmap (MAP_ANONYMOUS | MAP_PRIVATE)."""
        size = math.ceil(size / 4096) * 4096
        if size == 0:
            size = 4096
        try:
            if platform.system() == "Windows":
                mm = mmap.mmap(-1, size)
            else:
                mm = mmap.mmap(-1, size, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
            if platform.system() == "Linux" and size >= 2 * 1024 * 1024:
                try:
                    libc = ctypes.CDLL("libc.so.6")
                    libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
                    buf_ptr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
                    libc.madvise(ctypes.c_void_p(buf_ptr), size, MADV_HUGEPAGE)
                except Exception as e:
                    logger.debug(f"madvise MADV_HUGEPAGE failed: {e}")
            if platform.system() != "Windows":
                try:
                    libc = ctypes.CDLL(
                        "libc.so.6" if platform.system() == "Linux" else "libc.dylib"
                    )
                    libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
                    buf_ptr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
                    libc.mlock(ctypes.c_void_p(buf_ptr), size)
                except Exception as e:
                    logger.debug(f"mlock failed: {e}")
            buf_ptr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
            self._mmap_ptrs.append(buf_ptr)
            return mm
        except OSError as e:
            raise MemoryError(f"Failed to allocate {size} bytes: {e}")

    def build_arena(self) -> None:
        """Allocate the entire static arena block."""
        with self._lock:
            if self.current_offset > 0:
                aligned_total = math.ceil(self.current_offset / self.alignment) * self.alignment
                self.arena_mmap = self._allocate_mmap(aligned_total)

    def get_tensor_ptr(self, name: str) -> ctypes.c_void_p:
        """Retrieve the memory pointer for a given tensor."""
        with self._lock:
            if name in self.dynamic_allocations:
                mm = self.dynamic_allocations[name]
                addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
                return ctypes.c_void_p(addr)
            if name in self.offsets and self.arena_mmap is not None:
                (offset, _) = self.offsets[name]
                addr = ctypes.addressof(ctypes.c_char.from_buffer(self.arena_mmap)) + offset
                return ctypes.c_void_p(addr)
            raise RuntimeError(f"Tensor {name} not found in CPU memory planner")

    def set_tensor(
        self, name: str, host_data: memoryview, shape: tuple[int, ...], dtype: str
    ) -> None:
        """Write raw memory view data into the CPU arena."""
        with self._lock:
            size_bytes = len(host_data)
            if name in self.offsets and self.arena_mmap is not None:
                (offset, size) = self.offsets[name]
                if size_bytes > size:
                    mm = self._allocate_mmap(size_bytes)
                    mm[:size_bytes] = host_data
                    self.dynamic_allocations[name] = mm
                    self.dynamic_sizes[name] = size_bytes
                else:
                    self.arena_mmap[offset : offset + size_bytes] = host_data
            else:
                mm = self._allocate_mmap(size_bytes)
                mm[:size_bytes] = host_data
                self.dynamic_allocations[name] = mm
                self.dynamic_sizes[name] = size_bytes
            if name not in self.tensors_shape_dtype:
                self.tensors_shape_dtype[name] = (shape, dtype)

    def get_host_tensor(self, name: str) -> memoryview:
        """Retrieve the raw data from the arena."""
        with self._lock:
            if name in self.dynamic_allocations:
                size_bytes = self.dynamic_sizes[name]
                return memoryview(self.dynamic_allocations[name])[:size_bytes]
            if name in self.offsets and self.arena_mmap is not None:
                (offset, size_bytes) = self.offsets[name]
                return memoryview(self.arena_mmap)[offset : offset + size_bytes]
            raise RuntimeError(f"Tensor {name} not found.")

    def add_ref(self, name: str) -> None:
        """Execute the add ref operation."""
        with self._lock:
            if name in self.ref_counts:
                self.ref_counts[name] += 1

    def release_ref(self, name: str) -> None:
        """Execute the release ref operation."""
        with self._lock:
            if name in self.ref_counts:
                self.ref_counts[name] -= 1
                if self.ref_counts[name] == 0 and name in self.dynamic_allocations:
                    self.dynamic_allocations[name].close()
                    del self.dynamic_allocations[name]
                    if name in self.dynamic_sizes:
                        del self.dynamic_sizes[name]

    def cleanup(self) -> None:
        """Free all explicitly allocated memory."""
        with self._lock:
            if self.arena_mmap is not None:
                try:
                    self.arena_mmap.close()
                except BufferError:
                    _ignore = True
                self.arena_mmap = None
            for mm in self.dynamic_allocations.values():
                try:
                    mm.close()
                except BufferError:
                    _ignore = True
            self.dynamic_allocations.clear()
            self.dynamic_sizes.clear()

    def __del__(self) -> None:
        """Ensure cleanup on destruction."""
        self.cleanup()

    def allocate_dynamic(self, name: str, size: int, shape: tuple[int, ...], dtype: str) -> None:
        """Allocate an independent block for dynamically shaped tensors."""
        with self._lock:
            if name in self.dynamic_allocations:
                self.release_ref(name)
            self.dynamic_allocations[name] = self._allocate_mmap(size)
            self.dynamic_sizes[name] = size
            self.tensors_shape_dtype[name] = (shape, dtype)
            self.ref_counts[name] = 1
