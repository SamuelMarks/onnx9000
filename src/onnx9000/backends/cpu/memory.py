"""CPU memory planner."""

import numpy as np
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class MemoryPlanner:
    """Manages a single contiguous NumPy array (arena) for all activations."""

    def __init__(self) -> None:
        """Initialize the memory planner."""
        self.arena: np.ndarray = np.array([], dtype=np.uint8)
        self.offsets: Dict[str, Tuple[int, int]] = {}  # tensor_name -> (offset, size)
        self.current_offset: int = 0
        self.tensors_shape_dtype: Dict[str, Tuple[Tuple[int, ...], np.dtype]] = {}
        self.dynamic_tensors: Dict[str, np.ndarray] = {}

    def allocate_static(
        self, name: str, size_in_bytes: int, shape: Tuple[int, ...], dtype: np.dtype
    ) -> None:
        """Reserve space in the static arena."""
        self.offsets[name] = (self.current_offset, size_in_bytes)
        self.current_offset += size_in_bytes
        self.tensors_shape_dtype[name] = (shape, dtype)

    def build_arena(self) -> None:
        """Allocate the actual underlying numpy array based on total required size."""
        self.arena = np.empty((self.current_offset,), dtype=np.uint8)

    def _reallocate_arena(self, new_size: int) -> None:
        """Fallback mechanism if arena size is insufficient."""
        logger.warning(
            f"Reallocating arena from {self.arena.nbytes} to {new_size} bytes"
        )
        new_arena = np.empty((new_size,), dtype=np.uint8)
        new_arena[: self.arena.nbytes] = self.arena
        self.arena = new_arena

    def get_tensor(self, name: str) -> np.ndarray:
        """Get a tensor view from the arena or dynamic storage."""
        if name in self.dynamic_tensors:
            return self.dynamic_tensors[name]

        offset, size = self.offsets[name]
        shape, dtype = self.tensors_shape_dtype[name]
        if offset + size > self.arena.nbytes:
            raise RuntimeError(f"Tensor {name} out of bounds for arena.")

        # Slicing the arena
        view = self.arena[offset : offset + size].view(dtype=dtype).reshape(shape)
        return view

    def set_tensor(self, name: str, data: np.ndarray) -> None:
        """Store a tensor. For dynamic shapes, handle fallback."""
        if name in self.offsets:
            offset, size = self.offsets[name]
            if data.nbytes > size:
                # Fallback to dynamic allocation
                self.dynamic_tensors[name] = data
            else:
                shape, dtype = self.tensors_shape_dtype[name]
                if data.shape != shape or data.dtype != dtype:
                    # Shape or dtype changed dynamically, fallback to dynamic
                    self.dynamic_tensors[name] = data
                else:
                    view = (
                        self.arena[offset : offset + data.nbytes]
                        .view(dtype=data.dtype)
                        .reshape(data.shape)
                    )
                    np.copyto(view, data)
        else:
            self.dynamic_tensors[name] = data
