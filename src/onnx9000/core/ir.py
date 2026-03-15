"""Provide core functionality and interfaces for this module."""

import logging

logger = logging.getLogger(__name__)
"""Module providing core logic and structural definitions."""

from typing import Any, Optional, Union

import numpy as np

from onnx9000.core.dtypes import DType


class DynamicDim:
    """Represents a dynamic dimension in a tensor shape (e.g., 'batch_size' or -1)."""

    def __init__(self, value: Union[str, int]) -> None:
        """Provides   init   functionality and verification."""

        self.value = value

    def __repr__(self) -> str:
        """Provides   repr   functionality and verification."""

        return f"DynamicDim({self.value})"

    def __str__(self) -> str:
        """Provides   str   functionality and verification."""
        return str(self.value)

    def __eq__(self, other: Any) -> bool:
        """Provides   eq   functionality and verification."""

        if isinstance(other, DynamicDim):
            return self.value == other.value
        return False


class Tensor:
    """Internal Representation of a Tensor, optimized for C++ mapping and memory planning."""

    def __init__(
        self,
        name: str,
        shape: tuple[Union[int, DynamicDim], ...],
        dtype: DType,
        is_initializer: bool = False,
        requires_grad: bool = True,
        data: Optional[np.ndarray] = None,
    ) -> None:
        """Provides   init   functionality and verification."""

        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.is_initializer = is_initializer
        self.requires_grad = requires_grad
        self.data = data

        # Memory planning
        self.buffer_id: Optional[int] = None
        self.lifespan: tuple[int, int] = (-1, -1)  # (first_use_idx, last_use_idx)

    def __repr__(self) -> str:
        """Provides   repr   functionality and verification."""

        return f"ir.Tensor(name={self.name}, shape={self.shape}, dtype={self.dtype}, buf={self.buffer_id})"


class Node:
    """Internal Representation of an operation."""

    def __init__(
        self,
        op_type: str,
        inputs: list[str],
        outputs: list[str],
        attributes: dict[str, Any],
        name: str = "",
    ) -> None:
        """Provides   init   functionality and verification."""

        self.op_type = op_type
        self.inputs = inputs
        self.outputs = outputs
        self.attributes = attributes
        self.name = name

    def __repr__(self) -> str:
        """Provides   repr   functionality and verification."""

        return f"ir.Node({self.op_type}, {self.inputs} -> {self.outputs})"


class Graph:
    """Internal Representation of a complete topological execution plan."""

    def __init__(self, name: str) -> None:
        """Provides   init   functionality and verification."""

        self.name = name
        self.nodes: list[Node] = []
        self.tensors: dict[str, Tensor] = {}
        self.inputs: list[str] = []
        self.outputs: list[str] = []
        self.initializers: list[str] = []

    def add_tensor(self, tensor: Tensor) -> None:
        """Provides add tensor functionality and verification."""

        self.tensors[tensor.name] = tensor

    def add_node(self, node: Node) -> None:
        """Provides add node functionality and verification."""

        self.nodes.append(node)

    def print_visualizer(self) -> None:
        """Simple ASCII visualization of the graph."""
        logger.info(f"=== Graph: {self.name} ===")
        logger.info(f"Inputs: {self.inputs}")
        logger.info(f"Outputs: {self.outputs}")
        logger.info("Nodes:")
        for idx, node in enumerate(self.nodes):
            logger.info(f"  [{idx}] {node.op_type}: {node.inputs} -> {node.outputs}")
        logger.info("=========================")
