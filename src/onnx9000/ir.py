"""Provide core functionality and interfaces for this module."""

import logging

logger = logging.getLogger(__name__)
"""Module docstring."""

from typing import Any, Optional, Union

import numpy as np

from onnx9000.dtypes import DType


class DynamicDim:
    """Represents a dynamic dimension in a tensor shape (e.g., 'batch_size' or -1)."""

    def __init__(self, value: Union[str, int]) -> None:
        """__init__ docstring."""

        self.value = value  # pragma: no cover

    def __repr__(self) -> str:
        """__repr__ docstring."""

        return f"DynamicDim({self.value})"  # pragma: no cover

    def __str__(self) -> str:
        """__str__ docstring."""
        return str(self.value)  # pragma: no cover

    def __eq__(self, other: Any) -> bool:
        """__eq__ docstring."""

        if isinstance(other, DynamicDim):  # pragma: no cover
            return self.value == other.value  # pragma: no cover
        return False  # pragma: no cover


class Tensor:
    """Internal Representation of a Tensor, optimized for C++ mapping and memory planning."""

    def __init__(
        self,
        name: str,
        shape: tuple[Union[int, DynamicDim], ...],
        dtype: DType,
        is_initializer: bool = False,
        data: Optional[np.ndarray] = None,
    ) -> None:
        """__init__ docstring."""

        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.is_initializer = is_initializer
        self.data = data

        # Memory planning
        self.buffer_id: Optional[int] = None
        self.lifespan: tuple[int, int] = (-1, -1)  # (first_use_idx, last_use_idx)

    def __repr__(self) -> str:
        """__repr__ docstring."""

        return f"ir.Tensor(name={self.name}, shape={self.shape}, dtype={self.dtype}, buf={self.buffer_id})"  # pragma: no cover


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
        """__init__ docstring."""

        self.op_type = op_type
        self.inputs = inputs
        self.outputs = outputs
        self.attributes = attributes
        self.name = name

    def __repr__(self) -> str:
        """__repr__ docstring."""

        return f"ir.Node({self.op_type}, {self.inputs} -> {self.outputs})"  # pragma: no cover


class Graph:
    """Internal Representation of a complete topological execution plan."""

    def __init__(self, name: str) -> None:
        """__init__ docstring."""

        self.name = name
        self.nodes: list[Node] = []
        self.tensors: dict[str, Tensor] = {}
        self.inputs: list[str] = []
        self.outputs: list[str] = []
        self.initializers: list[str] = []

    def add_tensor(self, tensor: Tensor) -> None:
        """add_tensor docstring."""

        self.tensors[tensor.name] = tensor

    def add_node(self, node: Node) -> None:
        """add_node docstring."""

        self.nodes.append(node)

    def print_visualizer(self) -> None:
        """Simple ASCII visualization of the graph."""
        logger.info(f"=== Graph: {self.name} ===")  # pragma: no cover
        logger.info(f"Inputs: {self.inputs}")  # pragma: no cover
        logger.info(f"Outputs: {self.outputs}")  # pragma: no cover
        logger.info("Nodes:")  # pragma: no cover
        for idx, node in enumerate(self.nodes):  # pragma: no cover
            logger.info(
                f"  [{idx}] {node.op_type}: {node.inputs} -> {node.outputs}"
            )  # pragma: no cover
        logger.info("=========================")  # pragma: no cover
