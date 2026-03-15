"""
Frontend Sub-Package

Provides tracing and PyTorch-like interfaces to define and capture
computation graphs from native Python execution.
"""

import threading
from typing import Any, Optional

from onnx9000.frontends.frontend.tensor import Node, Tensor


class GraphBuilder:
    """Tracks a sequence of nodes for a given graph."""

    def __init__(self, name: str = "Graph") -> None:
        """Initializes the frontend builder or trace context."""

        self.name = name
        self.nodes: list[Node] = []
        self.inputs: list[Tensor] = []
        self.outputs: list[Tensor] = []
        self.parameters: list[Tensor] = []
        self._tensor_counter = 0

    def add_node(self, node: Node) -> None:
        """Appends a node to the execution plan."""
        self.nodes.append(node)


# Thread-local storage for the active GraphBuilder
_tls = threading.local()


def get_active_builder() -> Optional[GraphBuilder]:
    """Retrieves the active graph builder from thread local storage."""
    return getattr(_tls, "builder", None)


class Tracing:
    """Context manager for tracing operations into a GraphBuilder."""

    def __init__(self, builder: Optional[GraphBuilder] = None) -> None:
        """Initializes the frontend builder or trace context."""

        self.builder = builder or GraphBuilder()
        self.prev_builder: Optional[GraphBuilder] = None

    def __enter__(self) -> GraphBuilder:
        """Provides   enter   functionality and verification."""

        self.prev_builder = get_active_builder()
        _tls.builder = self.builder
        return self.builder

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Provides   exit   functionality and verification."""

        _tls.builder = self.prev_builder
