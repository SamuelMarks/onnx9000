"""Base abstractions for ONNX Graph Optimizations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from onnx9000.core.ir import Graph


@dataclass
class PassContext:
    """Tracks state during graph optimization passes."""

    pass_name: str
    nodes_removed: int = 0
    nodes_added: int = 0
    tensors_removed: int = 0
    tensors_added: int = 0
    modifications: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def log_change(self, msg: str) -> None:
        """Log a specific structural change."""
        self.modifications.append(msg)


class Pass(ABC):
    """Abstract base class for all Graph Passes (Surgeon, Simplifier, Olive, etc.)."""

    def __init__(self, name: str) -> None:
        """Initialize the optimization pass."""
        self.name = name

    @abstractmethod
    def run(self, graph: Graph) -> PassContext:
        """Execute the optimization pass directly on the graph (mutates in-place).

        Returns a PassContext detailing what was changed.
        """
        return PassContext(self.name)

    def __repr__(self) -> str:
        """Return the pass name."""
        return f"OptimizationPass({self.name})"
