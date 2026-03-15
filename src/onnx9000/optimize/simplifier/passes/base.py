"""Provides base.py module functionality."""

from abc import ABC, abstractmethod
from onnx9000.core.ir import Graph


class GraphPass(ABC):
    """Base class for graph optimization passes."""

    @abstractmethod
    def run(self, graph: Graph) -> bool:
        """Runs the pass on the graph. Returns True if graph was modified."""
        pass
