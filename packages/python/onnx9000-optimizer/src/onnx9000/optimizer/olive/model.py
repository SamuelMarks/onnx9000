"""Provides the OliveModel abstraction."""

from dataclasses import dataclass
from typing import Any, Optional

from onnx9000.core.ir import Graph


@dataclass
class OliveModel:
    """Class OliveModel implementation."""

    graph: Graph
    metadata: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Initialize defaults."""
        if self.metadata is None:
            self.metadata = {}
