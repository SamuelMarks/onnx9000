"""Provides the OliveModel abstraction."""

from dataclasses import dataclass

from onnx9000.core.ir import Graph


@dataclass
class OliveModel:
    """Class OliveModel implementation."""

    graph: Graph
    metadata: dict[str, int | float | str | bool | bytes | list | dict | tuple | None] | None = None

    def __post_init__(self) -> None:
        """Initialize defaults."""
        if self.metadata is None:
            self.metadata = {}
