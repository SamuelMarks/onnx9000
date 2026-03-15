"""
C++ Code Generation Utilities

Translates ONNX operations to equivalent C++ bindings and memory buffers.
"""

# mypy: ignore-errors
import abc
from typing import TYPE_CHECKING


from onnx9000.core.ir import Node


if TYPE_CHECKING:
    from onnx9000.backends.codegen.generator import Generator


class OpGenerator(abc.ABC):
    """
    Interface for C++ operator writers.
    """

    @abc.abstractmethod
    def generate(self, node: Node, generator_context: "Generator") -> str:
        """
        Generate C++ code for the given node.

        Args:
            node: The IR Node to generate code for.
            generator_context: The `Generator` instance, providing access to
                               tensor info, variable naming, and memory arenas.

        Returns:
            A string containing the generated C++ code.
        """
        pass
