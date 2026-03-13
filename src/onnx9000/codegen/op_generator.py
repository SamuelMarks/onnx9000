"""
C++ Code Generation Utilities

Translates ONNX operations to equivalent C++ bindings and memory buffers.
"""

# mypy: ignore-errors
import abc  # pragma: no cover
from typing import TYPE_CHECKING  # pragma: no cover

# pragma: no cover
from onnx9000.ir import Node  # pragma: no cover

# pragma: no cover
if TYPE_CHECKING:  # pragma: no cover
    from onnx9000.codegen.generator import Generator  # pragma: no cover


# pragma: no cover
class OpGenerator(abc.ABC):  # pragma: no cover
    """
    Interface for C++ operator writers.
    """

    @abc.abstractmethod
    def generate(
        self, node: Node, generator_context: "Generator"
    ) -> str:  # pragma: no cover
        """
        Generate C++ code for the given node.

        Args:
            node: The IR Node to generate code for.
            generator_context: The `Generator` instance, providing access to
                               tensor info, variable naming, and memory arenas.

        Returns:
            A string containing the generated C++ code.
        """
        pass  # pragma: no cover
