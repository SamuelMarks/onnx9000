"""C++ Code Generation Utilities.

Translates ONNX operations to equivalent C++ bindings and memory buffers.
"""

import abc
from typing import TYPE_CHECKING

from onnx9000.core.ir import Node


class OpGenerator(abc.ABC):
    """Interface for C++ operator writers."""

    @abc.abstractmethod
    def generate(self, node: Node, generator_context: "onnx9000.backends.codegen.Generator") -> str:
        """Generate C++ code for the given node.

        Args:
            node: The IR Node to generate code for.
            generator_context: The `Generator` instance, providing access to
                               tensor info, variable naming, and memory arenas.

        Returns:
            A string containing the generated C++ code.

        """
        return ""
