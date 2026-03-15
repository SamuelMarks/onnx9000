"""
Base Vector-Jacobian Product (VJP) Rules

This module provides the abstract base class for defining VJP rules,
which form the core of automatic differentiation by describing how
gradients are propagated backward through individual ONNX operators.
"""

import abc

from onnx9000.core.ir import Node


class VJPRule(abc.ABC):
    """
    Base class for Vector-Jacobian Product rules.
    Defines how gradients propagate backwards through an operation.
    """

    @abc.abstractmethod
    def build_backward_nodes(
        self, fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """
        Given a forward node and the names of the incoming gradient tensors (dL/dOut),
        returns a list of new nodes to compute gradients for the inputs,
        and a list of names for those input gradients (dL/dIn).

        Args:
            fwd_node (Node): The forward pass operation node.
            grad_outputs (list[str]): Names of the incoming gradient tensors.

        Returns:
            tuple[list[Node], list[str]]: A tuple containing:
                - List of nodes computing the backward pass.
                - List of names corresponding to the gradients with respect to the node inputs.
        """
        return [], []
