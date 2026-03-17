"""
Context managers for building ONNX control flow operations like If and Loop.
"""

from typing import Any, Union

from onnx9000.toolkit.script.builder import GraphBuilder
from onnx9000.toolkit.script.op import If, Loop, pop_active_builder, set_active_builder
from onnx9000.toolkit.script.var import Var


class BranchContext:
    """Manages the active graph builder context for a single branch of control flow."""

    def __init__(self, builder: GraphBuilder) -> None:
        """Initializes the branch context with its dedicated GraphBuilder."""
        self.builder = builder

    def __enter__(self) -> GraphBuilder:
        """Enters the branch context, setting this builder as the active one."""
        set_active_builder(self.builder)
        return self.builder

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exits the branch context, restoring the previously active builder."""
        pop_active_builder()


class IfContextManager:
    """Class IfContextManager implementation."""

    def __init__(self, parent_builder: GraphBuilder, cond: Var, num_outputs: int = 1) -> None:
        """Initializes the If context manager with its condition and number of outputs."""
        self.parent_builder = parent_builder
        self.cond = cond
        self.num_outputs = num_outputs
        self.then_builder = GraphBuilder(name=f"{parent_builder.name}_then")
        self.else_builder = GraphBuilder(name=f"{parent_builder.name}_else")

    def Then(self) -> BranchContext:
        """Returns a branch context for defining the 'Then' block of the If statement."""
        return BranchContext(self.then_builder)

    def Else(self) -> BranchContext:
        """Returns a branch context for defining the 'Else' block of the If statement."""
        return BranchContext(self.else_builder)

    def build(self) -> Union[Var, tuple[Var, ...], None]:
        """Finalizes the If operation, embedding the branches into the parent graph."""
        with self.parent_builder:
            return If(
                self.cond,
                then_branch=self.then_builder,
                else_branch=self.else_builder,
                num_outputs=self.num_outputs,
            )


class LoopContextManager:
    """Class LoopContextManager implementation."""

    def __init__(
        self, parent_builder: GraphBuilder, max_trip_count: Var, cond: Var, num_outputs: int = 1
    ) -> None:
        """Initializes the Loop context manager with its bounds and conditions."""
        self.parent_builder = parent_builder
        self.max_trip_count = max_trip_count
        self.cond = cond
        self.num_outputs = num_outputs
        self.body_builder = GraphBuilder(name=f"{parent_builder.name}_loop_body")

    def Body(self) -> BranchContext:
        """Returns a branch context for defining the body of the Loop."""
        return BranchContext(self.body_builder)

    def build(self) -> Union[Var, tuple[Var, ...], None]:
        """Finalizes the Loop operation, embedding its body into the parent graph."""
        with self.parent_builder:
            return Loop(
                self.max_trip_count, self.cond, body=self.body_builder, num_outputs=self.num_outputs
            )
