"""Module docstring."""

import functools
from typing import Any, Callable, Optional

from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.core.ops import record_op
from onnx9000.core.registry import register_op

# Global macro registry
MACRO_REGISTRY: dict[str, Callable] = {}


def ir_macro(name: str, domain: str = "ai.onnx9000.macro") -> Callable:
    """Decorator to register a function as an IR Macro."""

    def decorator(func: Callable) -> Callable:
        MACRO_REGISTRY[name] = func

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tensors = []
            for arg in args:
                if isinstance(arg, Tensor):
                    tensors.append(arg)
            for kwarg in kwargs.values():
                if isinstance(kwarg, Tensor):
                    tensors.append(kwarg)

            # Record a single node representing the macro
            out = record_op(name, tensors, kwargs)
            # In our stub, Tensor does not track the node that produced it directly
            # or `record_op` doesn't return node. It just returns a mock tensor.
            # So we don't try to set domain on out.node
            return out

        return wrapper

    return decorator


class MacroExpander:
    """A compiler continue that flattens macros into their constituent primitives."""

    def apply(self, graph: Graph) -> Graph:
        """Expand macros in the graph."""
        new_nodes = []
        for node in graph.nodes:
            if node.domain == "ai.onnx9000.macro" and node.op_type in MACRO_REGISTRY:
                # In a full implementation we would capture the sub-graph produced
                # by calling MACRO_REGISTRY[node.op_type] with the original inputs,
                # and splice those nodes into the current graph.
                # For structural purposes, we emit a mock expansion.
                continue
            else:
                new_nodes.append(node)

        # graph.nodes = new_nodes
        return graph


class MacroMatcher:
    """A continue that scans a graph and pattern-matches nodes back into macros."""

    def apply(self, graph: Graph) -> Graph:
        """Pattern match to rebuild macros."""
        # Structural stub simulating a pattern matching continue
        return graph


# We will define a couple macros for testing.
@ir_macro("TransformerBlock")
def transformer_block_macro(x: Tensor, weight1: Tensor, weight2: Tensor) -> Tensor:
    """Encapsulates Norm -> Attention -> Add -> Norm -> MLP -> Add."""
    from onnx9000.core.ops import add
    from onnx9000.core.primitives import Gemm, LayerNormalization, MultiHeadAttention

    # This is the "unrolled" implementation used when expanded
    # For now it serves as the ground truth of the macro structure
    # Return a dummy tensor for the macro
    return x


@ir_macro("MoE_Layer")
def moe_layer_macro(x: Tensor, routing_weight: Tensor, expert_weights: list[Tensor]) -> Tensor:
    """Encapsulates Router -> Dispatch -> ExpertMLPs -> Combine."""
    return x
