"""Frontend Sub-Package.

Provides tracing and PyTorch-like interfaces to define and capture
computation graphs from native Python execution.
"""

import threading
from typing import Any, Optional

from onnx9000.converters.frontend.tensor import Node, Tensor


class GraphBuilder:
    """Tracks a sequence of nodes for a given graph."""

    def __init__(self, name: str = "Graph") -> None:
        """Initialize the frontend builder or trace context."""
        self.name = name
        self.nodes: list[Node] = []
        self.inputs: list[Tensor] = []
        self.outputs: list[Tensor] = []
        self.parameters: list[Tensor] = []
        self._tensor_counter = 0
        self.doc_string = ""

    def to_graph(self):
        """To Graph function logic implementation."""
        from onnx9000.core.ir import Attribute as IRAttribute
        from onnx9000.core.ir import Graph
        from onnx9000.core.ir import Node as IRNode
        from onnx9000.core.ir import Tensor as IRTensor

        g = Graph(self.name)
        g.doc_string = getattr(self, "doc_string", "")
        for t in self.inputs:
            g.inputs.append(t.name)
            if t.name not in g.tensors:
                g.add_tensor(IRTensor(t.name, tuple(t.shape) if t.shape else (), t.dtype))
        for t in self.parameters:
            g.initializers.append(t.name)
            if t.name not in g.tensors:
                g.add_tensor(
                    IRTensor(
                        t.name,
                        tuple(t.shape) if t.shape else (),
                        t.dtype,
                        is_initializer=True,
                        data=t.numpy().tobytes(),
                    )
                )
        for n in self.nodes:
            g.add_node(
                IRNode(
                    n.op_type,
                    [i.name for i in n.inputs],
                    [o.name for o in n.outputs],
                    {k: IRAttribute(k, v) for (k, v) in (n.attributes or {}).items()},
                    name=n.name,
                    domain=n.domain,
                )
            )
        for t in self.outputs:
            g.outputs.append(t.name)
            if t.name not in g.tensors:
                g.add_tensor(IRTensor(t.name, tuple(t.shape) if t.shape else (), t.dtype))
        return g

    def add_node(self, node: Node) -> None:
        """Appends a node to the execution plan."""
        self.nodes.append(node)


_tls = threading.local()


def get_active_builder() -> Optional[GraphBuilder]:
    """Retrieve the active graph builder from thread local storage."""
    return getattr(_tls, "builder", None)


class Tracing:
    """Context manager for tracing operations into a GraphBuilder."""

    def __init__(self, builder: Optional[GraphBuilder] = None) -> None:
        """Initialize the frontend builder or trace context."""
        self.builder = builder or GraphBuilder()
        self.prev_builder: Optional[GraphBuilder] = None

    def __enter__(self) -> GraphBuilder:
        """Implement the __enter__ method."""
        self.prev_builder = get_active_builder()
        _tls.builder = self.builder
        return self.builder

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Implement the __exit__ method."""
        _tls.builder = self.prev_builder
