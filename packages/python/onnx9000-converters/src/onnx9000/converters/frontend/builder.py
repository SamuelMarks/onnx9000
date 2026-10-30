"""Frontend Sub-Package.

Provides tracing and PyTorch-like interfaces to define and capture
computation graphs from native Python execution.
"""

import threading

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

    def to_graph(self) -> int | float | str | bool | bytes | list | dict | tuple | None:
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
                data = t.numpy()
                if data is None:
                    import numpy as np

                    data = np.zeros(tuple(t.shape) if t.shape else (), dtype=np.float32)
                g.add_tensor(
                    IRTensor(
                        t.name,
                        tuple(t.shape) if t.shape else (),
                        t.dtype,
                        is_initializer=True,
                        data=data.tobytes(),
                    )
                )
        for n in self.nodes:
            g.add_node(
                IRNode(
                    n.op_type,
                    [i.name for i in n.inputs],
                    [o.name for o in n.outputs],
                    {k: IRAttribute(k, value=v) for (k, v) in (n.attributes or {}).items()},
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
        """Append a node to the execution plan."""
        self.nodes.append(node)


_tls = threading.local()


def get_active_builder() -> GraphBuilder | None:
    """Retrieve the active graph builder from thread local storage."""
    return getattr(_tls, "builder", None)


class Tracing:
    """Context manager for tracing operations into a GraphBuilder."""

    def __init__(self, builder: GraphBuilder | None = None) -> None:
        """Initialize the frontend builder or trace context."""
        self.builder = builder or GraphBuilder()
        self.prev_builder: GraphBuilder | None = None

    def __enter__(self) -> GraphBuilder:
        """Enter the runtime context managed by Tracing.

        Returns:
            Result of enter (GraphBuilder).
        """
        self.prev_builder = get_active_builder()
        _tls.builder = self.builder
        return self.builder

    def __exit__(
        self,
        exc_type: int | float | str | bool | bytes | list | dict | tuple | None,
        exc_val: int | float | str | bool | bytes | list | dict | tuple | None,
        exc_tb: int | float | str | bool | bytes | list | dict | tuple | None,
    ) -> None:
        """Exit the runtime context managed by Tracing and handle exceptions.

        Args:
            exc_type: Parameter specifying exc type (Union[int, float, str, bool, bytes, list, dict, tuple, None]).
            exc_val: Parameter specifying exc val (Union[int, float, str, bool, bytes, list, dict, tuple, None]).
            exc_tb: Parameter specifying exc tb (Union[int, float, str, bool, bytes, list, dict, tuple, None]).
        """
        _tls.builder = self.prev_builder
