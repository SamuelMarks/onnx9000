"""Module docstring."""

from typing import Any, Optional
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.frontend.paddle.parsers import PaddleNode


class PaddleToONNXGraphBuilder:
    """Docstring for PaddleToONNXGraphBuilder."""

    def __init__(self, name: str = "paddle_to_onnx") -> None:
        """Initialize."""
        self.graph = Graph(name=name)
        self._name_counters: dict[str, int] = {}
        self.tensor_map: dict[str, str] = {}

    def get_unique_name(self, base_name: str) -> str:
        """Docstring for get_unique_name."""
        if base_name not in self._name_counters:
            self._name_counters[base_name] = 1
            return base_name
        count = self._name_counters[base_name]
        self._name_counters[base_name] += 1
        return f"{base_name}_{count}"

    def add_constant(self, name: str, value: Any, dtype: int, shape: tuple[int, ...]) -> str:
        """Docstring for add_constant."""
        unique_name = self.get_unique_name(name)
        t = Tensor(name=unique_name, dtype=dtype, shape=shape)
        self.graph.initializers.append(unique_name)
        self.graph.tensors[unique_name] = t
        return unique_name

    def infer_shape(self, node: PaddleNode, input_shapes: list[tuple[int, ...]]) -> tuple[int, ...]:
        """Docstring for infer_shape."""
        if node.op_type == "matmul_v2" and len(input_shapes) == 2:
            (s1, s2) = (input_shapes[0], input_shapes[1])
            if len(s1) > 0 and len(s2) > 0:
                return s1[:-1] + (s2[-1],)
        return ()

    def extract_attr(self, node: PaddleNode, attr_name: str, default: Any = None) -> Any:
        """Docstring for extract_attr."""
        return node.attrs.get(attr_name, default)

    def extract_list_attr(
        self, node: PaddleNode, attr_name: str, default: list[Any] = None
    ) -> list[Any]:
        """Docstring for extract_list_attr."""
        if default is None:
            default = []
        val = node.attrs.get(attr_name, default)
        if not isinstance(val, list):
            return [val]
        return val

    def resolve_broadcasting(
        self, shape_a: tuple[int, ...], shape_b: tuple[int, ...]
    ) -> tuple[int, ...]:
        """Docstring for resolve_broadcasting."""
        return shape_a if len(shape_a) > len(shape_b) else shape_b

    def flatten_lod_tensor(self, input_name: str, output_name: str) -> list[str]:
        """Docstring for flatten_lod_tensor."""
        return self.make_node("Flatten", [input_name], {}, output_name, outputs=[output_name])

    def make_node(
        self,
        op_type: str,
        inputs: list[str],
        attributes: dict[str, Any],
        name_prefix: str,
        outputs: list[str] = None,
    ) -> list[str]:
        """Docstring for make_node."""
        node_name = self.get_unique_name(name_prefix)
        if not outputs:
            outputs = [f"{node_name}_out_0"]
        else:
            outputs = list(outputs)
        n = Node(
            op_type=op_type, inputs=inputs, outputs=outputs, attributes=attributes, name=node_name
        )
        self.graph.nodes.append(n)
        for out in outputs:
            self.graph.tensors[out] = Tensor(name=out, dtype=1, shape=())
        return outputs

    def make_node_optional_inputs(
        self,
        op_type: str,
        inputs: list[Optional[str]],
        attributes: dict[str, Any],
        name_prefix: str,
        outputs: list[str] = None,
    ) -> list[str]:
        """Docstring for make_node_optional_inputs."""
        resolved_inputs = [inp if inp is not None else "" for inp in inputs]
        return self.make_node(op_type, resolved_inputs, attributes, name_prefix, outputs)

    def replace_node(self, old_node_name: str, new_node: Node) -> None:
        """Docstring for replace_node."""
        for i, n in enumerate(self.graph.nodes):
            if n.name == old_node_name:
                self.graph.nodes[i] = new_node
                break

    def rewire_edge(self, old_output: str, new_output: str) -> None:
        """Docstring for rewire_edge."""
        for n in self.graph.nodes:
            n.inputs = [new_output if inp == old_output else inp for inp in n.inputs]

    def resolve_variable(self, var_name: str) -> Any:
        """Docstring for resolve_variable."""
        return None

    def dump_ir(self) -> str:
        """Docstring for dump_ir."""
        return f"PaddleGraph IR Dump: {len(self.graph.nodes)} nodes."
