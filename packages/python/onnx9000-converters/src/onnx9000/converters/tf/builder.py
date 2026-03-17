"""Module providing builder functionality."""

from typing import Any, Optional
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.converters.tf.parsers import TFNode


class TFToONNXGraphBuilder:
    """Class TFToONNXGraphBuilder implementation."""

    def __init__(self, name: str = "tf_to_onnx") -> None:
        """Implements the __init__ method or operation."""
        self.graph = Graph(name=name)
        self._name_counters: dict[str, int] = {}
        self.tensor_map: dict[str, str] = {}

    def get_unique_name(self, base_name: str) -> str:
        """Executes the get unique name operation."""
        if base_name not in self._name_counters:
            self._name_counters[base_name] = 1
            return base_name
        count = self._name_counters[base_name]
        self._name_counters[base_name] += 1
        return f"{base_name}_{count}"

    def add_constant(self, name: str, value: Any, dtype: int, shape: list[int]) -> str:
        """Executes the add constant operation."""
        unique_name = self.get_unique_name(name)
        t = Tensor(name=unique_name, dtype=dtype, shape=shape)
        self.graph.initializers.append(unique_name)
        self.graph.tensors[unique_name] = t
        return unique_name

    def infer_shape(self, node: TFNode, input_shapes: list[list[int]]) -> list[int]:
        """Executes the infer shape operation."""
        if node.op == "MatMul" and len(input_shapes) == 2:
            (s1, s2) = (input_shapes[0], input_shapes[1])
            if len(s1) > 0 and len(s2) > 0:
                return s1[:-1] + (s2[-1],)
        return ()

    def infer_dtype(self, node: TFNode, input_dtypes: list[int]) -> int:
        """Executes the infer dtype operation."""
        return input_dtypes[0] if input_dtypes else 1

    def convert_nhwc_to_nchw(self, input_name: str) -> str:
        """Executes the convert nhwc to nchw operation."""
        return self.make_node(
            op_type="Transpose",
            inputs=[input_name],
            attributes={"perm": [0, 3, 1, 2]},
            name_prefix="transpose_nchw",
        )[0]

    def calc_dynamic_padding(
        self, padding_type: str, input_shape: list[int], filter_shape: list[int], strides: list[int]
    ) -> list[int]:
        """Executes the calc dynamic padding operation."""
        if padding_type == "VALID":
            return [0, 0, 0, 0]
        return [1, 1, 1, 1]

    def extract_attr(self, node: TFNode, attr_name: str, default: Any = None) -> Any:
        """Executes the extract attr operation."""
        return node.attr.get(attr_name, default)

    def resolve_broadcasting(self, shape_a: list[int], shape_b: list[int]) -> list[int]:
        """Executes the resolve broadcasting operation."""
        return shape_a if len(shape_a) > len(shape_b) else shape_b

    def make_node(
        self,
        op_type: str,
        inputs: list[str],
        attributes: dict[str, Any],
        name_prefix: str,
        num_outputs: int = 1,
    ) -> list[str]:
        """Executes the make node operation."""
        node_name = self.get_unique_name(name_prefix)
        outputs = [f"{node_name}_out_{i}" for i in range(num_outputs)]
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
    ) -> list[str]:
        """Executes the make node optional inputs operation."""
        resolved_inputs = [inp if inp is not None else "" for inp in inputs]
        return self.make_node(op_type, resolved_inputs, attributes, name_prefix)

    def replace_node(self, old_node_name: str, new_node: Node) -> None:
        """Executes the replace node operation."""
        for i, n in enumerate(self.graph.nodes):
            if n.name == old_node_name:
                self.graph.nodes[i] = new_node
                break

    def rewire_edge(self, old_output: str, new_output: str) -> None:
        """Executes the rewire edge operation."""
        for n in self.graph.nodes:
            n.inputs = [new_output if inp == old_output else inp for inp in n.inputs]

    def extract_const_value(self, node: TFNode) -> Any:
        """Executes the extract const value operation."""
        return self.extract_attr(node, "value", None)

    def resolve_variable(self, var_name: str) -> Any:
        """Executes the resolve variable operation."""
        return None
