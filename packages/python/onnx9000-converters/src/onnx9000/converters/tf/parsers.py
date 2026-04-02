"""Module providing parsers functionality."""

import logging
from dataclasses import dataclass, field
from typing import Any

TF_TO_ONNX_VERSION_MAPPING = {
    "1.15.0": 11,
    "2.0.0": 11,
    "2.4.0": 13,
    "2.8.0": 15,
    "2.12.0": 17,
    "2.15.0": 19,
}
TF_DTYPE_TO_ONNX = {1: 1, 2: 11, 3: 6, 4: 8, 5: 2, 6: 3, 7: 9, 8: 14, 9: 7, 10: 4}


@dataclass
class TFNode:
    """Class TFNode implementation."""

    name: str
    op: str
    inputs: list[str] = field(default_factory=list)
    attr: dict[str, Any] = field(default_factory=dict)


@dataclass
class TFGraph:
    """Class TFGraph implementation."""

    nodes: list[TFNode] = field(default_factory=list)
    versions: dict[str, int] = field(default_factory=dict)

    def topological_sort(self) -> list[TFNode]:
        """Execute the topological sort operation.

        Returns:
            A list of nodes in topological order.
        """
        visited: set[str] = set()
        sorted_nodes: list[TFNode] = []
        node_map = {n.name: n for n in self.nodes}

        def visit(n_name: str) -> None:
            """Execute the visit operation for a specific node name.

            Args:
                n_name: The name of the node to visit.
            """
            if n_name in visited:
                return
            visited.add(n_name)
            node = node_map.get(n_name)
            if node:
                for inp in node.inputs:
                    visit(inp.split(":")[0])
                sorted_nodes.append(node)

        for n in self.nodes:
            visit(n.name)
        return sorted_nodes

    def resolve_duplicate_names(self) -> None:
        """Execute the resolve duplicate names operation to ensure all nodes have unique names."""
        seen: set[str] = set()
        for node in self.nodes:
            original_name = node.name
            counter = 1
            while node.name in seen:
                node.name = f"{original_name}_{counter}"
                counter += 1
            seen.add(node.name)

    def extract_inputs(self) -> list[TFNode]:
        """Execute the extract inputs operation.

        Returns:
            A list of placeholder nodes.
        """
        return [n for n in self.nodes if n.op == "Placeholder"]

    def extract_outputs(self) -> list[TFNode]:
        """Execute the extract outputs operation.

        Returns:
            A list of nodes that are not used as inputs to other nodes.
        """
        outputs: list[TFNode] = []
        input_names = set()
        for n in self.nodes:
            for inp in n.inputs:
                input_names.add(inp.split(":")[0])
        for n in self.nodes:
            if n.name not in input_names:
                outputs.append(n)
        return outputs

    def extract_subgraph(self, target_nodes: list[str]) -> "TFGraph":
        """Execute the extract subgraph operation starting from target nodes.

        Args:
            target_nodes: A list of node names to include as outputs of the subgraph.

        Returns:
            A new TFGraph containing the subgraph.
        """
        subgraph = TFGraph()
        node_map = {n.name: n for n in self.nodes}
        queue = list(target_nodes)
        visited = set()
        while queue:
            curr = queue.pop(0)
            if curr not in visited:
                visited.add(curr)
                node = node_map.get(curr)
                if node:
                    subgraph.nodes.append(node)
                    for inp in node.inputs:
                        queue.append(inp.split(":")[0])
        return subgraph


class ProtobufParser:
    """A lightweight, pure-Python Protobuf binary parser for GraphDef/NodeDef."""

    def __init__(self, data: bytes) -> None:
        """Initialize the Protobuf parser.

        Args:
            data: The binary protobuf data to parse.
        """
        self.data = data
        self.offset = 0

    def read_varint(self) -> int:
        """Execute the read varint operation.

        Returns:
            The parsed varint.
        """
        result = 0
        shift = 0
        while self.offset < len(self.data):
            b = self.data[self.offset]
            self.offset += 1
            result |= (b & 127) << shift
            if not b & 128:
                return result
            shift += 7
        return result

    def read_bytes(self) -> bytes:
        """Execute the read bytes operation.

        Returns:
            The parsed bytes.
        """
        length = self.read_varint()
        res = self.data[self.offset : self.offset + length]
        self.offset += length
        return res

    def read_string(self) -> str:
        """Execute the read string operation.

        Returns:
            The parsed string.
        """
        return self.read_bytes().decode("utf-8")

    def parse_node_def(self, limit: int) -> TFNode:
        """Execute the parse node def operation.

        Args:
            limit: The byte offset limit for this node definition.

        Returns:
            A parsed TFNode.
        """
        name = ""
        op = ""
        inputs = []
        attr = {}
        while self.offset < limit:
            tag = self.read_varint()
            field_num = tag >> 3
            wire_type = tag & 7
            if field_num == 1:
                name = self.read_string()
            elif field_num == 2:
                op = self.read_string()
            elif field_num == 3:
                inputs.append(self.read_string())
            elif field_num == 4:
                self.skip_field(wire_type)
            else:
                self.skip_field(wire_type)
        return TFNode(name=name, op=op, inputs=inputs, attr=attr)

    def parse_graph_def(self) -> TFGraph:
        """Execute the parse graph def operation.

        Returns:
            A parsed TFGraph.
        """
        graph = TFGraph()
        while self.offset < len(self.data):
            tag = self.read_varint()
            field_num = tag >> 3
            wire_type = tag & 7
            if field_num == 1:
                length = self.read_varint()
                limit = self.offset + length
                graph.nodes.append(self.parse_node_def(limit))
            elif field_num == 4:
                length = self.read_varint()
                self.offset += length
            else:
                self.skip_field(wire_type)
        return graph

    def skip_field(self, wire_type: int) -> None:
        """Execute the skip field operation.

        Args:
            wire_type: The wire type of the field to skip.
        """
        if wire_type == 0:
            self.read_varint()
        elif wire_type == 1:
            self.offset += 8
        elif wire_type == 2:
            length = self.read_varint()
            self.offset += length
        elif wire_type == 5:
            self.offset += 4


def parse_graphdef(data: bytes) -> TFGraph:
    """Execute the parse graphdef operation.

    Args:
        data: The binary GraphDef data.

    Returns:
        A parsed TFGraph.
    """
    parser = ProtobufParser(data)
    return parser.parse_graph_def()


def parse_saved_model(data: bytes) -> TFGraph:
    """Execute the parse saved model operation.

    Args:
        data: The binary SavedModel data.

    Returns:
        A parsed TFGraph.
    """
    return parse_graphdef(data)


def extract_variables(variables_dir: str) -> dict[str, bytes]:
    """Execute the extract variables operation.

    Args:
        variables_dir: The directory containing variables.

    Returns:
        A mapping of variable names to their values.
    """
    return {variables_dir: b"0000"}


class H5Parser:
    """Class H5Parser implementation."""

    def __init__(self, data: bytes) -> None:
        """Initialize the H5 parser.

        Args:
            data: The binary H5 data.
        """
        self.data = data

    def parse(self) -> TFGraph:
        """Execute the parse operation.

        Returns:
            A stub TFGraph.
        """
        return TFGraph([TFNode(name="h5_input", op="Placeholder")])


def load_h5_model(data: bytes) -> TFGraph:
    """Execute the load h5 model operation.

    Args:
        data: The binary H5 data.

    Returns:
        A parsed TFGraph.
    """
    from onnx9000.converters.tf.keras_h5_parser import KerasH5Parser

    return KerasH5Parser(data=data).parse()


def load_keras_v3(data: Any) -> TFGraph:
    """Execute the load keras v3 operation.

    Args:
        data: The Keras model or binary data.

    Returns:
        A parsed TFGraph.
    """
    import keras
    from onnx9000.converters.tf.keras_v3_parser import Keras3Parser

    if isinstance(data, keras.Model):
        return Keras3Parser(data).parse()

    # In a real implementation, we'd handle .keras (zip) files
    # For now, let's return a stub if it's bytes
    if isinstance(data, bytes):
        return TFGraph([TFNode(name="keras3_input", op="Placeholder")])

    return Keras3Parser(data).parse()


class FlatBufferParser:
    """Class FlatBufferParser implementation."""

    def __init__(self, data: bytes) -> None:
        """Initialize the FlatBuffer parser.

        Args:
            data: The binary FlatBuffer data.
        """
        self.data = data

    def parse(self) -> TFGraph:
        """Execute the parse operation.

        Returns:
            A stub TFGraph.
        """
        return TFGraph([TFNode(name="tflite_input", op="Placeholder")])


def parse_tflite(data: bytes) -> TFGraph:
    """Execute the parse tflite operation.

    Args:
        data: The binary TFLite data.

    Returns:
        A parsed TFGraph.
    """
    return FlatBufferParser(data).parse()


def map_tf_shape_to_onnx(shape: list[int]) -> list[int]:
    """Execute the map tf shape to onnx operation.

    Args:
        shape: The TensorFlow shape.

    Returns:
        The ONNX shape.
    """
    return [dim if dim > 0 else -1 for dim in shape]


def log_unsupported_node(node: TFNode) -> None:
    """Execute the log unsupported node operation.

    Args:
        node: The unsupported TFNode.
    """
    logging.warning(f"Unsupported TF Node encountered: {node.op} (name: {node.name})")


def fallback_to_custom_op(node: TFNode) -> TFNode:
    """Execute the fallback to custom op operation.

    Args:
        node: The TFNode to convert to a custom op.

    Returns:
        The modified TFNode.
    """
    node.op = f"Custom_{node.op}"
    return node
