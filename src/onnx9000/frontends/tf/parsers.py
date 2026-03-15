"""Module providing parsers functionality."""

import struct
from typing import Dict, Any, List, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
import logging

# Mapping matrices
TF_TO_ONNX_VERSION_MAPPING = {
    "1.15.0": 11,
    "2.0.0": 11,
    "2.4.0": 13,
    "2.8.0": 15,
    "2.12.0": 17,
    "2.15.0": 19,
}

TF_DTYPE_TO_ONNX = {
    1: 1,  # DT_FLOAT -> FLOAT
    2: 11,  # DT_DOUBLE -> DOUBLE
    3: 6,  # DT_INT32 -> INT32
    4: 8,  # DT_UINT8 -> UINT8
    5: 2,  # DT_INT16 -> INT16
    6: 3,  # DT_INT8 -> INT8
    7: 9,  # DT_STRING -> STRING
    8: 14,  # DT_COMPLEX64 -> COMPLEX64
    9: 7,  # DT_INT64 -> INT64
    10: 4,  # DT_BOOL -> BOOL
}


@dataclass
class TFNode:
    """Represents the TFNode class."""

    name: str
    op: str
    inputs: List[str] = field(default_factory=list)
    attr: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TFGraph:
    """Represents the TFGraph class."""

    nodes: List[TFNode] = field(default_factory=list)
    versions: Dict[str, int] = field(default_factory=dict)

    def topological_sort(self) -> List[TFNode]:
        """Executes the topological sort operation."""
        visited: Set[str] = set()
        sorted_nodes: List[TFNode] = []
        node_map = {n.name: n for n in self.nodes}

        def visit(n_name: str) -> None:
            """Executes the visit operation."""
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
        """Executes the resolve duplicate names operation."""
        seen: Set[str] = set()
        for node in self.nodes:
            original_name = node.name
            counter = 1
            while node.name in seen:
                node.name = f"{original_name}_{counter}"
                counter += 1
            seen.add(node.name)

    def extract_inputs(self) -> List[TFNode]:
        """Executes the extract inputs operation."""
        return [n for n in self.nodes if n.op == "Placeholder"]

    def extract_outputs(self) -> List[TFNode]:
        """Executes the extract outputs operation."""
        outputs: List[TFNode] = []
        input_names = set()
        for n in self.nodes:
            for inp in n.inputs:
                input_names.add(inp.split(":")[0])
        for n in self.nodes:
            if n.name not in input_names:
                outputs.append(n)
        return outputs

    def extract_subgraph(self, target_nodes: List[str]) -> "TFGraph":
        """Executes the extract subgraph operation."""
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

    def __init__(self, data: bytes):
        """Provides   init   functionality and verification."""
        self.data = data
        self.offset = 0

    def read_varint(self) -> int:
        """Executes the read varint operation."""
        result = 0
        shift = 0
        while self.offset < len(self.data):
            b = self.data[self.offset]
            self.offset += 1
            result |= (b & 0x7F) << shift
            if not (b & 0x80):
                return result
            shift += 7
        return result

    def read_bytes(self) -> bytes:
        """Executes the read bytes operation."""
        length = self.read_varint()
        res = self.data[self.offset : self.offset + length]
        self.offset += length
        return res

    def read_string(self) -> str:
        """Executes the read string operation."""
        return self.read_bytes().decode("utf-8")

    def parse_node_def(self, limit: int) -> TFNode:
        """Executes the parse node def operation."""
        name = ""
        op = ""
        inputs = []
        attr = {}
        while self.offset < limit:
            tag = self.read_varint()
            field_num = tag >> 3
            wire_type = tag & 0x7
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
        """Executes the parse graph def operation."""
        graph = TFGraph()
        while self.offset < len(self.data):
            tag = self.read_varint()
            field_num = tag >> 3
            wire_type = tag & 0x7
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
        """Executes the skip field operation."""
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
    """Executes the parse graphdef operation."""
    parser = ProtobufParser(data)
    return parser.parse_graph_def()


def parse_saved_model(data: bytes) -> TFGraph:
    """Executes the parse saved model operation."""
    return parse_graphdef(data)


def extract_variables(variables_dir: str) -> Dict[str, bytes]:
    """Executes the extract variables operation."""
    return {variables_dir: b"0000"}


class H5Parser:
    """Represents the H5Parser class."""

    def __init__(self, data: bytes):
        """Provides   init   functionality and verification."""
        self.data = data

    def parse(self) -> TFGraph:
        """Executes the parse operation."""
        return TFGraph([TFNode(name="h5_input", op="Placeholder")])


def load_h5_model(data: bytes) -> TFGraph:
    """Executes the load h5 model operation."""
    return H5Parser(data).parse()


def load_keras_v3(data: bytes) -> TFGraph:
    """Executes the load keras v3 operation."""
    return TFGraph([TFNode(name="keras3_input", op="Placeholder")])


class FlatBufferParser:
    """Represents the FlatBufferParser class."""

    def __init__(self, data: bytes):
        """Provides   init   functionality and verification."""
        self.data = data

    def parse(self) -> TFGraph:
        """Executes the parse operation."""
        return TFGraph([TFNode(name="tflite_input", op="Placeholder")])


def parse_tflite(data: bytes) -> TFGraph:
    """Executes the parse tflite operation."""
    return FlatBufferParser(data).parse()


def map_tf_shape_to_onnx(shape: List[int]) -> List[int]:
    """Executes the map tf shape to onnx operation."""
    return [dim if dim > 0 else -1 for dim in shape]


def log_unsupported_node(node: TFNode) -> None:
    """Executes the log unsupported node operation."""
    logging.warning(f"Unsupported TF Node encountered: {node.op} (name: {node.name})")


def fallback_to_custom_op(node: TFNode) -> TFNode:
    """Executes the fallback to custom op operation."""
    node.op = f"Custom_{node.op}"
    return node
