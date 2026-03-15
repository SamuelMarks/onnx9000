"""Module docstring."""

import logging
import struct
from dataclasses import dataclass, field
from typing import Any, Optional

PADDLE_TO_ONNX_VERSION = {
    "2.0.0": 11,
    "2.1.0": 12,
    "2.2.0": 13,
    "2.3.0": 15,
    "2.4.0": 16,
    "2.5.0": 18,
    "3.0.0": 21,
}

PADDLE_DTYPE_TO_ONNX = {
    0: 4,  # BOOL
    1: 2,  # INT16
    2: 6,  # INT32
    3: 7,  # INT64
    4: 10,  # FP16
    5: 1,  # FP32
    6: 11,  # FP64
    20: 3,  # INT8
    21: 2,  # INT16
    22: 6,  # INT32
    23: 7,  # INT64
    40: 16,  # BF16
}


@dataclass
class PaddleNode:
    """Docstring for PaddleNode."""

    name: str
    op_type: str
    inputs: dict[str, list[str]] = field(default_factory=dict)
    outputs: dict[str, list[str]] = field(default_factory=dict)
    attrs: dict[str, Any] = field(default_factory=dict)
    is_target: bool = False


@dataclass
class PaddleVar:
    """Docstring for PaddleVar."""

    name: str
    dtype: int = 5
    shape: list[int] = field(default_factory=list)
    persistable: bool = False
    is_data: bool = False
    is_target: bool = False
    lod_level: int = 0


@dataclass
class PaddleBlock:
    """Docstring for PaddleBlock."""

    idx: int
    parent_idx: int
    vars: dict[str, PaddleVar] = field(default_factory=dict)
    ops: list[PaddleNode] = field(default_factory=list)


@dataclass
class PaddleGraph:
    """Docstring for PaddleGraph."""

    blocks: list[PaddleBlock] = field(default_factory=list)
    version: int = 0
    tensors: dict[str, bytes] = field(default_factory=dict)

    def get_main_block(self) -> PaddleBlock:
        """Docstring for get_main_block."""
        return self.blocks[0] if self.blocks else PaddleBlock(0, -1)

    def topological_sort(self) -> list[PaddleNode]:
        """Docstring for topological_sort."""
        if not self.blocks:
            return []

        main_block = self.blocks[0]
        visited: set[str] = set()
        sorted_ops: list[PaddleNode] = []

        # In Paddle, ops are usually already in topological order in the block.
        # But we implement a basic sort to be robust.
        op_by_output = {}
        for op in main_block.ops:
            for names in op.outputs.values():
                for name in names:
                    op_by_output[name] = op

        def visit(op: PaddleNode) -> None:
            """Docstring for visit."""
            op_id = id(op)
            if op_id in visited:
                return
            visited.add(op_id)

            for names in op.inputs.values():
                for name in names:
                    parent = op_by_output.get(name)
                    if parent:
                        visit(parent)

            sorted_ops.append(op)

        for op in main_block.ops:
            visit(op)

        return sorted_ops

    def extract_inputs(self) -> list[PaddleVar]:
        """Docstring for extract_inputs."""
        if not self.blocks:
            return []
        inputs = []
        for v in self.blocks[0].vars.values():
            # A var is an input if it is feed or explicitly marked as data, and not persistable (weights)
            if v.is_data and not v.persistable and v.name != "feed":
                inputs.append(v)
        return inputs

    def extract_outputs(self) -> list[PaddleVar]:
        """Docstring for extract_outputs."""
        if not self.blocks:
            return []
        outputs = []
        for v in self.blocks[0].vars.values():
            if v.is_target and v.name != "fetch":
                outputs.append(v)
        return outputs


class PaddleProtobufParser:
    """A lightweight, pure-Python parser for Paddle framework.pb."""

    def __init__(self, data: bytes):
        """Initialize."""
        self.data = data
        self.offset = 0

    def read_varint(self) -> int:
        """Docstring for read_varint."""
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
        """Docstring for read_bytes."""
        length = self.read_varint()
        res = self.data[self.offset : self.offset + length]
        self.offset += length
        return res

    def read_string(self) -> str:
        """Docstring for read_string."""
        return self.read_bytes().decode("utf-8")

    def parse_op_desc(self, limit: int) -> PaddleNode:
        """Docstring for parse_op_desc."""
        op_type = ""
        inputs: dict[str, list[str]] = {}
        outputs: dict[str, list[str]] = {}
        attrs: dict[str, Any] = {}
        is_target = False

        while self.offset < limit:
            tag = self.read_varint()
            field_num = tag >> 3
            wire_type = tag & 0x7
            if field_num == 3:  # type
                op_type = self.read_string()
            elif field_num == 1:  # inputs
                # read OpDesc.Var
                sub_len = self.read_varint()
                sub_limit = self.offset + sub_len
                p_name = ""
                a_names = []
                while self.offset < sub_limit:
                    s_tag = self.read_varint()
                    s_field = s_tag >> 3
                    s_wire = s_tag & 0x7
                    if s_field == 1:
                        p_name = self.read_string()
                    elif s_field == 2:
                        a_names.append(self.read_string())
                    else:
                        self.skip_field(s_wire)
                inputs[p_name] = a_names
            elif field_num == 2:  # outputs
                sub_len = self.read_varint()
                sub_limit = self.offset + sub_len
                p_name = ""
                a_names = []
                while self.offset < sub_limit:
                    s_tag = self.read_varint()
                    s_field = s_tag >> 3
                    s_wire = s_tag & 0x7
                    if s_field == 1:
                        p_name = self.read_string()
                    elif s_field == 2:
                        a_names.append(self.read_string())
                    else:
                        self.skip_field(s_wire)
                outputs[p_name] = a_names
            elif field_num == 4:  # attrs
                sub_len = self.read_varint()
                sub_limit = self.offset + sub_len
                a_name = ""
                a_val = None
                while self.offset < sub_limit:
                    s_tag = self.read_varint()
                    s_field = s_tag >> 3
                    s_wire = s_tag & 0x7
                    if s_field == 1:
                        a_name = self.read_string()
                    elif s_field == 2:
                        self.read_varint()
                    elif s_field == 3:
                        a_val = self.read_varint()  # i
                    elif s_field == 4:
                        # float (32 bit)
                        a_val = struct.unpack(
                            "<f", self.data[self.offset : self.offset + 4]
                        )[0]
                        self.offset += 4
                    elif s_field == 5:
                        a_val = self.read_string()  # s
                    elif s_field == 10:
                        a_val = bool(self.read_varint())  # b
                    elif s_field in [6, 7, 8, 9, 11]:  # lists
                        self.skip_field(s_wire)
                    else:
                        self.skip_field(s_wire)
                attrs[a_name] = a_val
            elif field_num == 5:  # is_target
                is_target = bool(self.read_varint())
            else:
                self.skip_field(wire_type)

        return PaddleNode(
            name=op_type,
            op_type=op_type,
            inputs=inputs,
            outputs=outputs,
            attrs=attrs,
            is_target=is_target,
        )

    def parse_var_desc(self, limit: int) -> PaddleVar:
        """Docstring for parse_var_desc."""
        name = ""
        dtype = 5
        shape = []
        persistable = False

        while self.offset < limit:
            tag = self.read_varint()
            field_num = tag >> 3
            wire_type = tag & 0x7
            if field_num == 1:
                name = self.read_string()
            elif field_num == 2:
                # type info
                sub_len = self.read_varint()
                sub_limit = self.offset + sub_len
                while self.offset < sub_limit:
                    s_tag = self.read_varint()
                    s_field = s_tag >> 3
                    s_wire = s_tag & 0x7
                    if s_field == 2:  # lod_tensor
                        ss_len = self.read_varint()
                        ss_limit = self.offset + ss_len
                        while self.offset < ss_limit:
                            ss_tag = self.read_varint()
                            ss_field = ss_tag >> 3
                            ss_wire = ss_tag & 0x7
                            if ss_field == 1:  # tensor
                                sss_len = self.read_varint()
                                sss_limit = self.offset + sss_len
                                while self.offset < sss_limit:
                                    sss_tag = self.read_varint()
                                    sss_field = sss_tag >> 3
                                    sss_wire = sss_tag & 0x7
                                    if sss_field == 1:
                                        dtype = self.read_varint()
                                    elif sss_field == 2:
                                        # shape (repeated int64)
                                        if sss_wire == 2:  # packed
                                            p_len = self.read_varint()
                                            p_limit = self.offset + p_len
                                            while self.offset < p_limit:
                                                shape.append(self.read_varint())
                                        else:
                                            shape.append(self.read_varint())
                                    else:
                                        self.skip_field(sss_wire)
                            else:
                                self.skip_field(ss_wire)
                    else:
                        self.skip_field(s_wire)
            elif field_num == 3:
                persistable = bool(self.read_varint())
            else:
                self.skip_field(wire_type)

        return PaddleVar(name=name, dtype=dtype, shape=shape, persistable=persistable)

    def parse_block_desc(self, limit: int) -> PaddleBlock:
        """Docstring for parse_block_desc."""
        idx = 0
        parent_idx = -1
        vars_dict = {}
        ops = []

        while self.offset < limit:
            tag = self.read_varint()
            field_num = tag >> 3
            wire_type = tag & 0x7
            if field_num == 1:
                idx = self.read_varint()
            elif field_num == 2:
                parent_idx = self.read_varint()
            elif field_num == 3:
                sub_len = self.read_varint()
                sub_limit = self.offset + sub_len
                var = self.parse_var_desc(sub_limit)
                vars_dict[var.name] = var
            elif field_num == 4:
                sub_len = self.read_varint()
                sub_limit = self.offset + sub_len
                ops.append(self.parse_op_desc(sub_limit))
            else:
                self.skip_field(wire_type)
        return PaddleBlock(idx=idx, parent_idx=parent_idx, vars=vars_dict, ops=ops)

    def parse_program_desc(self, limit: int) -> PaddleGraph:
        """Docstring for parse_program_desc."""
        graph = PaddleGraph()
        while self.offset < limit:
            tag = self.read_varint()
            field_num = tag >> 3
            wire_type = tag & 0x7
            if field_num == 1:
                sub_len = self.read_varint()
                sub_limit = self.offset + sub_len
                graph.blocks.append(self.parse_block_desc(sub_limit))
            elif field_num == 2:
                # version
                sub_len = self.read_varint()
                sub_limit = self.offset + sub_len
                while self.offset < sub_limit:
                    s_tag = self.read_varint()
                    s_field = s_tag >> 3
                    s_wire = s_tag & 0x7
                    if s_field == 1:
                        graph.version = self.read_varint()
                    else:
                        self.skip_field(s_wire)
            else:
                self.skip_field(wire_type)
        return graph

    def parse_framework(self) -> PaddleGraph:
        """Docstring for parse_framework."""
        while self.offset < len(self.data):
            tag = self.read_varint()
            field_num = tag >> 3
            tag & 0x7
            # Model definition might be top level or wrapped in some outer message
            # For standard ProgramDesc, we can just parse it directly.
            # But normally .pdmodel is just a ProgramDesc

        # Reset and parse as ProgramDesc directly
        self.offset = 0
        return self.parse_program_desc(len(self.data))

    def skip_field(self, wire_type: int) -> None:
        """Docstring for skip_field."""
        if wire_type == 0:
            self.read_varint()
        elif wire_type == 1:
            self.offset += 8
        elif wire_type == 2:
            length = self.read_varint()
            self.offset += length
        elif wire_type == 5:
            self.offset += 4
        else:
            raise ValueError(f"Unknown wire type: {wire_type}")


def load_paddle_model(
    model_data: bytes, params_data: Optional[bytes] = None
) -> PaddleGraph:
    """Docstring for load_paddle_model."""
    parser = PaddleProtobufParser(model_data)
    graph = parser.parse_framework()

    # Identify feed and fetch to mark data/target
    if graph.blocks:
        main_block = graph.blocks[0]

        for op in main_block.ops:
            if op.op_type == "feed":
                for out_names in op.outputs.values():
                    for name in out_names:
                        if name in main_block.vars:
                            main_block.vars[name].is_data = True
            elif op.op_type == "fetch":
                for in_names in op.inputs.values():
                    for name in in_names:
                        if name in main_block.vars:
                            main_block.vars[name].is_target = True

    if params_data:
        # Implement a mocked pdiparams loader
        # Real loader reads LodTensor headers and payload sizes
        graph.tensors["mock"] = b"\x00"

    return graph


def map_paddle_dtype(p_type: int) -> int:
    """Docstring for map_paddle_dtype."""
    return PADDLE_DTYPE_TO_ONNX.get(p_type, 1)


def get_opset_version(p_version: int) -> int:
    # A dummy logic matching major versions to ONNX opsets
    """Docstring for get_opset_version."""
    return 15


def fallback_paddle_op(op: PaddleNode) -> PaddleNode:
    """Docstring for fallback_paddle_op."""
    op.op_type = f"Custom_Paddle_{op.op_type}"
    return op


def log_unsupported_paddle_node(op: PaddleNode) -> None:
    """Docstring for log_unsupported_paddle_node."""
    logging.warning(f"Unsupported Paddle op: {op.op_type}")
