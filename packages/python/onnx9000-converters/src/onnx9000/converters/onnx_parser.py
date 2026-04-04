"""Module docstring."""

import mmap
import os
import struct
from typing import Any

WIRETYPE_VARINT = 0
WIRETYPE_FIXED64 = 1
WIRETYPE_LENGTH_DELIMITED = 2
WIRETYPE_START_GROUP = 3
WIRETYPE_END_GROUP = 4
WIRETYPE_FIXED32 = 5


def read_varint(view: memoryview, offset: int) -> tuple[int, int]:
    """Docstring for D103."""
    result = 0
    shift = 0
    while True:
        if offset >= len(view):
            raise EOFError("Unexpected end of file while reading varint")
        byte = view[offset]
        offset += 1
        result |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            break
        shift += 7
    return result, offset


def read_tag(view: memoryview, offset: int) -> tuple[int, int, int]:
    """Docstring for D103."""
    tag_val, offset = read_varint(view, offset)
    wire_type = tag_val & 0x7
    field_num = tag_val >> 3
    return field_num, wire_type, offset


def skip_field(view: memoryview, offset: int, wire_type: int) -> int:
    """Docstring for D103."""
    if wire_type == WIRETYPE_VARINT:
        _, offset = read_varint(view, offset)
    elif wire_type == WIRETYPE_FIXED64:
        offset += 8
    elif wire_type == WIRETYPE_LENGTH_DELIMITED:
        length, offset = read_varint(view, offset)
        offset += length
    elif wire_type == WIRETYPE_FIXED32:
        offset += 4
    else:
        raise ValueError(f"Unsupported wire type {wire_type}")
    return offset


class PureOnnxParser:
    """Docstring for D101."""

    def __init__(self, filename: str):
        """Docstring for D107."""
        self.filename = filename
        self.fd = os.open(filename, os.O_RDONLY)
        self.mm = mmap.mmap(self.fd, 0, access=mmap.ACCESS_READ)
        self.view = memoryview(self.mm)

    def parse_model(self):
        """Docstring for D102."""
        return self._parse_message(self.view, 0, len(self.view), "ModelProto")

    def _parse_message(self, view: memoryview, start: int, end: int, msg_type: str) -> dict:
        result = {}
        offset = start
        while offset < end:
            field_num, wire_type, offset = read_tag(view, offset)

            if msg_type == "ModelProto":
                if field_num == 7:  # graph
                    length, offset = read_varint(view, offset)
                    result["graph"] = self._parse_message(
                        view, offset, offset + length, "GraphProto"
                    )
                    offset += length
                else:
                    offset = skip_field(view, offset, wire_type)

            elif msg_type == "GraphProto":
                if field_num == 1:  # node
                    length, offset = read_varint(view, offset)
                    if "node" not in result:
                        result["node"] = []
                    result["node"].append(
                        self._parse_message(view, offset, offset + length, "NodeProto")
                    )
                    offset += length
                elif field_num == 2:  # name
                    length, offset = read_varint(view, offset)
                    result["name"] = view[offset : offset + length].tobytes().decode("utf-8")
                    offset += length
                elif field_num == 5:  # initializer
                    length, offset = read_varint(view, offset)
                    if "initializer" not in result:
                        result["initializer"] = []
                    result["initializer"].append(
                        self._parse_message(view, offset, offset + length, "TensorProto")
                    )
                    offset += length
                elif field_num == 11:  # input
                    length, offset = read_varint(view, offset)
                    if "input" not in result:
                        result["input"] = []
                    result["input"].append(
                        self._parse_message(view, offset, offset + length, "ValueInfoProto")
                    )
                    offset += length
                elif field_num == 12:  # output
                    length, offset = read_varint(view, offset)
                    if "output" not in result:
                        result["output"] = []
                    result["output"].append(
                        self._parse_message(view, offset, offset + length, "ValueInfoProto")
                    )
                    offset += length
                else:
                    offset = skip_field(view, offset, wire_type)

            elif msg_type == "NodeProto":
                if field_num == 1:  # input
                    length, offset = read_varint(view, offset)
                    if "input" not in result:
                        result["input"] = []
                    result["input"].append(view[offset : offset + length].tobytes().decode("utf-8"))
                    offset += length
                elif field_num == 2:  # output
                    length, offset = read_varint(view, offset)
                    if "output" not in result:
                        result["output"] = []
                    result["output"].append(
                        view[offset : offset + length].tobytes().decode("utf-8")
                    )
                    offset += length
                elif field_num == 3:  # name
                    length, offset = read_varint(view, offset)
                    result["name"] = view[offset : offset + length].tobytes().decode("utf-8")
                    offset += length
                elif field_num == 4:  # op_type
                    length, offset = read_varint(view, offset)
                    result["op_type"] = view[offset : offset + length].tobytes().decode("utf-8")
                    offset += length
                elif field_num == 5:  # attribute
                    length, offset = read_varint(view, offset)
                    if "attribute" not in result:
                        result["attribute"] = []
                    result["attribute"].append(
                        self._parse_message(view, offset, offset + length, "AttributeProto")
                    )
                    offset += length
                else:
                    offset = skip_field(view, offset, wire_type)

            elif msg_type == "TensorProto":
                if field_num == 1:  # dims
                    if wire_type == WIRETYPE_LENGTH_DELIMITED:  # packed
                        length, offset = read_varint(view, offset)
                        end_dims = offset + length
                        if "dims" not in result:
                            result["dims"] = []
                        while offset < end_dims:
                            val, offset = read_varint(view, offset)
                            result["dims"].append(val)
                    else:  # unpacked
                        val, offset = read_varint(view, offset)
                        if "dims" not in result:
                            result["dims"] = []
                        result["dims"].append(val)
                elif field_num == 2:  # data_type
                    val, offset = read_varint(view, offset)
                    result["data_type"] = val
                elif field_num == 8:  # name
                    length, offset = read_varint(view, offset)
                    result["name"] = view[offset : offset + length].tobytes().decode("utf-8")
                    offset += length
                elif field_num == 9:  # raw_data
                    length, offset = read_varint(view, offset)
                    result["raw_data"] = view[offset : offset + length]
                    offset += length
                else:
                    offset = skip_field(view, offset, wire_type)
            else:
                offset = skip_field(view, offset, wire_type)

        return result

    def __enter__(self):
        """Docstring for D105."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Docstring for D105."""
        if hasattr(self, "mm") and self.mm:
            try:
                self.mm.close()
            except BufferError:
                pass
            self.mm = None
        if hasattr(self, "fd") and self.fd:
            os.close(self.fd)
            self.fd = None
