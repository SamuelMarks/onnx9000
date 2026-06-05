"""Caffe weights parser."""

import struct
from typing import Any, BinaryIO

import numpy as np


class ProtobufDecoder:
    """Zero-dependency protobuf decoder for Caffe models."""

    def __init__(self, data: bytes):
        """Initialize with binary data."""
        self.data = data
        self.pos = 0

    def read_varint(self) -> int:
        """Read a varint."""
        result = 0
        shift = 0
        while True:
            if self.pos >= len(self.data):
                raise EOFError("Unexpected end of data")
            b = self.data[self.pos]
            self.pos += 1
            result |= (b & 0x7F) << shift
            if not (b & 0x80):
                break
            shift += 7
        return result

    def read_tag(self) -> tuple[int, int]:
        """Read a protobuf tag."""
        if self.pos >= len(self.data):
            return 0, 0
        val = self.read_varint()
        return val >> 3, val & 0x7

    def read_string(self, length: int) -> str:
        """Read a string."""
        s = self.data[self.pos : self.pos + length].decode("utf-8")
        self.pos += length
        return s

    def read_bytes(self, length: int) -> bytes:
        """Read bytes."""
        b = self.data[self.pos : self.pos + length]
        self.pos += length
        return b

    def read_float32(self) -> float:
        """Read a 32-bit float."""
        val = struct.unpack("<f", self.data[self.pos : self.pos + 4])[0]
        self.pos += 4
        return val

    def skip(self, wire_type: int) -> None:
        """Skip a field."""
        if wire_type == 0:
            self.read_varint()
        elif wire_type == 1:
            self.pos += 8
        elif wire_type == 2:
            length = self.read_varint()
            self.pos += length
        elif wire_type == 5:
            self.pos += 4
        else:
            raise ValueError(f"Unknown wire type: {wire_type}")


def parse_blob(data: bytes) -> np.ndarray:
    """Parse a BlobProto."""
    decoder = ProtobufDecoder(data)
    shape: list[int] = []
    values: list[float] = []

    # Fallbacks for legacy dims
    num, channels, height, width = 0, 0, 0, 0

    while decoder.pos < len(decoder.data):
        field, wire = decoder.read_tag()
        if field == 0:
            break

        if field == 5 and wire == 2:  # packed data
            length = decoder.read_varint()
            end = decoder.pos + length
            while decoder.pos < end:
                values.append(decoder.read_float32())
        elif field == 5 and wire == 5:  # unpacked data
            values.append(decoder.read_float32())
        elif field == 7 and wire == 2:  # shape
            length = decoder.read_varint()
            shape_data = decoder.read_bytes(length)
            shape_dec = ProtobufDecoder(shape_data)
            while shape_dec.pos < len(shape_dec.data):
                sf, sw = shape_dec.read_tag()
                if sf == 1 and sw == 0:  # dim
                    shape.append(shape_dec.read_varint())
                elif sf == 1 and sw == 2:  # packed dim
                    dim_len = shape_dec.read_varint()
                    dim_end = shape_dec.pos + dim_len
                    while shape_dec.pos < dim_end:
                        shape.append(shape_dec.read_varint())
                else:
                    shape_dec.skip(sw)
        elif field == 1 and wire == 0:
            num = decoder.read_varint()
        elif field == 2 and wire == 0:
            channels = decoder.read_varint()
        elif field == 3 and wire == 0:
            height = decoder.read_varint()
        elif field == 4 and wire == 0:
            width = decoder.read_varint()
        else:
            decoder.skip(wire)

    arr = np.array(values, dtype=np.float32)
    if shape:
        arr = arr.reshape(shape)
    elif num or channels or height or width:
        # Avoid 0 dims if not specified
        n = num if num else 1
        c = channels if channels else 1
        h = height if height else 1
        w = width if width else 1
        # It's tricky to know exactly which are set, but usually it's NCHW
        # if total elements match NCHW we reshape
        if arr.size == n * c * h * w:
            arr = arr.reshape((n, c, h, w))

    return arr


def parse_layer(data: bytes) -> dict[str, Any]:
    """Parse a LayerParameter or V1LayerParameter."""
    decoder = ProtobufDecoder(data)
    name = ""
    blobs = []

    while decoder.pos < len(decoder.data):
        field, wire = decoder.read_tag()
        if field == 0:
            break

        if field == 1 and wire == 2:  # name or bottom in V1? Wait, V1 name is 1, V1 type is 2
            length = decoder.read_varint()
            name = decoder.read_string(length)
        elif field in (6, 50) and wire == 2:  # blobs (V1=6, V2=50)
            length = decoder.read_varint()
            blob_data = decoder.read_bytes(length)
            blobs.append(parse_blob(blob_data))
        else:
            decoder.skip(wire)

    return {"name": name, "blobs": blobs}


def load_caffemodel(f: BinaryIO) -> dict[str, list[np.ndarray]]:
    """Parse a .caffemodel file into a dictionary of weights.

    Args:
        f (BinaryIO): File object opened in binary mode.

    Returns:
        Dict[str, List[np.ndarray]]: Dictionary mapping layer name to list of weight arrays.
    """
    data = f.read()
    decoder = ProtobufDecoder(data)

    weights = {}

    while decoder.pos < len(decoder.data):
        field, wire = decoder.read_tag()
        if field == 0:
            break

        if field in (2, 100) and wire == 2:  # layer (V1=2, V2=100)
            length = decoder.read_varint()
            layer_data = decoder.read_bytes(length)
            layer_info = parse_layer(layer_data)
            if layer_info["name"] and layer_info["blobs"]:
                weights[layer_info["name"]] = layer_info["blobs"]
        else:
            decoder.skip(wire)

    return weights
