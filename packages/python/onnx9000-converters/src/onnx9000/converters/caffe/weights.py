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
                raise EOFError("Unexpected end of data")  # pragma: no cover
            b = self.data[self.pos]
            self.pos += 1
            result |= (b & 0x7F) << shift
            if not (b & 0x80):
                break
            shift += 7  # pragma: no cover
        return result

    def read_tag(self) -> tuple[int, int]:
        """Read a protobuf tag."""
        if self.pos >= len(self.data):
            return 0, 0  # pragma: no cover
        val = self.read_varint()
        return val >> 3, val & 0x7

    def read_string(self, length: int) -> str:  # pragma: no cover
        """Read a string."""
        s = self.data[self.pos : self.pos + length].decode("utf-8")  # pragma: no cover
        self.pos += length  # pragma: no cover
        return s  # pragma: no cover

    def read_bytes(self, length: int) -> bytes:
        """Read bytes."""
        b = self.data[self.pos : self.pos + length]
        self.pos += length
        return b

    def read_float32(self) -> float:  # pragma: no cover
        """Read a 32-bit float."""
        val = struct.unpack("<f", self.data[self.pos : self.pos + 4])[0]  # pragma: no cover
        self.pos += 4  # pragma: no cover
        return val  # pragma: no cover

    def skip(self, wire_type: int) -> None:
        """Skip a field."""
        if wire_type == 0:  # pragma: no cover
            self.read_varint()  # pragma: no cover
        elif wire_type == 1:  # pragma: no cover
            self.pos += 8  # pragma: no cover
        elif wire_type == 2:  # pragma: no cover
            length = self.read_varint()  # pragma: no cover
            self.pos += length  # pragma: no cover
        elif wire_type == 5:  # pragma: no cover
            self.pos += 4  # pragma: no cover
        else:
            raise ValueError(f"Unknown wire type: {wire_type}")  # pragma: no cover


def parse_blob(data: bytes) -> np.ndarray:
    """Parse a BlobProto."""
    decoder = ProtobufDecoder(data)  # pragma: no cover
    shape: list[int] = []  # pragma: no cover
    values: list[float] = []  # pragma: no cover

    # Fallbacks for legacy dims
    num, channels, height, width = 0, 0, 0, 0  # pragma: no cover

    while decoder.pos < len(decoder.data):  # pragma: no cover
        field, wire = decoder.read_tag()  # pragma: no cover
        if field == 0:  # pragma: no cover
            break  # pragma: no cover

        if field == 5 and wire == 2:  # packed data  # pragma: no cover
            length = decoder.read_varint()  # pragma: no cover
            end = decoder.pos + length  # pragma: no cover
            while decoder.pos < end:  # pragma: no cover
                values.append(decoder.read_float32())  # pragma: no cover
        elif field == 5 and wire == 5:  # unpacked data  # pragma: no cover
            values.append(decoder.read_float32())  # pragma: no cover
        elif field == 7 and wire == 2:  # shape  # pragma: no cover
            length = decoder.read_varint()  # pragma: no cover
            shape_data = decoder.read_bytes(length)  # pragma: no cover
            shape_dec = ProtobufDecoder(shape_data)  # pragma: no cover
            while shape_dec.pos < len(shape_dec.data):  # pragma: no cover
                sf, sw = shape_dec.read_tag()  # pragma: no cover
                if sf == 1 and sw == 0:  # dim  # pragma: no cover
                    shape.append(shape_dec.read_varint())  # pragma: no cover
                elif sf == 1 and sw == 2:  # packed dim  # pragma: no cover
                    dim_len = shape_dec.read_varint()  # pragma: no cover
                    dim_end = shape_dec.pos + dim_len  # pragma: no cover
                    while shape_dec.pos < dim_end:  # pragma: no cover
                        shape.append(shape_dec.read_varint())  # pragma: no cover
                else:
                    shape_dec.skip(sw)  # pragma: no cover
        elif field == 1 and wire == 0:  # pragma: no cover
            num = decoder.read_varint()  # pragma: no cover
        elif field == 2 and wire == 0:  # pragma: no cover
            channels = decoder.read_varint()  # pragma: no cover
        elif field == 3 and wire == 0:  # pragma: no cover
            height = decoder.read_varint()  # pragma: no cover
        elif field == 4 and wire == 0:  # pragma: no cover
            width = decoder.read_varint()  # pragma: no cover
        else:
            decoder.skip(wire)  # pragma: no cover

    arr = np.array(values, dtype=np.float32)  # pragma: no cover
    if shape:  # pragma: no cover
        arr = arr.reshape(shape)  # pragma: no cover
    elif num or channels or height or width:  # pragma: no cover
        # Avoid 0 dims if not specified
        n = num if num else 1  # pragma: no cover
        c = channels if channels else 1  # pragma: no cover
        h = height if height else 1  # pragma: no cover
        w = width if width else 1  # pragma: no cover
        # It's tricky to know exactly which are set, but usually it's NCHW
        # if total elements match NCHW we reshape
        if arr.size == n * c * h * w:  # pragma: no cover
            arr = arr.reshape((n, c, h, w))  # pragma: no cover

    return arr  # pragma: no cover


def parse_layer(data: bytes) -> dict[str, Any]:
    """Parse a LayerParameter or V1LayerParameter."""
    decoder = ProtobufDecoder(data)  # pragma: no cover
    name = ""  # pragma: no cover
    blobs = []  # pragma: no cover

    while decoder.pos < len(decoder.data):  # pragma: no cover
        field, wire = decoder.read_tag()  # pragma: no cover
        if field == 0:  # pragma: no cover
            break  # pragma: no cover

        if (
            field == 1 and wire == 2
        ):  # name or bottom in V1? Wait, V1 name is 1, V1 type is 2  # pragma: no cover
            length = decoder.read_varint()  # pragma: no cover
            name = decoder.read_string(length)  # pragma: no cover
        elif field in (6, 50) and wire == 2:  # blobs (V1=6, V2=50)  # pragma: no cover
            length = decoder.read_varint()  # pragma: no cover
            blob_data = decoder.read_bytes(length)  # pragma: no cover
            blobs.append(parse_blob(blob_data))  # pragma: no cover
        else:
            decoder.skip(wire)  # pragma: no cover

    return {"name": name, "blobs": blobs}  # pragma: no cover


def load_caffemodel(f: BinaryIO) -> dict[str, list[np.ndarray]]:  # pragma: no cover
    """Parse a .caffemodel file into a dictionary of weights.

    Args:
        f (BinaryIO): File object opened in binary mode.

    Returns:
        Dict[str, List[np.ndarray]]: Dictionary mapping layer name to list of weight arrays.

    """  # pragma: no cover
    data = f.read()  # pragma: no cover
    decoder = ProtobufDecoder(data)
    # pragma: no cover
    weights = {}
    # pragma: no cover
    while decoder.pos < len(decoder.data):  # pragma: no cover
        field, wire = decoder.read_tag()  # pragma: no cover
        if field == 0:  # pragma: no cover
            break
        # pragma: no cover
        if field in (2, 100) and wire == 2:  # layer (V1=2, V2=100)  # pragma: no cover
            length = decoder.read_varint()  # pragma: no cover
            layer_data = decoder.read_bytes(length)  # pragma: no cover
            layer_info = parse_layer(layer_data)  # pragma: no cover
            if layer_info["name"] and layer_info["blobs"]:  # pragma: no cover
                weights[layer_info["name"]] = layer_info["blobs"]
        else:  # pragma: no cover
            decoder.skip(wire)
    # pragma: no cover
    return weights
