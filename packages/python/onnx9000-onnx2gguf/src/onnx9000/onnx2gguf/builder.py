"""Module providing core GGUF Builder logic."""

import struct
from enum import Enum
from typing import Any, Union, BinaryIO


class GGUFValueType(Enum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


class GGUFTensorType(Enum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q8_0 = 8


class GGUFWriter:
    """Zero-dependency GGUF Writer."""

    def __init__(self, f: BinaryIO):
        self.f = f
        self.kvs: list[tuple[str, GGUFValueType, Any]] = []
        self.tensors: list[dict[str, Any]] = []
        self.tensor_data_offset = 0
        self.tensor_sizes = []

    def add_uint8(self, key: str, val: int) -> None:
        self.kvs.append((key, GGUFValueType.UINT8, val))

    def add_int8(self, key: str, val: int) -> None:
        self.kvs.append((key, GGUFValueType.INT8, val))

    def add_uint16(self, key: str, val: int) -> None:
        self.kvs.append((key, GGUFValueType.UINT16, val))

    def add_int16(self, key: str, val: int) -> None:
        self.kvs.append((key, GGUFValueType.INT16, val))

    def add_uint32(self, key: str, val: int) -> None:
        self.kvs.append((key, GGUFValueType.UINT32, val))

    def add_int32(self, key: str, val: int) -> None:
        self.kvs.append((key, GGUFValueType.INT32, val))

    def add_float32(self, key: str, val: float) -> None:
        self.kvs.append((key, GGUFValueType.FLOAT32, val))

    def add_uint64(self, key: str, val: int) -> None:
        self.kvs.append((key, GGUFValueType.UINT64, val))

    def add_int64(self, key: str, val: int) -> None:
        self.kvs.append((key, GGUFValueType.INT64, val))

    def add_float64(self, key: str, val: float) -> None:
        self.kvs.append((key, GGUFValueType.FLOAT64, val))

    def add_bool(self, key: str, val: bool) -> None:
        self.kvs.append((key, GGUFValueType.BOOL, val))

    def add_string(self, key: str, val: str) -> None:
        self.kvs.append((key, GGUFValueType.STRING, val))

    def add_array(self, key: str, val: list[Any], array_type: GGUFValueType) -> None:
        self.kvs.append((key, GGUFValueType.ARRAY, (array_type, val)))

    def _write_string(self, s: str) -> None:
        encoded = s.encode("utf-8")
        if len(encoded) > 2**31 - 1:
            raise ValueError("String exceeds standard allocation limits")
        self.f.write(struct.pack("<Q", len(encoded)))
        self.f.write(encoded)

    def _write_val(self, vtype: GGUFValueType, val: Any) -> None:
        if vtype == GGUFValueType.UINT8:
            self.f.write(struct.pack("<B", val))
        elif vtype == GGUFValueType.INT8:
            self.f.write(struct.pack("<b", val))
        elif vtype == GGUFValueType.UINT16:
            self.f.write(struct.pack("<H", val))
        elif vtype == GGUFValueType.INT16:
            self.f.write(struct.pack("<h", val))
        elif vtype == GGUFValueType.UINT32:
            self.f.write(struct.pack("<I", val))
        elif vtype == GGUFValueType.INT32:
            self.f.write(struct.pack("<i", val))
        elif vtype == GGUFValueType.FLOAT32:
            self.f.write(struct.pack("<f", float(val)))
        elif vtype == GGUFValueType.UINT64:
            self.f.write(struct.pack("<Q", val))
        elif vtype == GGUFValueType.INT64:
            self.f.write(struct.pack("<q", val))
        elif vtype == GGUFValueType.FLOAT64:
            self.f.write(struct.pack("<d", float(val)))
        elif vtype == GGUFValueType.BOOL:
            self.f.write(struct.pack("<?", bool(val)))
        elif vtype == GGUFValueType.STRING:
            self._write_string(val)
        elif vtype == GGUFValueType.ARRAY:
            atype, arr = val
            self.f.write(struct.pack("<I", atype.value))
            self.f.write(struct.pack("<Q", len(arr)))
            for item in arr:
                self._write_val(atype, item)
        else:
            raise ValueError(f"Unknown value type: {vtype}")

    def add_tensor_info(self, name: str, shape: list[int], dtype: GGUFTensorType) -> None:
        if not isinstance(dtype, GGUFTensorType):
            raise ValueError("Invalid tensor type")
        self.tensors.append(
            {"name": name, "shape": shape, "type": dtype.value, "offset": self.tensor_data_offset}
        )

        # Compute size in bytes and update offset
        # GGUF shapes are listed in [x, y, z] order, but size is product * itemsize
        items = 1
        for d in shape:
            items *= d

        if dtype == GGUFTensorType.F32:
            size = items * 4
        elif dtype == GGUFTensorType.F16:
            size = items * 2
        elif dtype == GGUFTensorType.Q4_0:
            assert items % 32 == 0, "Q4_0 requires multiples of 32"
            size = (items // 32) * 18
        elif dtype == GGUFTensorType.Q4_1:
            assert items % 32 == 0, "Q4_1 requires multiples of 32"
            size = (items // 32) * 20
        else:
            assert items % 32 == 0, "Q8_0 requires multiples of 32"
            size = (items // 32) * 34

        alignment = 32
        for k, vtype, val in self.kvs:
            if k == "general.alignment" and vtype == GGUFValueType.UINT32:
                alignment = val

        padding = (alignment - (size % alignment)) % alignment
        self.tensor_sizes.append(size + padding)
        self.tensor_data_offset += size + padding

    def write_header_to_file(self) -> None:
        # Magic
        self.f.write(b"GGUF")
        # Version 3
        self.f.write(struct.pack("<I", 3))
        # Tensor count
        self.f.write(struct.pack("<Q", len(self.tensors)))
        # KV count
        self.f.write(struct.pack("<Q", len(self.kvs)))

        # Write KVs
        for k, vtype, val in self.kvs:
            self._write_string(k)
            self.f.write(struct.pack("<I", vtype.value))
            self._write_val(vtype, val)

        # Write tensor infos
        for t in self.tensors:
            self._write_string(t["name"])
            self.f.write(struct.pack("<I", len(t["shape"])))
            for dim in t["shape"]:
                self.f.write(struct.pack("<Q", dim))
            self.f.write(struct.pack("<I", t["type"]))
            self.f.write(struct.pack("<Q", t["offset"]))

        # Compute padding for the end of the header
        alignment = 32
        for k, vtype, val in self.kvs:
            if k == "general.alignment" and vtype == GGUFValueType.UINT32:
                alignment = val

        current_pos = self.f.tell()
        padding = (alignment - (current_pos % alignment)) % alignment
        self.f.write(b"\x00" * padding)

    def write_tensor_data(self, tensor_bytes: bytes) -> None:
        self.f.write(tensor_bytes)
        alignment = 32
        for k, vtype, val in self.kvs:
            if k == "general.alignment" and vtype == GGUFValueType.UINT32:
                alignment = val
        padding = (alignment - (len(tensor_bytes) % alignment)) % alignment
        self.f.write(b"\x00" * padding)
