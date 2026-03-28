"""Module providing onnx2gguf functionality."""

import struct
from typing import BinaryIO, Any
from .builder import GGUFValueType, GGUFTensorType


class GGUFReader:
    """Provides functionality for G G U F Reader."""

    def __init__(self, f: BinaryIO):
        """Initializes a new instance of the class."""
        self.f = f
        self.kvs = {}
        self.tensors = {}
        self._read_header()

    def _read_string(self) -> str:
        """read string."""
        length = struct.unpack("<Q", self.f.read(8))[0]
        return self.f.read(length).decode("utf-8")

    def _read_val(self, vtype: GGUFValueType) -> Any:
        """read val."""
        if vtype == GGUFValueType.UINT8:
            return struct.unpack("<B", self.f.read(1))[0]
        elif vtype == GGUFValueType.INT8:
            return struct.unpack("<b", self.f.read(1))[0]
        elif vtype == GGUFValueType.UINT16:
            return struct.unpack("<H", self.f.read(2))[0]
        elif vtype == GGUFValueType.INT16:
            return struct.unpack("<h", self.f.read(2))[0]
        elif vtype == GGUFValueType.UINT32:
            return struct.unpack("<I", self.f.read(4))[0]
        elif vtype == GGUFValueType.INT32:
            return struct.unpack("<i", self.f.read(4))[0]
        elif vtype == GGUFValueType.FLOAT32:
            return struct.unpack("<f", self.f.read(4))[0]
        elif vtype == GGUFValueType.UINT64:
            return struct.unpack("<Q", self.f.read(8))[0]
        elif vtype == GGUFValueType.INT64:
            return struct.unpack("<q", self.f.read(8))[0]
        elif vtype == GGUFValueType.FLOAT64:
            return struct.unpack("<d", self.f.read(8))[0]
        elif vtype == GGUFValueType.BOOL:
            return struct.unpack("<?", self.f.read(1))[0]
        elif vtype == GGUFValueType.STRING:
            return self._read_string()
        elif vtype == GGUFValueType.ARRAY:
            atype_int = struct.unpack("<I", self.f.read(4))[0]
            atype = GGUFValueType(atype_int)
            length = struct.unpack("<Q", self.f.read(8))[0]
            arr = []
            for _ in range(length):
                arr.append(self._read_val(atype))
            return arr
        else:
            raise ValueError(f"Unknown value type: {vtype}")

    def _read_header(self):
        """read header."""
        magic = self.f.read(4)
        if magic != b"GGUF":
            raise ValueError("Not a GGUF file")
        version = struct.unpack("<I", self.f.read(4))[0]
        if version not in (2, 3):
            raise ValueError(f"Unsupported GGUF version {version}")
        tensor_count = struct.unpack("<Q", self.f.read(8))[0]
        kv_count = struct.unpack("<Q", self.f.read(8))[0]
        for _ in range(kv_count):
            key = self._read_string()
            vtype_int = struct.unpack("<I", self.f.read(4))[0]
            vtype = GGUFValueType(vtype_int)
            val = self._read_val(vtype)
            self.kvs[key] = val
        for _ in range(tensor_count):
            name = self._read_string()
            n_dims = struct.unpack("<I", self.f.read(4))[0]
            shape = []
            for _ in range(n_dims):
                shape.append(struct.unpack("<Q", self.f.read(8))[0])
            ttype_int = struct.unpack("<I", self.f.read(4))[0]
            ttype = GGUFTensorType(ttype_int)
            offset = struct.unpack("<Q", self.f.read(8))[0]
            self.tensors[name] = {"name": name, "shape": shape, "type": ttype, "offset": offset}
        self.alignment = self.kvs.get("general.alignment", 32)
        pos = self.f.tell()
        padding = (self.alignment - pos % self.alignment) % self.alignment
        self.f.read(padding)
        self.data_start = self.f.tell()

    def get_tensor(self, name: str) -> bytes:
        """Gets tensor."""
        if name not in self.tensors:
            raise KeyError(name)
        t = self.tensors[name]
        items = 1
        for d in t["shape"]:
            items *= d
        dtype = t["type"]
        if dtype == GGUFTensorType.F32:
            size = items * 4
        elif dtype == GGUFTensorType.F16:
            size = items * 2
        elif dtype == GGUFTensorType.Q4_0:
            size = items // 32 * 18
        elif dtype == GGUFTensorType.Q4_1:
            size = items // 32 * 20
        elif dtype == GGUFTensorType.Q8_0:
            size = items // 32 * 34
        else:
            raise ValueError("Unknown type")
        self.f.seek(self.data_start + t["offset"])
        return self.f.read(size)
