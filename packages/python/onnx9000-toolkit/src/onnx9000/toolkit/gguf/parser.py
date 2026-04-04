"""Module docstring."""

import mmap
import os
import struct
from typing import Any, Union

from onnx9000.core.dtypes import DType
from onnx9000.core.ir import QuantizedTensor, Tensor


class GGUFError(Exception):
    """Docstring for D101."""


class GGUFParser:
    """Docstring for D101."""

    def __init__(self, filename: str):
        """Docstring for D107."""
        self.filename = filename
        self.file_size = os.path.getsize(self.filename)
        self.fd = os.open(self.filename, os.O_RDONLY)
        self.mm = mmap.mmap(self.fd, 0, access=mmap.ACCESS_READ)
        self.view = memoryview(self.mm)
        self.offset = 0

        self.magic = self._read_bytes(4)
        if self.magic != b"GGUF":
            raise GGUFError(f"Invalid magic bytes: {self.magic}")

        self.version = self._read_uint32()
        self.tensor_count = self._read_uint64()
        self.kv_count = self._read_uint64()

        self.metadata = {}
        for _ in range(self.kv_count):
            key = self._read_string()
            val = self._read_value()
            self.metadata[key] = val

        self.tensors = []
        for _ in range(self.tensor_count):
            name = self._read_string()
            n_dims = self._read_uint32()
            shape = []
            for _ in range(n_dims):
                shape.append(self._read_uint64())
            ggml_type = self._read_uint32()
            offset = self._read_uint64()
            self.tensors.append(
                {"name": name, "shape": shape, "ggml_type": ggml_type, "offset": offset}
            )

        self.alignment = self.metadata.get("general.alignment", 32)

        # Calculate start of data
        padding = (self.alignment - (self.offset % self.alignment)) % self.alignment
        self.data_offset = self.offset + padding

    def _read_bytes(self, n: int) -> bytes:
        data = self.view[self.offset : self.offset + n].tobytes()
        self.offset += n
        return data

    def _read_uint32(self) -> int:
        return struct.unpack("<I", self._read_bytes(4))[0]

    def _read_uint64(self) -> int:
        return struct.unpack("<Q", self._read_bytes(8))[0]

    def _read_string(self) -> str:
        length = self._read_uint64()
        return self._read_bytes(length).decode("utf-8")

    def _read_value(self) -> Any:
        vtype = self._read_uint32()
        if vtype == 0:
            return struct.unpack("<B", self._read_bytes(1))[0]  # UINT8
        elif vtype == 1:
            return struct.unpack("<b", self._read_bytes(1))[0]  # INT8
        elif vtype == 2:
            return struct.unpack("<H", self._read_bytes(2))[0]  # UINT16
        elif vtype == 3:
            return struct.unpack("<h", self._read_bytes(2))[0]  # INT16
        elif vtype == 4:
            return struct.unpack("<I", self._read_bytes(4))[0]  # UINT32
        elif vtype == 5:
            return struct.unpack("<i", self._read_bytes(4))[0]  # INT32
        elif vtype == 6:
            return struct.unpack("<f", self._read_bytes(4))[0]  # FLOAT32
        elif vtype == 7:
            return bool(struct.unpack("<?", self._read_bytes(1))[0])  # BOOL
        elif vtype == 8:
            return self._read_string()  # STRING
        elif vtype == 9:  # ARRAY
            atype = self._read_uint32()
            alen = self._read_uint64()
            arr = []
            for _ in range(alen):
                if atype == 0:
                    arr.append(struct.unpack("<B", self._read_bytes(1))[0])
                elif atype == 1:
                    arr.append(struct.unpack("<b", self._read_bytes(1))[0])
                elif atype == 2:
                    arr.append(struct.unpack("<H", self._read_bytes(2))[0])
                elif atype == 3:
                    arr.append(struct.unpack("<h", self._read_bytes(2))[0])
                elif atype == 4:
                    arr.append(struct.unpack("<I", self._read_bytes(4))[0])
                elif atype == 5:
                    arr.append(struct.unpack("<i", self._read_bytes(4))[0])
                elif atype == 6:
                    arr.append(struct.unpack("<f", self._read_bytes(4))[0])
                elif atype == 7:
                    arr.append(bool(struct.unpack("<?", self._read_bytes(1))[0]))
                elif atype == 8:
                    arr.append(self._read_string())
                elif atype == 10:
                    arr.append(self._read_uint64())
                elif atype == 11:
                    arr.append(struct.unpack("<q", self._read_bytes(8))[0])
                elif atype == 12:
                    arr.append(struct.unpack("<d", self._read_bytes(8))[0])
            return arr
        elif vtype == 10:
            return self._read_uint64()  # UINT64
        elif vtype == 11:
            return struct.unpack("<q", self._read_bytes(8))[0]  # INT64
        elif vtype == 12:
            return struct.unpack("<d", self._read_bytes(8))[0]  # FLOAT64
        else:
            raise GGUFError(f"Unsupported GGUF value type {vtype}")

    def get_onnx9000_tensor(self, name: str) -> Union[Tensor, QuantizedTensor]:
        """Docstring for D102."""
        for t in self.tensors:
            if t["name"] == name:
                ggml_type = t["ggml_type"]
                shape = tuple(t["shape"][::-1])

                elements = 1
                for dim in shape:
                    elements *= dim

                qtype = None
                block_size = None
                dtype = None
                byte_length = 0

                if ggml_type == 0:
                    dtype = DType.FLOAT32
                    byte_length = elements * 4
                elif ggml_type == 1:
                    dtype = DType.FLOAT16
                    byte_length = elements * 2
                elif ggml_type == 2:
                    qtype = "Q4_0"
                    block_size = 32
                    byte_length = (elements // 32) * 18
                elif ggml_type == 8:
                    qtype = "Q8_0"
                    block_size = 32
                    byte_length = (elements // 32) * 34
                else:
                    dtype = DType.FLOAT32
                    byte_length = elements

                tensor_start = self.data_offset + t["offset"]
                tensor_data = self.view[tensor_start : tensor_start + byte_length]

                if qtype:
                    return QuantizedTensor(
                        name=name, type=qtype, block_size=block_size, shape=shape, data=tensor_data
                    )
                else:
                    return Tensor(
                        name=name, shape=shape, dtype=dtype, is_initializer=True, data=tensor_data
                    )
        raise KeyError(f"Tensor {name} not found")

    def keys(self):
        """Docstring for D102."""
        return [t["name"] for t in self.tensors]

    def __enter__(self):
        """Docstring for D105."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Docstring for D105."""
        if hasattr(self, "view") and self.view is not None:
            self.view.release()
            self.view = None
        if hasattr(self, "mm") and self.mm:
            try:
                self.mm.close()
            except BufferError:
                _ignore = True
            self.mm = None
        if hasattr(self, "fd") and self.fd:
            os.close(self.fd)
            self.fd = None
