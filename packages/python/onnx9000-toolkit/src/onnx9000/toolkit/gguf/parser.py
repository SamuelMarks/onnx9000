"""Parser module."""

import mmap
import os
import struct

from onnx9000.core.dtypes import DType
from onnx9000.core.ir import QuantizedTensor, Tensor


class GGUFError(Exception):
    """Represents gguferror inheriting from Exception."""

    def __init__(self, message: str = "An error occurred") -> None:
        """Initialize GGUFError with configured parameters and state.

        Args:
            message: Parameter specifying message (str).
        """
        super().__init__(message)


class GGUFParser:
    """Represents ggufparser, supporting operations such as get_onnx9000_tensor, keys."""

    def __init__(self, filename: str) -> None:
        """Initialize GGUFParser with configured parameters and state.

        Args:
            filename: Parameter specifying filename (str).

        Raises:
            GGUFError: If an invalid state or argument is encountered during execution.
        """
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
        """Read bytes."""
        data = self.view[self.offset : self.offset + n].tobytes()
        self.offset += n
        return data

    def _read_uint32(self) -> int:
        """Read uint32."""
        return struct.unpack("<I", self._read_bytes(4))[0]

    def _read_uint64(self) -> int:
        """Read uint64."""
        return struct.unpack("<Q", self._read_bytes(8))[0]

    def _read_string(self) -> str:
        """Read string."""
        length = self._read_uint64()
        return self._read_bytes(length).decode("utf-8")

    def _read_value(self) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Read value."""
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
                else:
                    pass
            return arr
        elif vtype == 10:
            return self._read_uint64()  # UINT64
        elif vtype == 11:
            return struct.unpack("<q", self._read_bytes(8))[0]  # INT64
        elif vtype == 12:
            return struct.unpack("<d", self._read_bytes(8))[0]  # FLOAT64
        else:
            raise GGUFError(f"Unsupported GGUF value type {vtype}")

    def get_onnx9000_tensor(self, name: str) -> Tensor | QuantizedTensor:
        """Execute get onnx9000 tensor on GGUFParser.

        Args:
            name: Parameter specifying name (str).

        Returns:
            Result of get onnx9000 tensor (Tensor | QuantizedTensor).

        Raises:
            KeyError: If an invalid state or argument is encountered during execution.
        """
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
                        name=name,
                        type=qtype,
                        block_size=block_size,
                        shape=shape,
                        data=tensor_data,
                    )
                else:
                    return Tensor(
                        name=name,
                        shape=shape,
                        dtype=dtype,
                        is_initializer=True,
                        data=tensor_data,
                    )
        raise KeyError(f"Tensor {name} not found")

    def keys(self) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Execute keys on GGUFParser.

        Returns:
            Result of keys.
        """
        return [t["name"] for t in self.tensors]

    def __contains__(self, key: str) -> bool:
        """Check if a tensor exists in the GGUF file.

        Args:
            key: Tensor name to check.

        Returns:
            True if present, False otherwise.
        """
        return any(t["name"] == key for t in self.tensors)

    def __enter__(self) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Enter the runtime context managed by GGUFParser.

        Returns:
            Result of enter.
        """
        return self

    def __exit__(
        self,
        exc_type: int | float | str | bool | bytes | list | dict | tuple | None,
        exc_val: int | float | str | bool | bytes | list | dict | tuple | None,
        exc_tb: int | float | str | bool | bytes | list | dict | tuple | None,
    ) -> None:
        """Exit the runtime context managed by GGUFParser and handle exceptions.

        Args:
            exc_type: Parameter specifying exc type.
            exc_val: Parameter specifying exc val.
            exc_tb: Parameter specifying exc tb.
        """
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
