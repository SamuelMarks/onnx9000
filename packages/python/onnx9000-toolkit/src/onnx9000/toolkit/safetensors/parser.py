"""Parser for the safetensors format."""

import io
import json
import mmap
import os
import struct
import sys
from collections.abc import Iterator
from typing import Any, Optional, Union


class SafetensorsError(Exception):
    """Base exception for all safetensors-related errors."""

    __dummy__ = True


class SafetensorsHeaderTooLargeError(SafetensorsError):
    """Raised when the safetensors header size exceeds the allowed limit."""

    __dummy__ = True


class SafetensorsInvalidHeaderError(SafetensorsError):
    """Raised when the safetensors header is not valid UTF-8."""

    __dummy__ = True


class SafetensorsInvalidJSONError(SafetensorsError):
    """Raised when the safetensors header is not valid JSON."""

    __dummy__ = True


class SafetensorsDuplicateKeyError(SafetensorsError):
    """Raised when the safetensors header contains duplicate tensor names."""

    __dummy__ = True


class SafetensorsInvalidOffsetError(SafetensorsError):
    """Raised when the tensor offsets are invalid (e.g., negative or reversed)."""

    __dummy__ = True


class SafetensorsOutOfBoundsError(SafetensorsError):
    """Raised when the tensor data extends beyond the file boundaries."""

    __dummy__ = True


class SafetensorsOverlapError(SafetensorsError):
    """Raised when two or more tensors have overlapping data regions."""

    __dummy__ = True


class SafetensorsAlignmentError(SafetensorsError):
    """Raised when a tensor's data offset is not 8-byte aligned."""

    __dummy__ = True


class SafetensorsInvalidDtypeError(SafetensorsError):
    """Raised when an unknown or unsupported dtype is encountered."""

    __dummy__ = True


class SafetensorsShapeMismatchError(SafetensorsError):
    """Raised when the tensor shape does not match its data size."""

    __dummy__ = True


class SafetensorsFileEmptyError(SafetensorsError):
    """Raised when attempting to load an empty file."""

    __dummy__ = True


class SafetensorsFileTooSmallError(SafetensorsError):
    """Raised when the file is too small to contain a valid safetensors header."""

    __dummy__ = True


class SafetensorsWriteError(SafetensorsError):
    """Raised when an error occurs while writing a safetensors file."""

    __dummy__ = True


_DTYPE_SIZES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U64": 8,
    "U32": 4,
    "U16": 2,
    "U8": 1,
    "BOOL": 1,
}


def _calculate_volume(shape: list[int]) -> int:
    """Calculate the total number of elements in a given shape.

    Args:
        shape: A list of integers representing the dimensions.

    Returns:
        The total volume (product of dimensions).

    Raises:
        SafetensorsShapeMismatchError: If the shape is invalid.
    """
    if not isinstance(shape, list):
        raise SafetensorsShapeMismatchError("Shape must be an array/list")
    vol = 1
    for dim in shape:
        if dim < 0:
            raise SafetensorsShapeMismatchError(f"Negative dimension found: {dim}")
        if dim > 2**50:
            raise SafetensorsShapeMismatchError(f"Dimension too large: {dim}")
        vol *= dim
    return vol


class SafeTensors:
    """A memory-mapped reader for safetensors files."""

    def __init__(
        self,
        data: Union[str, os.PathLike, bytes, io.IOBase],
        mmap_hint: bool = True,
        verify_hash: Optional[str] = None,
    ):
        """Initialize the SafeTensors reader.

        Args:
            data: A path to a file, bytes, or a file-like object.
            mmap_hint: Whether to use memory mapping hints if available.
            verify_hash: Optional SHA256 hash to verify the entire file content.

        Raises:
            SafetensorsFileEmptyError: If the input data is empty.
            SafetensorsFileTooSmallError: If the input data is too small.
            TypeError: If the input data type is not supported.
        """
        self.fd = None
        self.mm = None
        self.buffer = None

        if isinstance(data, (str, os.PathLike)):
            self.filename = str(data)
            self.file_size = os.path.getsize(self.filename)

            if self.file_size == 0:
                raise SafetensorsFileEmptyError("File is empty")
            if self.file_size < 8:
                raise SafetensorsFileTooSmallError("File too small to contain header size")

            try:
                self.fd = os.open(self.filename, os.O_RDONLY)
                self.mm = mmap.mmap(self.fd, 0, access=mmap.ACCESS_READ)
            except OSError as e:
                raise SafetensorsError(f"Failed to map file {self.filename}: {e}")
            except MemoryError as e:
                raise SafetensorsError(
                    f"Failed to map file {self.filename} (insufficient address space/RAM): {e}"
                )

            if mmap_hint and hasattr(mmap, "MADV_WILLNEED"):
                try:
                    self.mm.madvise(mmap.MADV_WILLNEED)
                except Exception:
                    return None
            self.data_view = memoryview(self.mm)
        elif isinstance(data, bytes):
            self.file_size = len(data)
            if self.file_size == 0:
                raise SafetensorsFileEmptyError("Buffer is empty")
            if self.file_size < 8:
                raise SafetensorsFileTooSmallError("Buffer too small to contain header size")
            self.data_view = memoryview(data)
        elif hasattr(data, "read") and hasattr(data, "seek") and hasattr(data, "tell"):
            if hasattr(data, "getbuffer"):
                self.data_view = memoryview(data.getbuffer())
                self.file_size = len(self.data_view)
            else:
                data.seek(0, 2)
                self.file_size = data.tell()
                data.seek(0)
                if self.file_size == 0:
                    raise SafetensorsFileEmptyError("Stream is empty")
                if self.file_size < 8:
                    raise SafetensorsFileTooSmallError("Stream too small to contain header size")
                self.buffer = data.read()
                self.data_view = memoryview(self.buffer)
        else:
            raise TypeError("data must be a path, bytes, or file-like object")

        if verify_hash:
            import hashlib

            h = hashlib.sha256()
            h.update(self.data_view)
            if h.hexdigest() != verify_hash:
                self._close()
                raise SafetensorsError(
                    f"SHA256 mismatch: expected {verify_hash}, got {h.hexdigest()}"
                )

        # Read header size (8-byte unsigned little-endian)
        header_size_bytes = self.data_view[:8]
        self.header_size = struct.unpack("<Q", header_size_bytes)[0]

        if self.header_size > 100 * 1024 * 1024:
            self._close()
            raise SafetensorsHeaderTooLargeError("Header size exceeds 100MB limit")

        if self.header_size + 8 > self.file_size:
            self._close()
            raise SafetensorsOutOfBoundsError("Header size exceeds file boundaries")

        header_bytes = self.data_view[8 : 8 + self.header_size]
        try:
            header_str = header_bytes.tobytes().decode("utf-8")
        except UnicodeDecodeError as e:
            self._close()
            raise SafetensorsInvalidHeaderError(f"Invalid UTF-8 header: {e}")

        try:
            self.header = json.loads(header_str)
            if not isinstance(self.header, dict):
                self._close()
                raise SafetensorsInvalidJSONError("JSON header must be a dictionary/object")
        except json.JSONDecodeError as e:
            self._close()
            raise SafetensorsInvalidJSONError(f"Invalid JSON header: {e}")
        except RecursionError as e:
            self._close()
            raise SafetensorsInvalidJSONError(f"JSON deeply nested recursion limits reached: {e}")

        self.metadata = self.header.pop("__metadata__", {})
        self.metadata_length = len(self.metadata)

        # XSS Protection
        for k, v in self.metadata.items():
            if isinstance(v, str) and ("<script" in v.lower() or "javascript:" in v.lower()):
                self._close()
                raise SafetensorsError("Executable script tags detected in metadata")

        if self.metadata and "format" not in self.metadata:
            import logging

            logging.getLogger(__name__).warning(
                "Safetensors metadata is missing standard HuggingFace 'format' key."
            )

        self.tensors = {}
        self._validate_header()

    def _validate_header(self):
        """Validate the safetensors header for consistency and correctness."""
        seen_regions = []
        for name, info in self.header.items():
            if not isinstance(info, dict):
                self._close()
                raise SafetensorsInvalidJSONError(f"Tensor value must be a dict: {name}")

            if name in self.tensors:
                self._close()
                raise SafetensorsDuplicateKeyError(f"Duplicate tensor name: {name}")

            dtype = info.get("dtype")
            if dtype not in _DTYPE_SIZES:
                if dtype in ("C64", "C128"):
                    # Map complex to pairs natively or fail if unsupported entirely. We will strictly reject complex until standard evolves.
                    self._close()
                    raise SafetensorsInvalidDtypeError(
                        f"Complex types ({dtype}) are not currently supported by standard safetensors."
                    )
                self._close()
                raise SafetensorsInvalidDtypeError(f"Unknown dtype: {dtype}")

            shape = info.get("shape", [])
            offsets = info.get("data_offsets", [0, 0])
            begin, end = offsets[0], offsets[1]

            if begin > end:
                self._close()
                raise SafetensorsInvalidOffsetError(f"Invalid offsets: begin {begin} > end {end}")

            if begin % 8 != 0:
                self._close()
                raise SafetensorsAlignmentError(f"Offset begin {begin} is not 8-byte aligned")

            # File offsets relative to data section
            abs_end = 8 + self.header_size + end

            if abs_end > self.file_size:
                self._close()
                raise SafetensorsOutOfBoundsError(f"Data region for {name} exceeds file boundaries")

            expected_size = _calculate_volume(shape) * _DTYPE_SIZES[dtype]
            if expected_size != (end - begin):
                self._close()
                raise SafetensorsShapeMismatchError(
                    f"Shape volume * dtype size ({expected_size}) != offset size ({end - begin}) for {name}"
                )

            seen_regions.append((begin, end))
            self.tensors[name] = info

        # Check overlaps and unreferenced data
        seen_regions.sort()
        for i in range(len(seen_regions) - 1):
            if seen_regions[i][1] > seen_regions[i + 1][0]:
                self._close()
                raise SafetensorsOverlapError("Tensor data regions overlap")

        # Check unreferenced data
        total_mapped = sum(end - begin for begin, end in seen_regions)
        data_section_size = self.file_size - 8 - self.header_size
        if total_mapped < data_section_size:
            import logging

            logging.getLogger(__name__).warning(
                f"Unreferenced data detected: {data_section_size - total_mapped} bytes are unmapped."
            )

    def get_tensor(self, name: str) -> memoryview:
        """Get a specific tensor as a memoryview.

        Args:
            name: The name of the tensor to retrieve.

        Returns:
            A memoryview of the tensor data.

        Raises:
            TypeError: If the name is not a string.
            KeyError: If the tensor is not found.
        """
        if not isinstance(name, str):
            raise TypeError(f"Key must be string, got {type(name)}")
        if name not in self.tensors:
            raise KeyError(f"Tensor {name} not found")

        info = self.tensors[name]
        begin, end = info["data_offsets"]
        abs_begin = 8 + self.header_size + begin
        abs_end = 8 + self.header_size + end
        view = self.data_view[abs_begin:abs_end]

        # Try to cast to multi-dimensional view for native lazy slicing
        struct_char_map = {
            "F64": "d",
            "F32": "f",
            "I64": "q",
            "I32": "i",
            "I16": "h",
            "I8": "b",
            "U64": "Q",
            "U32": "I",
            "U16": "H",
            "U8": "B",
            "BOOL": "?",
        }
        dtype = info["dtype"]
        shape = info.get("shape", [])

        if dtype in struct_char_map and shape:
            try:
                view = view.cast(struct_char_map[dtype], shape=tuple(shape))
            except Exception:
                pass  # ignore if python doesn't support the shape cast

        return view

    def iter_tensors(self) -> Iterator[tuple[str, memoryview]]:
        """Yield tensors lazily one by one.

        Yields:
            A tuple of (tensor name, tensor memoryview).
        """
        for name in self.tensors:
            yield name, self.get_tensor(name)

    def get_pinned_tensor(self, name: str) -> memoryview:
        """Get a tensor and pin its memory (CUDA Pinned Memory emulation).

        Args:
            name: The name of the tensor to retrieve.

        Returns:
            A memoryview of the pinned tensor data.
        """
        import ctypes
        import mmap

        view = self.get_tensor(name)
        size = len(view)

        if size == 0:
            return view

        # Allocate anonymous mmap
        if sys.platform == "win32":
            pinned_mm = mmap.mmap(-1, size)
        else:
            pinned_mm = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)

        pinned_mm.write(view)
        pinned_mm.seek(0)

        # Try to mlock
        if sys.platform != "win32":
            try:
                libc = ctypes.CDLL(None)
                if hasattr(libc, "mlock"):
                    # Writable mmap can be converted to ctypes pointer
                    buf = (ctypes.c_char * size).from_buffer(pinned_mm)
                    addr = ctypes.addressof(buf)
                    res = libc.mlock(ctypes.c_void_p(addr), ctypes.c_size_t(size))
                    if res != 0:
                        import logging

                        logging.getLogger(__name__).warning(f"mlock failed for tensor {name}")
            except Exception as e:
                import logging

                logging.getLogger(__name__).debug(f"Could not mlock tensor {name}: {e}")
        else:
            try:
                kernel32 = ctypes.windll.kernel32
                buf = (ctypes.c_char * size).from_buffer(pinned_mm)
                addr = ctypes.addressof(buf)
                res = kernel32.VirtualLock(ctypes.c_void_p(addr), ctypes.c_size_t(size))
                if res == 0:
                    import logging

                    logging.getLogger(__name__).warning(f"VirtualLock failed for tensor {name}")
            except Exception as e:
                import logging

                logging.getLogger(__name__).debug(f"Could not VirtualLock tensor {name}: {e}")

        return memoryview(pinned_mm)

    def get_tensors(self, *names: str) -> dict[str, memoryview]:
        """Return multiple tensors in a dictionary.

        Args:
            *names: Variable length list of tensor names.

        Returns:
            A dictionary mapping names to memoryviews.
        """
        return {name: self.get_tensor(name) for name in names}

    def get_memory_footprint(self) -> int:
        """Calculate the total size of all tensor payloads in bytes.

        Returns:
            Total size in bytes.
        """
        return sum(
            info["data_offsets"][1] - info["data_offsets"][0] for info in self.tensors.values()
        )

    def get_onnx9000_tensor(self, name: str):
        """Map a safetensor to a framework-agnostic onnx9000.core.ir.Tensor.

        Args:
            name: The name of the tensor to retrieve.

        Returns:
            An onnx9000.core.ir.Tensor object.

        Raises:
            ImportError: If onnx9000.core is not available.
            SafetensorsInvalidDtypeError: If the dtype cannot be mapped.
        """
        try:
            from onnx9000.core.dtypes import DType
            from onnx9000.core.ir import Tensor
        except ImportError:
            raise ImportError("onnx9000.core is not available for get_onnx9000_tensor()")

        _dtype_map = {
            "F64": DType.FLOAT64,
            "F32": DType.FLOAT32,
            "F16": DType.FLOAT16,
            "BF16": DType.BFLOAT16,
            "I64": DType.INT64,
            "I32": DType.INT32,
            "I16": DType.INT16,
            "I8": DType.INT8,
            "U64": DType.UINT64,
            "U32": DType.UINT32,
            "U16": DType.UINT16,
            "U8": DType.UINT8,
            "BOOL": DType.BOOL,
        }

        info = self.tensors[name]
        dtype_str = info["dtype"]
        shape = tuple(info["shape"])

        if dtype_str not in _dtype_map:
            raise SafetensorsInvalidDtypeError(f"Cannot map {dtype_str} to ONNX DType")

        data = self.get_tensor(name)
        return Tensor(
            name=name,
            shape=shape,
            dtype=_dtype_map[dtype_str],
            is_initializer=True,
            requires_grad=False,
            data=data,
        )

    def get_numpy(self, name: str, downcast_f16: bool = False, quantize_int8: bool = False):
        """Convert a tensor to a numpy array.

        Args:
            name: The name of the tensor to retrieve.
            downcast_f16: Whether to downcast float32/float64 to float16.
            quantize_int8: Whether to perform symmetric quantization to int8.

        Returns:
            A numpy array.

        Raises:
            ImportError: If numpy is not available.
            SafetensorsInvalidDtypeError: If the dtype cannot be mapped.
        """
        try:
            import numpy as np
        except ImportError:
            raise ImportError("numpy is not available for get_numpy()")

        _np_dtypes = {
            "F64": np.dtype("<f8"),
            "F32": np.dtype("<f4"),
            "F16": np.dtype("<f2"),
            "I64": np.dtype("<i8"),
            "I32": np.dtype("<i4"),
            "I16": np.dtype("<i2"),
            "I8": np.dtype("i1"),
            "U64": np.dtype("<u8"),
            "U32": np.dtype("<u4"),
            "U16": np.dtype("<u2"),
            "U8": np.dtype("u1"),
            "BOOL": np.dtype("bool"),
        }

        info = self.tensors[name]
        dtype = info["dtype"]
        shape = info["shape"]

        if dtype not in _np_dtypes:
            raise SafetensorsInvalidDtypeError(f"Cannot map {dtype} to numpy")

        view = self.get_tensor(name)
        arr = np.frombuffer(view, dtype=_np_dtypes[dtype]).reshape(shape)
        arr.flags.writeable = False

        if downcast_f16 and arr.dtype == np.float32:
            arr = arr.astype(np.float16)
        elif downcast_f16 and arr.dtype == np.float64:
            arr = arr.astype(np.float16)

        if quantize_int8 and arr.dtype in (np.float32, np.float16):
            # Simplified symmetric quantization
            amax = np.abs(arr).max()
            scale = amax / 127.0 if amax > 0 else 1.0
            arr = np.round(arr / scale).astype(np.int8)

        return arr

    def keys(self) -> list[str]:
        """Return the names of all tensors in the file.

        Returns:
            A list of tensor names.
        """
        return list(self.tensors.keys())

    def __getitem__(self, key: str) -> memoryview:
        """Get a tensor by name using indexing.

        Args:
            key: The name of the tensor.

        Returns:
            A memoryview of the tensor data.
        """
        return self.get_tensor(key)

    def __contains__(self, key: str) -> bool:
        """Check if a tensor exists in the file.

        Args:
            key: The name of the tensor.

        Returns:
            True if the tensor exists, False otherwise.
        """
        return key in self.tensors

    def _close(self):
        """Close the file and release resources."""
        if hasattr(self, "data_view") and self.data_view is not None:
            self.data_view.release()
            self.data_view = None
        if hasattr(self, "mm") and self.mm:
            try:
                self.mm.close()
            except BufferError:
                pass  # Pointers are still exported; GC will handle it when they die
            self.mm = None
        if hasattr(self, "fd") and self.fd:
            os.close(self.fd)
            self.fd = None

    def __del__(self):
        """Clean up resources when the object is deleted."""
        self._close()

    def __enter__(self):
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and close resources."""
        self._close()


def save(tensors: dict[str, Any], metadata: Optional[dict[str, str]] = None) -> bytes:
    """Serialize tensors and return raw bytes.

    Args:
        tensors: A dictionary mapping names to tensor-like objects (bytes, numpy arrays, etc.).
        metadata: Optional dictionary of string metadata.

    Returns:
        The serialized safetensors data as bytes.
    """
    header = {}

    # Support appending format version identifiers natively
    meta = metadata or {}
    if "format" not in meta:
        meta["format"] = "pt"  # Default HuggingFace compatibility format
    if "version" not in meta:
        meta["version"] = "1.0"

    header["__metadata__"] = meta

    offset = 0
    data_buffer = io.BytesIO()

    for name, data in tensors.items():
        if not isinstance(name, str):
            raise TypeError(f"Dictionary keys must be strings, got {type(name)}")
        if name in header:
            raise SafetensorsDuplicateKeyError(f"Duplicate key {name}")

        dtype_str = "U8"
        shape = []
        raw_bytes = data

        if hasattr(data, "tobytes"):
            # np.ndarray, memoryview
            raw_bytes = data.tobytes()
            if hasattr(data, "shape"):
                shape = list(data.shape)
            if hasattr(data, "dtype"):
                # basic mapping for numpy
                dt_name = (
                    str(data.dtype.name).upper()
                    if hasattr(data.dtype, "name")
                    else str(data.dtype).upper()
                )
                if dt_name.startswith("FLOAT64"):
                    dtype_str = "F64"
                elif dt_name.startswith("FLOAT32"):
                    dtype_str = "F32"
                elif dt_name.startswith("FLOAT16"):
                    dtype_str = "F16"
                elif dt_name.startswith("INT64"):
                    dtype_str = "I64"
                elif dt_name.startswith("INT32"):
                    dtype_str = "I32"
                elif dt_name.startswith("INT16"):
                    dtype_str = "I16"
                elif dt_name.startswith("INT8"):
                    dtype_str = "I8"
                elif dt_name.startswith("UINT64"):
                    dtype_str = "U64"
                elif dt_name.startswith("UINT32"):
                    dtype_str = "U32"
                elif dt_name.startswith("UINT16"):
                    dtype_str = "U16"
                elif dt_name.startswith("UINT8"):
                    dtype_str = "U8"
                elif dt_name.startswith("BOOL"):
                    dtype_str = "BOOL"
        elif hasattr(data, "raw_data") and hasattr(data, "dims"):
            # ONNX TensorProto
            raw_bytes = data.raw_data
            shape = list(data.dims)
            # map data.data_type to dtype_str
            # TensorProto.DataType.FLOAT = 1 -> "F32", etc.
            # Simplified fallback for ONNX TensorProto if needed
            from onnx9000.core.dtypes import DType

            _inv_dtype = {
                DType.FLOAT64.value: "F64",
                DType.FLOAT32.value: "F32",
                DType.FLOAT16.value: "F16",
                DType.INT64.value: "I64",
                DType.INT32.value: "I32",
                DType.INT16.value: "I16",
                DType.INT8.value: "I8",
                DType.UINT64.value: "U64",
                DType.UINT32.value: "U32",
                DType.UINT16.value: "U16",
                DType.UINT8.value: "U8",
                DType.BOOL.value: "BOOL",
                DType.BFLOAT16.value: "BF16",
            }
            if hasattr(data, "data_type") and data.data_type in _inv_dtype:
                dtype_str = _inv_dtype[data.data_type]

        size = len(raw_bytes)
        if not shape:
            shape = [size]

        header[name] = {"dtype": dtype_str, "shape": shape, "data_offsets": [offset, offset + size]}
        data_buffer.write(raw_bytes)
        offset += size

        # Safetensors standard: data_offsets should be 8-byte aligned.
        remainder = offset % 8
        if remainder != 0:
            padding = 8 - remainder
            data_buffer.write(b"\0" * padding)
            offset += padding

    header_json = json.dumps(header).encode("utf-8")
    header_size = len(header_json)

    # Padding header to 8-byte alignment
    padding = (8 - (header_size % 8)) % 8
    header_json += b" " * padding
    header_size = len(header_json)

    out = io.BytesIO()
    out.write(struct.pack("<Q", header_size))
    out.write(header_json)
    out.write(data_buffer.getvalue())
    return out.getvalue()


def save_file(
    tensors: dict[str, Any],
    filename: str,
    metadata: Optional[dict[str, str]] = None,
    overwrite: bool = True,
):
    """Save tensors to a file in safetensors format.

    Args:
        tensors: A dictionary mapping names to tensor-like objects.
        filename: The path to the file to save.
        metadata: Optional dictionary of string metadata.
        overwrite: Whether to overwrite the file if it already exists.

    Raises:
        SafetensorsWriteError: If the file exists and overwrite is False.
    """
    if not overwrite and os.path.exists(filename):
        raise SafetensorsWriteError(f"File {filename} already exists and overwrite=False")
    with open(filename, "wb") as f:
        f.write(save(tensors, metadata))


def load_file(
    filename: str, prefix: str = "", pattern: Optional[str] = None, device: str = "cpu"
) -> dict[str, Any]:
    """Load tensors from a safetensors file.

    Args:
        filename: The path to the safetensors file.
        prefix: Optional prefix to add to all loaded tensor names.
        pattern: Optional regex pattern to filter tensor names.
        device: Target device for loaded tensors (currently ignored).

    Returns:
        A dictionary mapping names to numpy arrays.
    """
    import re

    result = {}

    regex = re.compile(pattern) if pattern else None

    with SafeTensors(filename) as st:
        for name in st.keys():
            if not isinstance(name, str):
                raise TypeError(f"Dictionary keys must be strings, got {type(name)}")

            if regex and not regex.match(name):
                continue

            new_name = prefix + name
            result[new_name] = st.get_numpy(name)

    return result


def load(data: bytes, prefix: str = "", pattern: Optional[str] = None) -> dict[str, Any]:
    """Load tensors from safetensors bytes.

    Args:
        data: The raw safetensors bytes.
        prefix: Optional prefix to add to all loaded tensor names.
        pattern: Optional regex pattern to filter tensor names.

    Returns:
        A dictionary mapping names to numpy arrays.
    """
    import re

    result = {}
    regex = re.compile(pattern) if pattern else None

    with SafeTensors(data) as st:
        for name in st.keys():
            if not isinstance(name, str):
                raise TypeError(f"Dictionary keys must be strings, got {type(name)}")
            if regex and not regex.match(name):
                continue
            new_name = prefix + name
            result[new_name] = st.get_numpy(name)

    return result


def check_safetensors(filename: str) -> bool:
    """Check if a file is a valid safetensors file.

    Args:
        filename: The path to the file to check.

    Returns:
        True if the file is valid, False otherwise.
    """
    try:
        with SafeTensors(filename):
            return True
    except SafetensorsError:
        return False


check_file_validity = check_safetensors


def get_metadata(filename: str) -> dict[str, str]:
    """Get the metadata from a safetensors file.

    Args:
        filename: The path to the safetensors file.

    Returns:
        A dictionary of metadata.
    """
    with SafeTensors(filename) as st:
        return st.metadata


def get_tensor(filename: str, key: str) -> memoryview:
    """Extract a single tensor from a safetensors file.

    Args:
        filename: The path to the safetensors file.
        key: The name of the tensor to extract.

    Returns:
        A memoryview of the tensor data (copied to bytes to ensure safety after file close).
    """
    with SafeTensors(filename) as st:
        # Since we use memoryview connected to mmap, we must extract and copy to bytes if we want to return safely.
        # However, the user might want a zero-copy view. Returning bytes is the safest if file is closed.
        # Actually returning bytes loses memoryview benefits, but is the only way here.
        return memoryview(bytes(st.get_tensor(key)))


def safe_open(filename: str, framework: str = "pt", device: str = "cpu"):
    """Open a safetensors file (emulates HuggingFace API).

    Args:
        filename: The path to the safetensors file.
        framework: The framework to use (ignored).
        device: The device to use (ignored).

    Returns:
        A SafeTensors reader object.
    """
    return SafeTensors(filename)


class SafeTensorsSharded:
    """A reader for sharded safetensors models."""

    def __init__(self, index_filename: str):
        """Initialize the sharded safetensors reader.

        Args:
            index_filename: The path to the index JSON file.

        Raises:
            SafetensorsError: If the index file is invalid.
        """
        self.index_filename = index_filename
        self.base_dir = os.path.dirname(index_filename)

        with open(index_filename, encoding="utf-8") as f:
            self.index = json.load(f)
            print(f"\nLOADED SHARDED INDEX from {index_filename}: {self.index}")

        self.metadata = self.index.get("metadata", {})
        self.weight_map = self.index.get("weight_map", {})
        self._files = {}

    def _get_file(self, filename: str) -> SafeTensors:
        """Get a SafeTensors reader for a specific shard file.

        Args:
            filename: The name of the shard file.

        Returns:
            A SafeTensors reader object.

        Raises:
            SafetensorsError: If path traversal is detected.
        """
        if ".." in filename or filename.startswith("/") or filename.startswith("\\"):
            raise SafetensorsError(f"Path traversal detected in sharded index: {filename}")
        if filename not in self._files:
            filepath = os.path.join(self.base_dir, filename)
            self._files[filename] = SafeTensors(filepath)
        return self._files[filename]

    def get_tensor(self, name: str) -> memoryview:
        """Get a tensor from the sharded model.

        Args:
            name: The name of the tensor.

        Returns:
            A memoryview of the tensor data.

        Raises:
            KeyError: If the tensor is not found in the index.
        """
        if name not in self.weight_map:
            raise KeyError(f"Tensor {name} not found in sharded index")

        filename = self.weight_map[name]
        st = self._get_file(filename)
        return st.get_tensor(name)

    def get_numpy(self, name: str):
        """Get a tensor from the sharded model as a numpy array.

        Args:
            name: The name of the tensor.

        Returns:
            A numpy array.

        Raises:
            KeyError: If the tensor is not found in the index.
        """
        if name not in self.weight_map:
            raise KeyError(f"Tensor {name} not found in sharded index")

        filename = self.weight_map[name]
        st = self._get_file(filename)
        return st.get_numpy(name)

    def keys(self) -> list[str]:
        """Return the names of all tensors in the sharded model.

        Returns:
            A list of tensor names.
        """
        return list(self.weight_map.keys())

    def __contains__(self, key: str) -> bool:
        """Check if a tensor exists in the sharded model.

        Args:
            key: The name of the tensor.

        Returns:
            True if the tensor exists, False otherwise.
        """
        return key in self.weight_map

    def __getitem__(self, key: str) -> memoryview:
        """Get a tensor by name using indexing.

        Args:
            key: The name of the tensor.

        Returns:
            A memoryview of the tensor data.
        """
        return self.get_tensor(key)

    def _close(self):
        """Close all shard files."""
        for st in self._files.values():
            st._close()
        self._files.clear()

    def __del__(self):
        """Clean up resources when the object is deleted."""
        self._close()

    def __enter__(self):
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and close resources."""
        self._close()


def save_sharded(
    tensors: dict[str, Any],
    base_dir: str,
    metadata: Optional[dict[str, str]] = None,
    max_shard_size: int = 10 * 1024 * 1024 * 1024,
    prefix: str = "model",
):
    """Save tensors as a sharded safetensors model.

    Args:
        tensors: A dictionary mapping names to tensor-like objects.
        base_dir: The directory to save the shards and index.
        metadata: Optional dictionary of string metadata.
        max_shard_size: Maximum size of each shard in bytes.
        prefix: Prefix for the shard filenames.
    """
    os.makedirs(base_dir, exist_ok=True)

    shards = []
    current_shard_tensors = {}
    current_shard_size = 0

    for name, data in tensors.items():
        if hasattr(data, "tobytes"):
            size = data.nbytes if hasattr(data, "nbytes") else len(data.tobytes())
        elif hasattr(data, "raw_data"):
            size = len(data.raw_data)
        else:
            size = len(data)

        if current_shard_size + size > max_shard_size and current_shard_tensors:
            shards.append(current_shard_tensors)
            current_shard_tensors = {}
            current_shard_size = 0

        current_shard_tensors[name] = data
        current_shard_size += size

    if current_shard_tensors:
        shards.append(current_shard_tensors)

    num_shards = len(shards)
    weight_map = {}

    for i, shard_tensors in enumerate(shards):
        shard_filename = f"{prefix}-{i + 1:05d}-of-{num_shards:05d}.safetensors"
        shard_path = os.path.join(base_dir, shard_filename)
        save_file(shard_tensors, shard_path, metadata if i == 0 else None)

        for name in shard_tensors.keys():
            weight_map[name] = shard_filename

    index_data = {"metadata": metadata or {}, "weight_map": weight_map}
    index_path = os.path.join(base_dir, f"{prefix}.safetensors.index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2)
