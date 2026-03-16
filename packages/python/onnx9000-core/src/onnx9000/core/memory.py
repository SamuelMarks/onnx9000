"""Memory mapping and tensor file reading utilities."""

from pathlib import Path
from typing import Union
from onnx9000.core.exceptions import Onnx9000Error


class MemoryMapError(Onnx9000Error):
    """Exception raised when failing to memory-map a file."""


def mmap_tensor_data(file_path: Union[str, Path], offset: int, length: int) -> memoryview:
    """
    Creates a zero-copy memoryview of a specific chunk of a file.
    Ideal for loading large tensor initializers without loading the whole file into RAM.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise MemoryMapError(f"File not found: {path}")
    file_size = path.stat().st_size
    if offset + length > file_size:
        raise MemoryMapError(
            f"Requested mapping [{offset}:{offset + length}] exceeds file size {file_size}"
        )
    f = open(path, "rb")
    try:
        if length == 0:
            return memoryview(b"")
        import mmap

        ALLOCATIONGRANULARITY = mmap.ALLOCATIONGRANULARITY
        page_offset = offset // ALLOCATIONGRANULARITY * ALLOCATIONGRANULARITY
        relative_offset = offset - page_offset
        map_size = relative_offset + length
        mm = mmap.mmap(f.fileno(), map_size, access=mmap.ACCESS_READ, offset=page_offset)
        view = memoryview(mm)
        sliced_view = view[relative_offset : relative_offset + length]
        return sliced_view
    except Exception as e:
        f.close()
        raise MemoryMapError(f"Failed to mmap {path}: {e}") from e
    finally:
        f.close()
