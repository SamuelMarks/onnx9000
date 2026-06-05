import json
import mmap
import os
import struct
from typing import Any, BinaryIO


class SafeTensorsMmapParser:
    """Zero-copy mmap streaming parser for safetensors."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.header_len: int = 0
        self.header: dict[str, Any] = {}

    def parse(self) -> None:
        """Parse the safetensors header to setup chunking."""
        with open(self.file_path, "rb") as f:
            length_bytes = f.read(8)
            if len(length_bytes) < 8:
                return
            self.header_len = struct.unpack("<Q", length_bytes)[0]
            header_bytes = f.read(self.header_len)
            self.header = json.loads(header_bytes.decode("utf-8"))

    def stream_tensor(self, tensor_name: str) -> Any:
        """Stream a tensor directly into a chunked block.

        Args:
            tensor_name: Name of the tensor in the safetensors file.

        Returns:
            A memoryview of the tensor data mapped directly from disk.
        """
        if not self.header or tensor_name not in self.header:
            return None

        tensor_info = self.header[tensor_name]
        data_offsets = tensor_info["data_offsets"]
        start_offset = 8 + self.header_len + data_offsets[0]
        length = data_offsets[1] - data_offsets[0]

        with open(self.file_path, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            view = memoryview(mm)
            return view[start_offset : start_offset + length]


class GSPMDReconciler:
    """Utilities to stitch multi-slice JAX TPU arrays into unified pseudo-flat arrays."""

    @staticmethod
    def stitch_shards(
        shards: list[bytes],
        axis: int = 0,
        shard_shape: tuple[int, ...] = None,
        dtype: str = None,
    ) -> bytes:
        """Stitch multiple bytes objects (representing sharded arrays) into one.

        Args:
            shards: List of byte slices.
            axis: The axis to concatenate along (0 for simplest byte concatenation).
            shard_shape: Optional shape of each shard to properly concatenate along non-zero axes.
            dtype: Optional numpy dtype string of the shards.

        Returns:
            The concatenated bytes representing a pseudo-flat array.
        """
        if shard_shape is not None and dtype is not None and axis != 0:
            import numpy as np

            arrays = [np.frombuffer(s, dtype=dtype).reshape(shard_shape) for s in shards]
            concatenated = np.concatenate(arrays, axis=axis)
            return concatenated.tobytes()

        return b"".join(shards)


class BFloat16Upcaster:
    """Streams bfloat16 to float32 upcasting during parse time."""

    @staticmethod
    def upcast_bfloat16_to_float32(bf16_bytes: bytes) -> bytes:
        """Upcast a sequence of bfloat16 bytes to float32 bytes.

        Args:
            bf16_bytes: Bytes representing bfloat16 values.

        Returns:
            Bytes representing float32 values.
        """
        # Each bfloat16 is 2 bytes. We upcast by padding 16 zero bits to the right.
        result = bytearray(len(bf16_bytes) * 2)
        for i in range(0, len(bf16_bytes), 2):
            # bfloat16 represents the top 16 bits of a float32
            result[i * 2] = 0
            result[i * 2 + 1] = 0
            result[i * 2 + 2] = bf16_bytes[i]
            result[i * 2 + 3] = bf16_bytes[i + 1]
        return bytes(result)


class MsgPackFlaxDeserializer:
    """Unpacks legacy Flax variables without importing `flax` locally."""

    @staticmethod
    def deserialize(data: bytes) -> dict[str, Any]:
        """Deserialize msgpack encoded Flax variables.

        Args:
            data: MsgPack encoded bytes.

        Returns:
            Dictionary representing the Flax variables.
        """
        import msgpack

        # Flax uses standard msgpack for older versions
        return msgpack.unpackb(data, raw=False)
