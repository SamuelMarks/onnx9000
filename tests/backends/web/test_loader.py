"""Module providing core logic and structural definitions."""

import pytest
import asyncio
from onnx9000.backends.web.loader import (
    StreamProtobufParser,
    ONNXProgressiveLoader,
    HTTPChunkLoader,
)


async def make_stream(data: bytes, chunk_size: int = 2):
    """Provides semantic functionality and verification."""
    for i in range(0, len(data), chunk_size):
        yield data[i : i + chunk_size]


def test_stream_protobuf_parser():
    """Provides semantic functionality and verification."""

    async def run():
        """Provides semantic functionality and verification."""
        data = bytes([172, 2]) + b"hello"
        stream = make_stream(data, 2)
        parser = StreamProtobufParser(stream)
        val = await parser.read_varint()
        assert val == 300
        b = await parser.read_bytes(5)
        assert b == b"hello"

    asyncio.run(run())


def test_stream_protobuf_parser_eof():
    """Provides semantic functionality and verification."""

    async def run():
        """Provides semantic functionality and verification."""
        stream = make_stream(b"\xac", 2)
        parser = StreamProtobufParser(stream)
        with pytest.raises(EOFError):
            await parser.read_varint()
        stream2 = make_stream(b"a", 2)
        parser2 = StreamProtobufParser(stream2)
        with pytest.raises(EOFError):
            await parser2.read_bytes(5)

    asyncio.run(run())


def test_progressive_loader():
    """Provides semantic functionality and verification."""

    async def run():
        """Provides semantic functionality and verification."""
        loader = ONNXProgressiveLoader("dummy.onnx")
        data = bytes([8, 5, 18, 3]) + b"abc" + bytes([186, 62, 4]) + b"data"
        stream = make_stream(data, 3)
        await loader.parse_topology(stream)
        assert "mock" in loader.tensors_manifest
        assert loader.tensors_manifest["mock"]["length"] == 4

    asyncio.run(run())


def test_progressive_loader_eof():
    """Provides semantic functionality and verification."""

    async def run():
        """Provides semantic functionality and verification."""
        loader = ONNXProgressiveLoader("dummy.onnx")
        data = bytes([15])
        stream = make_stream(data, 3)
        with pytest.raises(ValueError):
            await loader.parse_topology(stream)
        data = bytes([13, 1, 2, 3, 4])
        stream = make_stream(data, 3)
        await loader.parse_topology(stream)

    asyncio.run(run())


def test_http_chunk_loader():
    """Provides semantic functionality and verification."""

    async def run():
        """Provides semantic functionality and verification."""
        loader = HTTPChunkLoader("dummy.bin")
        data = await loader.fetch_chunk(0, 100)
        assert data == b"real_chunk_data"

    asyncio.run(run())
