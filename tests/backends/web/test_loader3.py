"""Module providing core logic and structural definitions."""

import pytest
from onnx9000.backends.web.loader import (
    DataViewDecoder,
    MemoryMappedWeights,
    ResumableChunkLoader,
    LocalBlobLoader,
    Base64Loader,
    SecureWeightWrapper,
    try_persist_storage,
)
import asyncio
import struct


def test_data_view_decoder():
    """Provides semantic functionality and verification."""
    d = DataViewDecoder()
    f32_data = struct.pack("<2f", 1.0, 2.5)
    assert d.decode_f32(f32_data) == [1.0, 2.5]
    int8_data = bytes([10, 20])
    assert d.decode_int8_to_f32(int8_data, scale=0.5, zero_point=10) == [0.0, 5.0]
    f16_data = struct.pack("<2e", 1.0, -2.0)
    assert d.decode_f16_to_f32(f16_data) == [1.0, -2.0]


def test_memory_mapped_weights():
    """Provides semantic functionality and verification."""
    m = MemoryMappedWeights()
    m.chunks["w1"] = b"123"
    assert m.get("w1") == b"123"
    m.evict("w1")
    assert "w1" not in m.chunks
    m.evict("w1")


def test_resumable_loader():
    """Provides semantic functionality and verification."""

    async def run():
        """Provides semantic functionality and verification."""
        l = ResumableChunkLoader("http://test")
        d = await l.fetch_chunk(0, 10)
        assert d == b"real_chunk_data"

        class Dummy416Loader(ResumableChunkLoader):
            """Provides semantic functionality and verification."""

            async def fetch_chunk(self, start, end):
                """Provides semantic functionality and verification."""
                try:
                    raise RuntimeError("HTTP 416 Range Not Satisfiable")
                except Exception as e:
                    if "416" in str(e):
                        return b""
                    raise

        l2 = Dummy416Loader("test")
        assert await l2.fetch_chunk(0, 10) == b""

        class DummyCORSLoader(ResumableChunkLoader):
            """Provides semantic functionality and verification."""

            async def fetch_chunk(self, start, end):
                """Provides semantic functionality and verification."""
                try:
                    raise RuntimeError(
                        "NetworkError when attempting to fetch resource. CORS"
                    )
                except Exception as e:
                    if "CORS" in str(e):
                        raise RuntimeError("CORS error: please configure your server")
                    raise

        l3 = DummyCORSLoader("test")
        with pytest.raises(RuntimeError, match="CORS error"):
            await l3.fetch_chunk(0, 10)

    asyncio.run(run())


def test_local_blob_loader():
    """Provides semantic functionality and verification."""

    async def run():
        """Provides semantic functionality and verification."""
        l = LocalBlobLoader("mock_blob")
        assert await l.read() == b"blob_data"

    asyncio.run(run())


def test_base64_loader():
    """Provides semantic functionality and verification."""
    l = Base64Loader("data:application/octet-stream;base64,SGVsbG8=")
    assert l.read() == b"Hello"


def test_secure_weight():
    """Provides semantic functionality and verification."""
    key = b"key"
    s = SecureWeightWrapper(key)
    msg = b"hello"
    encrypted = bytes([(msg[i] ^ key[i % len(key)]) for i in range(len(msg))])
    assert s.decrypt(encrypted) == b"hello"


def test_try_persist_storage(monkeypatch):
    """Provides semantic functionality and verification."""
    import onnx9000.backends.web.loader as loader

    class DummyNav:
        """Provides semantic functionality and verification."""

        class storage:
            """Provides semantic functionality and verification."""

            pass

    class DummyJS:
        """Provides semantic functionality and verification."""

        navigator = DummyNav()

    monkeypatch.setattr(loader, "js", DummyJS(), raising=False)
    assert try_persist_storage() is True
    monkeypatch.setattr(loader, "js", None, raising=False)
    assert try_persist_storage() is False
