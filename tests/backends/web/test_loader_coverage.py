"""Module providing core logic and structural definitions."""

import pytest
import asyncio
from onnx9000.backends.web.loader import ONNXProgressiveLoader, HTTPPool


def test_loader_line_73():
    """Provides semantic functionality and verification."""

    async def run():
        """Provides semantic functionality and verification."""

        class MockStream:
            """Provides semantic functionality and verification."""

            def __init__(self, data):
                """Provides semantic functionality and verification."""
                self.data = data
                self.pos = 0

            async def __anext__(self):
                """Provides semantic functionality and verification."""
                if self.pos >= len(self.data):
                    raise StopAsyncIteration
                chunk = self.data[self.pos : self.pos + 1]
                self.pos += 1
                return chunk

            def __aiter__(self):
                """Provides semantic functionality and verification."""
                return self

        loader = ONNXProgressiveLoader("http://abc")
        tag_bytes = bytes([9])
        data_bytes = b"12345678"
        stream = MockStream(tag_bytes + data_bytes)
        await loader.parse_topology(stream)

    asyncio.run(run())


def test_loader_line_332():
    """Provides semantic functionality and verification."""
    pool = HTTPPool(size=10)
    assert pool.size == 10


from unittest.mock import MagicMock, patch
from onnx9000.backends.web.loader import (
    HTTPChunkLoader,
    ResumableChunkLoader,
    LocalBlobLoader,
    RetryBackoff,
)
import onnx9000.backends.web.loader as loader_module


def test_loader_line_100_105():
    """Provides semantic functionality and verification."""

    async def run():
        """Provides semantic functionality and verification."""
        with patch.object(loader_module, "js", MagicMock()) as mock_js:
            mock_resp = MagicMock()
            mock_resp.status = 206

            async def mock_arrayBuffer():
                """Provides semantic functionality and verification."""
                return b"buf"

            mock_resp.arrayBuffer = mock_arrayBuffer

            async def mock_fetch(url, opts):
                """Provides semantic functionality and verification."""
                return mock_resp

            mock_js.fetch = mock_fetch
            mock_js.Uint8Array.new = lambda x: b"bytes"
            chunk_loader = HTTPChunkLoader("http://example.com")
            res = await chunk_loader.fetch_chunk(0, 10)
            assert res == b"bytes"
            mock_resp.status = 404
            with pytest.raises(RuntimeError, match="HTTP 404"):
                await chunk_loader.fetch_chunk(0, 10)

    asyncio.run(run())


def test_loader_line_229_236():
    """Provides semantic functionality and verification."""

    async def run():
        """Provides semantic functionality and verification."""
        with patch.object(loader_module, "js", MagicMock()) as mock_js:

            async def mock_fetch_416(url, opts):
                """Provides semantic functionality and verification."""
                raise Exception("HTTP 416")

            mock_js.fetch = mock_fetch_416
            res_loader = ResumableChunkLoader("http://example.com")
            res = await res_loader.fetch_chunk(0, 10)
            assert res == b""

            async def mock_fetch_cors(url, opts):
                """Provides semantic functionality and verification."""
                raise Exception("CORS error")

            mock_js.fetch = mock_fetch_cors
            with pytest.raises(
                RuntimeError, match="CORS error: please configure your server"
            ):
                await res_loader.fetch_chunk(0, 10)

            async def mock_fetch_other(url, opts):
                """Provides semantic functionality and verification."""
                raise ValueError("other")

            mock_js.fetch = mock_fetch_other
            with pytest.raises(ValueError, match="other"):
                await res_loader.fetch_chunk(0, 10)

    asyncio.run(run())


def test_loader_line_247_248():
    """Provides semantic functionality and verification."""

    async def run():
        """Provides semantic functionality and verification."""
        with patch.object(loader_module, "js", MagicMock()) as mock_js:
            mock_blob = MagicMock()

            async def mock_arrayBuffer():
                """Provides semantic functionality and verification."""
                return b"buf"

            mock_blob.arrayBuffer = mock_arrayBuffer
            mock_js.Uint8Array.new = lambda x: b"blob_bytes"
            blob_loader = LocalBlobLoader(mock_blob)
            res = await blob_loader.read()
            assert res == b"blob_bytes"

    asyncio.run(run())


def test_loader_line_332_retry():
    """Provides semantic functionality and verification."""

    async def run():
        """Provides semantic functionality and verification."""
        attempts = [0]

        async def mock_func():
            """Provides semantic functionality and verification."""
            attempts[0] += 1
            if attempts[0] < 3:
                raise ValueError("fail")
            return "success"

        res = await RetryBackoff.execute(mock_func, max_retries=3)
        assert res == "success"
        assert attempts[0] == 3
        attempts[0] = 0

        async def mock_func_fail():
            """Provides semantic functionality and verification."""
            attempts[0] += 1
            raise ValueError("fail")

        with pytest.raises(ValueError, match="fail"):
            await RetryBackoff.execute(mock_func_fail, max_retries=3)

    asyncio.run(run())
