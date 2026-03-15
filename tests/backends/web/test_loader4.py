"""Module providing core logic and structural definitions."""

import pytest
import asyncio
from onnx9000.backends.web.loader import (
    BandwidthEstimator,
    RetryBackoff,
    DecompressionStream,
    ChunkMerger,
    VisualDebugger,
    ProgressiveLoaderAPI,
    EndiannessHandler,
    QuotaFallbackCache,
    ProgressiveOrchestrator,
    ResumableChunkLoader,
    ManifestParser,
)


def test_bandwidth_estimator():
    """Provides semantic functionality and verification."""
    b = BandwidthEstimator()
    b.record(1000, 10.0)
    assert b.estimate_bytes_per_ms() == 100.0


def test_retry_backoff():
    """Provides semantic functionality and verification."""

    async def run():
        """Provides semantic functionality and verification."""
        calls = []

        async def fail_twice():
            """Provides semantic functionality and verification."""
            calls.append(1)
            if len(calls) < 3:
                raise ValueError("fail")
            return "ok"

        res = await RetryBackoff.execute(fail_twice, 3)
        assert res == "ok"
        assert len(calls) == 3

        async def fail_always():
            """Provides semantic functionality and verification."""
            raise ValueError("fail")

        with pytest.raises(ValueError):
            await RetryBackoff.execute(fail_always, 2)

    asyncio.run(run())


def test_decompression():
    """Provides semantic functionality and verification."""
    import gzip

    data = b"hello world"
    compressed = gzip.compress(data)
    assert DecompressionStream.decompress_gzip(compressed) == data


def test_chunk_merger():
    """Provides semantic functionality and verification."""
    assert ChunkMerger.merge([b"a", b"b", b"c"]) == b"abc"


def test_visual_debugger():
    """Provides semantic functionality and verification."""
    m = ManifestParser()
    m.tensors = {"w1": {}, "w2": {}}
    assert "2 tensors" in VisualDebugger.generate_html(m)


def test_progressive_loader_api():
    """Provides semantic functionality and verification."""
    api = ProgressiveLoaderAPI()
    assert api.ui_state == "init"
    api.step()
    assert api.ui_state == "loading"


def test_endianness_handler():
    """Provides semantic functionality and verification."""
    assert isinstance(EndiannessHandler.is_little_endian(), bool)


def test_quota_fallback():
    """Provides semantic functionality and verification."""

    async def run():
        """Provides semantic functionality and verification."""
        c = QuotaFallbackCache()

        async def put_mock(self, key, data, etag):
            """Provides semantic functionality and verification."""
            if data == b"too_big":
                raise MemoryError("quota")
            self.cache[key] = data

        import onnx9000.backends.web.loader as l

        original_put = l.IDBCache.put_chunk
        l.IDBCache.put_chunk = put_mock
        await c.put_chunk("w1", b"ok", "1")
        assert "w1" in c.cache
        await c.put_chunk("w2", b"too_big", "2")
        assert "w2" in c.cache
        l.IDBCache.put_chunk = original_put

    asyncio.run(run())


def test_progressive_orchestrator():
    """Provides semantic functionality and verification."""
    o = ProgressiveOrchestrator(ResumableChunkLoader("test"))
    m = ManifestParser()
    m.tensors = {"layer1": {}, "layer2": {}}
    assert o.prioritize_layers(m) == ["layer1", "layer2"]
    o.clear_buffers()
    o.load_minimal()

    async def run():
        """Provides semantic functionality and verification."""
        await o.prefetch("layer2")

    asyncio.run(run())


def test_massive_64bit_offset():
    """Provides semantic functionality and verification."""
    m = ManifestParser()
    m.parse({"tensors": {"w1": {"offset": 5 * 1024**3, "length": 100}}})
    assert m.tensors["w1"]["offset"] == 5 * 1024**3
