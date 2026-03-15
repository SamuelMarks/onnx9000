"""Module providing core logic and structural definitions."""

import pytest
from onnx9000.backends.web.loader import (
    ManifestParser,
    ProgressAPI,
    IDBCache,
    AdaptiveLoader,
    dry_run_memory_estimate,
)
import asyncio


def test_manifest_parser():
    """Provides semantic functionality and verification."""
    p = ManifestParser()
    p.parse({"tensors": {"w1": {"offset": 0, "length": 100}}})
    assert "w1" in p.tensors
    assert p.tensors["w1"]["length"] == 100


def test_progress_api():
    """Provides semantic functionality and verification."""
    calls = []

    def cb(pct):
        """Provides semantic functionality and verification."""
        calls.append(pct)

    p = ProgressAPI(cb)
    p.set_total(100)
    p.add_loaded(25)
    p.add_loaded(25)
    assert calls == [0.0, 0.25, 0.5]


def test_idb_cache():
    """Provides semantic functionality and verification."""

    async def run():
        """Provides semantic functionality and verification."""
        c = IDBCache(version="v1")
        await c.put_chunk("w1", b"data1", "etag1")
        d = await c.get_chunk("w1", "etag1")
        assert d == b"data1"
        d = await c.get_chunk("w1", "etag2")
        assert d is None
        c.version = "v2"
        d = await c.get_chunk("w1", "etag1")
        assert d is None

    asyncio.run(run())


def test_adaptive_loader():
    """Provides semantic functionality and verification."""
    l = AdaptiveLoader()
    s1 = l.get_next_size(50)
    assert s1 == 2 * 1024 * 1024
    s2 = l.get_next_size(600)
    assert s2 == 1024 * 1024
    s3 = l.get_next_size(200)
    assert s3 == 1024 * 1024


def test_dry_run_memory_estimate():
    """Provides semantic functionality and verification."""
    p = ManifestParser()
    p.parse({"tensors": {"w1": {"length": 100}, "w2": {"length": 200}}})
    mem = dry_run_memory_estimate(p)
    assert mem == 300
