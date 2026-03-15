"""Module providing core logic and structural definitions."""

import pytest
import asyncio
from onnx9000.backends.web.wasm import WasmRuntime, WASMOrchestrator, HybridExecutor


class DummyJS:
    """Provides semantic functionality and verification."""

    WebAssembly = True

    class Float32Array:
        """Provides semantic functionality and verification."""

        @staticmethod
        def new(buf, off, length):
            """Provides semantic functionality and verification."""
            return "mock_view"


@pytest.fixture
def mock_js(monkeypatch):
    """Provides semantic functionality and verification."""
    import onnx9000.backends.web.wasm as wasm

    monkeypatch.setattr(wasm, "js", DummyJS())


def test_wasm_runtime(mock_js):
    """Provides semantic functionality and verification."""

    async def run():
        """Provides semantic functionality and verification."""
        r = WasmRuntime()
        assert await r.init() is True

        class MockMemory:
            """Provides semantic functionality and verification."""

            buffer = "buf"

        r.memory = MockMemory()
        v = r.get_memory_view(0, 100)
        assert v == "mock_view"

    asyncio.run(run())


def test_wasm_fallback(monkeypatch):
    """Provides semantic functionality and verification."""
    import onnx9000.backends.web.wasm as wasm

    monkeypatch.setattr(wasm, "js", None)

    async def run():
        """Provides semantic functionality and verification."""
        r = WasmRuntime()
        assert await r.init() is False
        assert r.get_memory_view(0, 10) is None

    asyncio.run(run())


def test_wasm_orchestrator():
    """Provides semantic functionality and verification."""
    o = WASMOrchestrator()
    assert o.track_memory_expansion() == 0
    assert o.get_thread_pool() is None
    assert o.extract_stack_trace() == "stack trace mock"

    async def run():
        """Provides semantic functionality and verification."""
        await o.run_op("Add", [], [])

    asyncio.run(run())


def test_hybrid_executor():
    """Provides semantic functionality and verification."""
    h = HybridExecutor(WASMOrchestrator(), "gpu")
    h.execute_hybrid("node")
