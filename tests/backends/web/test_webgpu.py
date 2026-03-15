"""Module providing core logic and structural definitions."""

import pytest
import asyncio
from onnx9000.backends.web.webgpu import (
    WebGPUCore,
    WebGPUMemoryPool,
    TensorBuffer,
    WebGPULimits,
)


class MockGPUBuffer:
    """Provides semantic functionality and verification."""

    def __init__(self):
        """Provides semantic functionality and verification."""
        self.destroyed = False

    def destroy(self):
        """Provides semantic functionality and verification."""
        self.destroyed = True


class MockDevice:
    """Provides semantic functionality and verification."""

    def __init__(self):
        """Provides semantic functionality and verification."""
        self.limits = type("Limits", (), {"maxBufferSize": 1024, "maxBindGroups": 4})()

    def createCommandEncoder(self):
        """Provides semantic functionality and verification."""
        return "mock_encoder"


class MockAdapter:
    """Provides semantic functionality and verification."""

    async def requestDevice(self, opts):
        """Provides semantic functionality and verification."""
        return MockDevice()


class MockGPU:
    """Provides semantic functionality and verification."""

    class wgslLanguageFeatures:
        """Provides semantic functionality and verification."""

        @staticmethod
        def has(feat):
            """Provides semantic functionality and verification."""
            return True

    async def requestAdapter(self, opts):
        """Provides semantic functionality and verification."""
        return MockAdapter()


class DummyNav:
    """Provides semantic functionality and verification."""

    gpu = MockGPU()


class DummyJS:
    """Provides semantic functionality and verification."""

    navigator = DummyNav()


@pytest.fixture
def mock_js(monkeypatch):
    """Provides semantic functionality and verification."""
    import onnx9000.backends.web.webgpu as webgpu

    monkeypatch.setattr(webgpu, "js", DummyJS())


def test_tensor_buffer():
    """Provides semantic functionality and verification."""
    b = TensorBuffer(100, 1, MockGPUBuffer())
    assert b.size == 100
    assert not b.in_use
    b.destroy()
    assert b.gpu_buffer.destroyed


def test_memory_pool():
    """Provides semantic functionality and verification."""
    p = WebGPUMemoryPool("mock_device")
    b1 = p.allocate(100, 1)
    assert b1.size == 256
    assert b1.in_use
    b2 = p.allocate(256, 1)
    assert b2.size == 256
    p.free(b1)
    assert not b1.in_use
    assert len(p.free_buffers[256]) == 1
    b3 = p.allocate(100, 1)
    assert b3 is b1
    assert b3.in_use
    b4 = p.allocate(500, 1)
    p.free(b4)
    b4.gpu_buffer = MockGPUBuffer()
    p.defragment()
    assert b4.gpu_buffer.destroyed
    assert len(p.free_buffers) == 0


def test_webgpu_core(mock_js):
    """Provides semantic functionality and verification."""

    async def run():
        """Provides semantic functionality and verification."""
        core = WebGPUCore()
        success = await core.init()
        assert success is True
        assert core.adapter is not None
        assert core.device is not None
        assert core.limits.max_buffer_size == 1024
        assert core.has_f16 is True
        core.create_command_encoder()
        assert core.command_encoder == "mock_encoder"
        assert core.get_storage_format("float32") == "f32"
        assert core.get_storage_format("float16") == "f16"
        assert core.get_storage_format("int32") == "i32"
        assert core.get_storage_format("uint32") == "u32"
        assert core.get_storage_format("unknown") == "u32"

    asyncio.run(run())


def test_webgpu_fallback(monkeypatch):
    """Provides semantic functionality and verification."""
    import onnx9000.backends.web.webgpu as webgpu

    monkeypatch.setattr(webgpu, "js", None)

    async def run():
        """Provides semantic functionality and verification."""
        core = WebGPUCore()
        success = await core.init()
        assert success is False

    asyncio.run(run())
