"""Module providing core logic and structural definitions."""

import pytest
import asyncio
from unittest.mock import MagicMock
from onnx9000.backends.web.webgpu import (
    WebGPULimits,
    WebGPUCore,
    WebGPUDeviceEvents,
    BufferUtils,
)


def test_webgpu_limits_empty():
    """Provides semantic functionality and verification."""
    limits = WebGPULimits(None)
    assert limits.max_buffer_size == 0
    assert limits.max_bind_groups == 0


def test_webgpu_core_no_adapter(monkeypatch):
    """Provides semantic functionality and verification."""
    import onnx9000.backends.web.webgpu as webgpu

    class MockGPU:
        """Provides semantic functionality and verification."""

        async def requestAdapter(self, opts):
            """Provides semantic functionality and verification."""
            return None

    class DummyNav:
        """Provides semantic functionality and verification."""

        gpu = MockGPU()

    class DummyJS:
        """Provides semantic functionality and verification."""

        navigator = DummyNav()

    monkeypatch.setattr(webgpu, "js", DummyJS())

    async def run():
        """Provides semantic functionality and verification."""
        core = WebGPUCore()
        success = await core.init()
        assert success is False

    asyncio.run(run())


def test_webgpu_device_events(monkeypatch):
    """Provides semantic functionality and verification."""
    import onnx9000.backends.web.webgpu as webgpu

    mock_console = MagicMock()

    class DummyJS:
        """Provides semantic functionality and verification."""

        console = mock_console

    monkeypatch.setattr(webgpu, "js", DummyJS())

    class MockLostInfo:
        """Provides semantic functionality and verification."""

        def __init__(self):
            """Provides semantic functionality and verification."""
            self.reason = "test reason"
            self.message = "test message"

    class MockDevice:
        """Provides semantic functionality and verification."""

        @property
        def lost(self):
            """Provides semantic functionality and verification."""
            future = asyncio.Future()
            future.set_result(MockLostInfo())
            return future

    device = MockDevice()
    on_lost = MagicMock()

    async def run():
        """Provides semantic functionality and verification."""
        await WebGPUDeviceEvents.monitor_device_lost(device, on_lost)

    asyncio.run(run())
    mock_console.warn.assert_called_once_with(
        "GPUDevice lost: test reason - test message"
    )
    on_lost.assert_called_once()


def test_async_readback_js(monkeypatch):
    """Provides semantic functionality and verification."""
    import onnx9000.backends.web.webgpu as webgpu

    mock_uint8array = MagicMock()
    mock_uint8array.new.return_value = [1, 2, 3]

    class DummyJS:
        """Provides semantic functionality and verification."""

        Uint8Array = mock_uint8array

    monkeypatch.setattr(webgpu, "js", DummyJS())

    class MockBuffer:
        """Provides semantic functionality and verification."""

        async def mapAsync(self, mode):
            """Provides semantic functionality and verification."""
            pass

        def getMappedRange(self, offset, size):
            """Provides semantic functionality and verification."""
            return "dummy_data"

        def unmap(self):
            """Provides semantic functionality and verification."""
            pass

    buffer = MockBuffer()

    async def run():
        """Provides semantic functionality and verification."""
        res = await BufferUtils.map_to_js(buffer, 3)
        return res

    res = asyncio.run(run())
    mock_uint8array.new.assert_called_once_with("dummy_data")
    assert res == bytes([1, 2, 3])
