"""Module providing core logic and structural definitions."""

import pytest
import asyncio
from onnx9000.backends.web.webgpu import (
    PipelineCache,
    WebGPUDeviceEvents,
    WGSLGenerators,
    ExecutionProfiler,
    BufferUtils,
)


class MockDeviceForPipeline:
    """Provides semantic functionality and verification."""

    def __init__(self):
        """Provides semantic functionality and verification."""
        self.modules = []

    def createShaderModule(self, desc):
        """Provides semantic functionality and verification."""
        if "fail" in desc["code"]:
            raise RuntimeError("compilation error")
        self.modules.append(desc)
        return "mock_module"

    async def createComputePipelineAsync(self, desc):
        """Provides semantic functionality and verification."""
        return "mock_pipeline_" + desc["label"]


def test_pipeline_cache():
    """Provides semantic functionality and verification."""

    async def run():
        """Provides semantic functionality and verification."""
        d = MockDeviceForPipeline()
        c = PipelineCache(d)
        p = await c.get_compute_pipeline("good code", "my_shader")
        assert p == "mock_pipeline_my_shader_pipeline"
        assert len(d.modules) == 1
        p2 = await c.get_compute_pipeline("good code", "my_shader")
        assert p2 == "mock_pipeline_my_shader_pipeline"
        assert len(d.modules) == 1
        with pytest.raises(RuntimeError):
            await c.get_compute_pipeline("fail code", "bad_shader")

    asyncio.run(run())


def test_device_events():
    """Provides semantic functionality and verification."""

    async def run():
        """Provides semantic functionality and verification."""
        calls = []

        def on_lost():
            """Provides semantic functionality and verification."""
            calls.append(1)

        class LostInfo:
            """Provides semantic functionality and verification."""

            reason = "test"
            message = "test msg"

        class MockLostDevice:
            """Provides semantic functionality and verification."""

            def __init__(self):
                """Provides semantic functionality and verification."""
                loop = asyncio.get_running_loop()
                f = loop.create_future()
                f.set_result(LostInfo())
                self.lost = f

        await WebGPUDeviceEvents.monitor_device_lost(MockLostDevice(), on_lost)
        assert len(calls) == 1

    asyncio.run(run())


def test_wgsl_generators():
    """Provides semantic functionality and verification."""
    g = WGSLGenerators()
    assert "get_linear_index" in g.get_linear_index_wgsl()
    assert "broadcast_indices" in g.get_broadcast_wgsl()


def test_execution_profiler():
    """Provides semantic functionality and verification."""

    class MockFeatures:
        """Provides semantic functionality and verification."""

        def has(self, f):
            """Provides semantic functionality and verification."""
            return f == "timestamp-query"

    class MockDeviceProf:
        """Provides semantic functionality and verification."""

        features = MockFeatures()

        def createQuerySet(self, desc):
            """Provides semantic functionality and verification."""
            return "query_set_obj"

    p = ExecutionProfiler(MockDeviceProf())
    assert p.query_set == "query_set_obj"

    class MockEncoder:
        """Provides semantic functionality and verification."""

        def __init__(self):
            """Provides semantic functionality and verification."""
            self.written = []

        def writeTimestamp(self, qs, idx):
            """Provides semantic functionality and verification."""
            self.written.append((qs, idx))

    e = MockEncoder()
    p.record_start(e, 0)
    p.record_end(e, 1)
    assert e.written == [("query_set_obj", 0), ("query_set_obj", 1)]
    assert p.generate_flamegraph() == {"traceEvents": []}


def test_buffer_utils():
    """Provides semantic functionality and verification."""

    async def run():
        """Provides semantic functionality and verification."""

        class MockEncoderC:
            """Provides semantic functionality and verification."""

            def __init__(self):
                """Provides semantic functionality and verification."""
                self.cleared = []

            def clearBuffer(self, b, off, sz):
                """Provides semantic functionality and verification."""
                self.cleared.append((b, off, sz))

        e = MockEncoderC()
        BufferUtils.clear_buffer(e, "buf1", 100)
        assert e.cleared == [("buf1", 0, 100)]

        class MockBuf:
            """Provides semantic functionality and verification."""

            def __init__(self):
                """Provides semantic functionality and verification."""
                self.mapped = False

            async def mapAsync(self, mode):
                """Provides semantic functionality and verification."""
                self.mapped = True

            def getMappedRange(self, off, sz):
                """Provides semantic functionality and verification."""
                return b"test_data"

            def unmap(self):
                """Provides semantic functionality and verification."""
                self.mapped = False

        b = MockBuf()
        res = await BufferUtils.map_to_js(b, 9)
        assert res == b"mock"
        assert not b.mapped

    asyncio.run(run())
