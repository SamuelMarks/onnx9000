"""Module providing core logic and structural definitions."""

import pytest
import asyncio
from onnx9000.backends.web.orchestrator import WebWorkerOrchestrator
from onnx9000.backends.web.worker import WebWorkerEnv


class DummyWorker:
    """Provides semantic functionality and verification."""

    def __init__(self, url: str):
        """Provides semantic functionality and verification."""
        self.url = url
        self.listeners = {}
        self.posted = []
        self.terminated = False

    def addEventListener(self, event, callback):
        """Provides semantic functionality and verification."""
        self.listeners[event] = callback

    def postMessage(self, msg):
        """Provides semantic functionality and verification."""
        self.posted.append(msg)

    def terminate(self):
        """Provides semantic functionality and verification."""
        self.terminated = True


class DummyJS:
    """Provides semantic functionality and verification."""

    class Worker:
        """Provides semantic functionality and verification."""

        @staticmethod
        def new(url):
            """Provides semantic functionality and verification."""
            return DummyWorker(url)

    class navigator:
        """Provides semantic functionality and verification."""

        gpu = True
        deviceMemory = 8

    class performance:
        """Provides semantic functionality and verification."""

        class memory:
            """Provides semantic functionality and verification."""

            jsHeapSizeLimit = 1000
            totalJSHeapSize = 500
            usedJSHeapSize = 250

    SharedArrayBuffer = True

    def postMessage(self, msg):
        """Provides semantic functionality and verification."""
        self.posted.append(msg)

    def addEventListener(self, event, cb):
        """Provides semantic functionality and verification."""
        self.listeners[event] = cb

    def __init__(self):
        """Provides semantic functionality and verification."""
        self.posted = []
        self.listeners = {}


def dummy_create_proxy(cb):
    """Provides semantic functionality and verification."""
    return cb


def dummy_to_js(x):
    """Provides semantic functionality and verification."""
    return x


@pytest.fixture
def mock_js(monkeypatch):
    """Provides semantic functionality and verification."""
    import onnx9000.backends.web.orchestrator as orch
    import onnx9000.backends.web.worker as worker

    d_js = DummyJS()
    monkeypatch.setattr(orch, "js", d_js)
    monkeypatch.setattr(orch, "create_proxy", dummy_create_proxy)
    monkeypatch.setattr(orch, "to_js", dummy_to_js)
    monkeypatch.setattr(worker, "js", d_js)
    monkeypatch.setattr(worker, "to_js", dummy_to_js)
    return orch


def test_init_async_and_handshake(mock_js):
    """Provides semantic functionality and verification."""

    async def run():
        """Provides semantic functionality and verification."""
        w = WebWorkerOrchestrator("dummy.js")
        init_task = asyncio.create_task(w.init_async())
        await asyncio.sleep(0.01)

        class DummyEvent:
            """Provides semantic functionality and verification."""

            def __init__(self, data):
                """Provides semantic functionality and verification."""
                self.data = data

        w._on_message(
            DummyEvent(
                {
                    "id": "0",
                    "type": "handshake",
                    "payload": {"webgpu": True},
                    "timestamp": 123.0,
                }
            )
        )
        await init_task
        assert w.capabilities["webgpu"] is True

    asyncio.run(run())


def test_send_request(mock_js):
    """Provides semantic functionality and verification."""

    async def run():
        """Provides semantic functionality and verification."""
        w = WebWorkerOrchestrator("dummy.js")
        w.init()
        req_task = asyncio.create_task(w.send_request("test", {"a": 1}))
        await asyncio.sleep(0.01)
        assert len(w.worker.posted) == 1
        msg_id = w.worker.posted[0]["id"]

        class DummyEvent:
            """Provides semantic functionality and verification."""

            def __init__(self, data):
                """Provides semantic functionality and verification."""
                self.data = data

        w._on_message(
            DummyEvent(
                {
                    "id": msg_id,
                    "type": "test_response",
                    "payload": "ok",
                    "timestamp": 123.0,
                }
            )
        )
        res = await req_task
        assert res == "ok"

    asyncio.run(run())


def test_send_request_error(mock_js):
    """Provides semantic functionality and verification."""

    async def run():
        """Provides semantic functionality and verification."""
        w = WebWorkerOrchestrator("dummy.js")
        w.init()
        req_task = asyncio.create_task(w.send_request("test", {}))
        await asyncio.sleep(0.01)
        msg_id = w.worker.posted[0]["id"]

        class DummyEvent:
            """Provides semantic functionality and verification."""

            def __init__(self, data):
                """Provides semantic functionality and verification."""
                self.data = data

        w._on_message(
            DummyEvent(
                {
                    "id": msg_id,
                    "type": "test_response",
                    "error": "failed",
                    "timestamp": 123.0,
                }
            )
        )
        with pytest.raises(RuntimeError):
            await req_task

    asyncio.run(run())


def test_send_stream(mock_js):
    """Provides semantic functionality and verification."""

    async def run():
        """Provides semantic functionality and verification."""
        w = WebWorkerOrchestrator("dummy.js")
        w.init()
        chunks = []

        def on_chunk(c):
            """Provides semantic functionality and verification."""
            chunks.append(c)

        req_task = asyncio.create_task(w.send_stream("gen", {}, on_chunk))
        await asyncio.sleep(0.01)
        msg_id = w.worker.posted[0]["id"]

        class DummyEvent:
            """Provides semantic functionality and verification."""

            def __init__(self, data):
                """Provides semantic functionality and verification."""
                self.data = data

        w._on_message(
            DummyEvent(
                {"id": msg_id, "type": "gen_stream", "payload": "A", "timestamp": 123.0}
            )
        )
        w._on_message(
            DummyEvent(
                {"id": msg_id, "type": "gen_stream", "payload": "B", "timestamp": 123.0}
            )
        )
        w._on_message(
            DummyEvent(
                {
                    "id": msg_id,
                    "type": "gen_response",
                    "payload": "AB",
                    "timestamp": 123.0,
                }
            )
        )
        res = await req_task
        assert res == "AB"
        assert chunks == ["A", "B"]

    asyncio.run(run())


def test_worker_env_boot_and_memory(monkeypatch):
    """Provides semantic functionality and verification."""
    import onnx9000.backends.web.worker as worker

    d_js = DummyJS()
    monkeypatch.setattr(worker, "js", d_js)
    monkeypatch.setattr(worker, "to_js", dummy_to_js)
    env = worker.WebWorkerEnv()
    env.boot()
    assert len(d_js.posted) == 1
    assert d_js.posted[0]["type"] == "handshake"
    assert d_js.posted[0]["payload"]["webgpu"] is True

    class DummyEvent:
        """Provides semantic functionality and verification."""

        def __init__(self, data):
            """Provides semantic functionality and verification."""
            self.data = data

    env.on_message(DummyEvent({"id": "mem", "type": "get_memory"}))
    assert len(d_js.posted) == 2
    assert d_js.posted[1]["type"] == "get_memory_response"
    assert d_js.posted[1]["payload"]["deviceMemory"] == 8
    assert d_js.posted[1]["payload"]["usedJSHeapSize"] == 250


def test_init_async_no_js(monkeypatch):
    """Provides semantic functionality and verification."""
    import onnx9000.backends.web.orchestrator as orch

    monkeypatch.setattr(orch, "js", None)

    async def run():
        """Provides semantic functionality and verification."""
        w = WebWorkerOrchestrator("dummy.js")
        await w.init_async()
        assert w.worker is None

    asyncio.run(run())
