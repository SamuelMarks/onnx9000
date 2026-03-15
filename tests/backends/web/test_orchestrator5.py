"""Module providing core logic and structural definitions."""

import pytest
import asyncio
from onnx9000.backends.web.orchestrator import WebWorkerOrchestrator
from onnx9000.backends.web.worker import WebWorkerEnv, VFSMock
from onnx9000.backends.web.rpc import serialize_fallback, deserialize_fallback


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

    class console:
        """Provides semantic functionality and verification."""

        logs = []

        @staticmethod
        def log(msg):
            """Provides semantic functionality and verification."""
            DummyJS.console.logs.append(msg)

    class Worker:
        """Provides semantic functionality and verification."""

        @staticmethod
        def new(url):
            """Provides semantic functionality and verification."""
            return DummyWorker(url)

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
        self.console = self.__class__.console


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


def test_vfs_mock():
    """Provides semantic functionality and verification."""
    vfs = VFSMock()
    vfs.write("model.bin", b"123")
    assert vfs.read("model.bin") == b"123"
    with pytest.raises(FileNotFoundError):
        vfs.read("missing.bin")


def test_debug_logging(mock_js):
    """Provides semantic functionality and verification."""
    import onnx9000.backends.web.worker as worker

    env = worker.WebWorkerEnv()
    env.set_debug(True)
    assert "WebWorkerEnv debug mode enabled" in mock_js.js.console.logs

    class DummyEvent:
        """Provides semantic functionality and verification."""

        def __init__(self, data):
            """Provides semantic functionality and verification."""
            self.data = data

    env.on_message(DummyEvent({"id": "1", "type": "test_msg"}))
    assert "Worker received: test_msg" in mock_js.js.console.logs


def test_fallback_serialization():
    """Provides semantic functionality and verification."""
    data = {"hello": "world"}
    ser = serialize_fallback(data)
    assert isinstance(ser, (str, bytes, bytearray))
    deser = deserialize_fallback(ser)
    assert deser == data


def test_soft_kill(mock_js):
    """Provides semantic functionality and verification."""

    async def run():
        """Provides semantic functionality and verification."""
        w = WebWorkerOrchestrator("dummy.js")
        w.init()
        msg_id = w.enqueue("task1", {})
        kill_task = asyncio.create_task(w.soft_kill())
        await asyncio.sleep(0.01)
        assert w.worker is not None

        class DummyEvent:
            """Provides semantic functionality and verification."""

            def __init__(self, data):
                """Provides semantic functionality and verification."""
                self.data = data

        w._on_message(DummyEvent({"id": msg_id, "type": "resp"}))
        await kill_task
        assert w.worker is None

    asyncio.run(run())
