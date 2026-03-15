"""Module providing core logic and structural definitions."""

import pytest
from onnx9000.backends.web.orchestrator import WebWorkerOrchestrator
from onnx9000.backends.web.rpc import RPCMessage


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

    monkeypatch.setattr(orch, "js", DummyJS())
    monkeypatch.setattr(orch, "create_proxy", dummy_create_proxy)
    monkeypatch.setattr(orch, "to_js", dummy_to_js)


def test_orchestrator_init(mock_js):
    """Provides semantic functionality and verification."""
    w = WebWorkerOrchestrator("dummy.js")
    w.init()
    assert w.worker is not None
    assert w.worker.url == "dummy.js"
    assert "message" in w.worker.listeners
    assert "error" in w.worker.listeners


def test_orchestrator_lifecycle(mock_js):
    """Provides semantic functionality and verification."""
    w = WebWorkerOrchestrator("dummy.js")
    w.init()
    worker_ref = w.worker
    assert not worker_ref.terminated
    w.terminate()
    assert worker_ref.terminated
    assert w.worker is None
    w.restart()
    assert w.worker is not None
    assert w.worker is not worker_ref


def test_orchestrator_error_oom(mock_js):
    """Provides semantic functionality and verification."""
    w = WebWorkerOrchestrator("dummy.js")
    w.init()

    class DummyErrorEvent:
        """Provides semantic functionality and verification."""

        def __init__(self, msg):
            """Provides semantic functionality and verification."""
            self.message = msg

    assert w._oom_restarts == 0
    w._on_error(DummyErrorEvent("out of memory occurred"))
    assert w._oom_restarts == 1
    assert w.worker is not None
    w._on_error(DummyErrorEvent("normal error"))
    assert w._oom_restarts == 1


def test_orchestrator_queue_priority(mock_js):
    """Provides semantic functionality and verification."""
    w = WebWorkerOrchestrator("dummy.js")
    w.init()
    msg1 = w.enqueue("task1", {}, priority=1)
    assert w._active_task_id is not None
    msg2 = w.enqueue("task2", {}, priority=2)
    msg3 = w.enqueue("task3", {}, priority=0)
    assert len(w._task_queue) == 2
    assert len(w.worker.posted) == 1

    class DummyEvent:
        """Provides semantic functionality and verification."""

        def __init__(self, data):
            """Provides semantic functionality and verification."""
            self.data = data

    w._on_message(DummyEvent({"id": msg1, "type": "resp"}))
    assert len(w.worker.posted) == 2
    assert w.worker.posted[1]["type"] == "task3"
    assert w._active_task_id is not None


def test_orchestrator_to_py_conversion(mock_js):
    """Provides semantic functionality and verification."""
    w = WebWorkerOrchestrator("dummy.js")
    w.init()
    result = []

    def callback(msg):
        """Provides semantic functionality and verification."""
        result.append(msg)

    msg_id = w.send("test", {})
    w._pending_requests[msg_id] = callback

    class ToPyData:
        """Provides semantic functionality and verification."""

        def __init__(self, data):
            """Provides semantic functionality and verification."""
            self._data = data

        def to_py(self):
            """Provides semantic functionality and verification."""
            return self._data

    class DummyEvent:
        """Provides semantic functionality and verification."""

        def __init__(self, data):
            """Provides semantic functionality and verification."""
            self.data = data

    js_data = ToPyData({"id": msg_id, "type": "test_to_py", "payload": "converted"})
    w._on_message(DummyEvent(js_data))
    assert len(result) == 1
    assert result[0].payload == "converted"
    assert result[0].type == "test_to_py"


def test_orchestrator_send_no_js(monkeypatch):
    """Provides semantic functionality and verification."""
    import onnx9000.backends.web.orchestrator as orch

    monkeypatch.setattr(orch, "js", None)
    w = WebWorkerOrchestrator("dummy.js")
    w.init()
    msg_id = w.send("test", {})
    assert isinstance(msg_id, str)
