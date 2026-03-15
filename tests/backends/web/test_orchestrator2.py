"""Module providing core logic and structural definitions."""

import pytest
import time
from onnx9000.backends.web.orchestrator import WebWorkerOrchestrator, WorkerPool
from onnx9000.backends.web.rpc import CancellationToken


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
    return orch


def test_heartbeat_and_timeout(mock_js):
    """Provides semantic functionality and verification."""
    w = WebWorkerOrchestrator("dummy.js")
    w.init()
    assert w.check_health() is True
    w._last_heartbeat -= 10.0
    assert w.check_health() is False
    assert w.worker is not None


def test_heartbeat_message(mock_js):
    """Provides semantic functionality and verification."""
    w = WebWorkerOrchestrator("dummy.js")
    w.init()
    old_hb = w._last_heartbeat
    time.sleep(0.01)

    class DummyEvent:
        """Provides semantic functionality and verification."""

        def __init__(self, data):
            """Provides semantic functionality and verification."""
            self.data = data

    w._on_message(DummyEvent({"id": "hb", "type": "heartbeat"}))
    assert w._last_heartbeat > old_hb


def test_cancellation_in_queue(mock_js):
    """Provides semantic functionality and verification."""
    w = WebWorkerOrchestrator("dummy.js")
    w.init()
    token = CancellationToken()
    msg1 = w.enqueue("task1", {})
    msg2 = w.enqueue("task2", {}, priority=2, token=token)
    assert w._active_task_id is not None
    token.cancel()

    class DummyEvent:
        """Provides semantic functionality and verification."""

        def __init__(self, data):
            """Provides semantic functionality and verification."""
            self.data = data

    w._on_message(DummyEvent({"id": msg1, "type": "resp"}))
    assert not w._active_task_id is not None
    assert len(w.worker.posted) == 1


def test_cancellation_active(mock_js):
    """Provides semantic functionality and verification."""
    w = WebWorkerOrchestrator("dummy.js")
    w.init()
    token = CancellationToken()
    msg1 = w.enqueue("task1", {}, token=token)
    token.cancel()
    assert len(w.worker.posted) == 2
    assert w.worker.posted[1]["type"] == "cancel"


def test_worker_pool(mock_js):
    """Provides semantic functionality and verification."""
    pool = WorkerPool(2, "dummy.js")
    pool.init()
    msg1 = pool.enqueue("task1", {})
    msg2 = pool.enqueue("task2", {})
    msg3 = pool.enqueue("task3", {})
    assert len(pool.workers[0]._task_queue) + len(pool.workers[0].worker.posted) > 0
    assert len(pool.workers[1]._task_queue) + len(pool.workers[1].worker.posted) > 0


def test_check_health_no_worker(mock_js):
    """Provides semantic functionality and verification."""
    w = WebWorkerOrchestrator("dummy.js")
    assert w.check_health() is False
