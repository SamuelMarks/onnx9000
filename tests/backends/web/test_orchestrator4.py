"""Module providing core logic and structural definitions."""

import pytest
import asyncio
from onnx9000.backends.web.orchestrator import WebWorkerOrchestrator
from onnx9000.backends.web.worker import WebWorkerEnv, AtomicLock
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


def test_message_sequence_preservation(mock_js):
    """Provides semantic functionality and verification."""
    w = WebWorkerOrchestrator("dummy.js")
    w.init()
    msg1 = w.enqueue("task1", {"idx": 1}, priority=2)
    msg2 = w.enqueue("task2", {"idx": 2}, priority=2)
    msg3 = w.enqueue("task3", {"idx": 3}, priority=2)
    assert w.worker.posted[0]["type"] == "task1"

    class DummyEvent:
        """Provides semantic functionality and verification."""

        def __init__(self, data):
            """Provides semantic functionality and verification."""
            self.data = data

    w._on_message(
        DummyEvent(
            {"id": msg1, "type": "task1_response", "payload": "ok", "timestamp": 123.0}
        )
    )
    assert w.worker.posted[1]["type"] == "task2"
    w._on_message(
        DummyEvent(
            {"id": msg2, "type": "task2_response", "payload": "ok", "timestamp": 123.0}
        )
    )
    assert w.worker.posted[2]["type"] == "task3"


def test_shared_array_buffer_fallback(monkeypatch):
    """Provides semantic functionality and verification."""
    import onnx9000.backends.web.worker as worker

    class NoSABJS:
        """Provides semantic functionality and verification."""

        pass

    monkeypatch.setattr(worker, "js", NoSABJS())
    lock = AtomicLock()
    assert lock.sab is None
    assert lock.int32_array is None
    lock.lock()
    lock.unlock()


def test_batched_message_dispatch(mock_js):
    """Provides semantic functionality and verification."""
    env = WebWorkerEnv()
    env.register("batch", lambda payloads: [("done_" + p) for p in payloads])

    class DummyEvent:
        """Provides semantic functionality and verification."""

        def __init__(self, data):
            """Provides semantic functionality and verification."""
            self.data = data

    env.on_message(
        DummyEvent({"id": "batch_1", "type": "batch", "payload": ["A", "B", "C"]})
    )
    assert len(mock_js.js.posted) == 1
    resp = mock_js.js.posted[0]
    assert resp["type"] == "batch_response"
    assert resp["payload"] == ["done_A", "done_B", "done_C"]


def test_rpc_message_latency_profiling(mock_js):
    """Provides semantic functionality and verification."""
    w = WebWorkerOrchestrator("dummy.js")
    w.init()
    import time

    msg1 = w.enqueue("task1", {})

    class DummyEvent:
        """Provides semantic functionality and verification."""

        def __init__(self, data):
            """Provides semantic functionality and verification."""
            self.data = data

    past = time.time() - 0.5
    w._on_message(DummyEvent({"id": msg1, "type": "task1_response", "timestamp": past}))
    assert len(w._telemetry) == 1
    assert w._telemetry[0]["id"] == msg1
    assert w._telemetry[0]["latency"] >= 0.5
