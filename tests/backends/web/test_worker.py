"""Module providing core logic and structural definitions."""

import pytest
from onnx9000.backends.web.worker import AtomicLock, WebWorkerEnv
from onnx9000.backends.web.rpc import RPCMessage


class DummyAtomics:
    """Provides semantic functionality and verification."""

    def __init__(self):
        """Provides semantic functionality and verification."""
        self.locked = 0

    def compareExchange(self, arr, idx, expected, value):
        """Provides semantic functionality and verification."""
        old = self.locked
        if self.locked == expected:
            self.locked = value
        return old

    def wait(self, arr, idx, value):
        """Provides semantic functionality and verification."""
        pass

    def store(self, arr, idx, value):
        """Provides semantic functionality and verification."""
        self.locked = value

    def notify(self, arr, idx, count):
        """Provides semantic functionality and verification."""
        pass


class DummySharedArrayBuffer:
    """Provides semantic functionality and verification."""

    @staticmethod
    def new(size):
        """Provides semantic functionality and verification."""
        return [0] * size


class DummyInt32Array:
    """Provides semantic functionality and verification."""

    @staticmethod
    def new(sab):
        """Provides semantic functionality and verification."""
        return sab


class DummyJS:
    """Provides semantic functionality and verification."""

    Atomics = DummyAtomics()
    SharedArrayBuffer = DummySharedArrayBuffer()
    Int32Array = DummyInt32Array()

    def __init__(self):
        """Provides semantic functionality and verification."""
        self.posted = []
        self.listeners = {}

    def postMessage(self, msg):
        """Provides semantic functionality and verification."""
        self.posted.append(msg)

    def addEventListener(self, event, cb):
        """Provides semantic functionality and verification."""
        self.listeners[event] = cb


def dummy_to_js(x):
    """Provides semantic functionality and verification."""
    return x


@pytest.fixture
def mock_js(monkeypatch):
    """Provides semantic functionality and verification."""
    import onnx9000.backends.web.worker as worker

    dummy_js = DummyJS()
    monkeypatch.setattr(worker, "js", dummy_js)
    monkeypatch.setattr(worker, "to_js", dummy_to_js)
    return dummy_js


def test_atomic_lock(mock_js):
    """Provides semantic functionality and verification."""
    lock = AtomicLock()
    assert lock.sab is not None
    assert lock.int32_array is not None
    assert mock_js.Atomics.locked == 0
    lock.lock()
    assert mock_js.Atomics.locked == 1
    lock.unlock()
    assert mock_js.Atomics.locked == 0


def test_atomic_lock_no_js(monkeypatch):
    """Provides semantic functionality and verification."""
    import onnx9000.backends.web.worker as worker

    monkeypatch.setattr(worker, "js", None)
    lock = AtomicLock()
    lock.lock()
    lock.unlock()


def test_worker_env_handlers(mock_js):
    """Provides semantic functionality and verification."""
    env = WebWorkerEnv()

    def handler(payload):
        """Provides semantic functionality and verification."""
        return {"status": "ok", "echo": payload}

    env.register("ping", handler)

    class DummyEvent:
        """Provides semantic functionality and verification."""

        def __init__(self, data):
            """Provides semantic functionality and verification."""
            self.data = data

    env.on_message(DummyEvent({"id": "1", "type": "ping", "payload": "hello"}))
    assert len(mock_js.posted) == 1
    resp = mock_js.posted[0]
    assert resp["id"] == "1"
    assert resp["type"] == "ping_response"
    assert resp["payload"] == {"status": "ok", "echo": "hello"}
    assert resp["error"] is None


def test_worker_env_error_serialization(mock_js):
    """Provides semantic functionality and verification."""
    env = WebWorkerEnv()

    def bad_handler(payload):
        """Provides semantic functionality and verification."""
        raise RuntimeError("Oops!")

    env.register("crash", bad_handler)

    class DummyEvent:
        """Provides semantic functionality and verification."""

        def __init__(self, data):
            """Provides semantic functionality and verification."""
            self.data = data

    env.on_message(DummyEvent({"id": "2", "type": "crash", "payload": None}))
    assert len(mock_js.posted) == 1
    assert mock_js.posted[0]["error"] == "Oops!"
    env.on_message(DummyEvent({"id": "3", "type": "unknown", "payload": None}))
    assert len(mock_js.posted) == 2
    assert "Unknown RPC type" in mock_js.posted[1]["error"]


def test_worker_env_to_py_conversion(mock_js):
    """Provides semantic functionality and verification."""
    env = WebWorkerEnv()
    env.register("ping", lambda p: "pong")

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

    js_data = ToPyData({"id": "4", "type": "ping", "payload": None})
    env.on_message(DummyEvent(js_data))
    assert len(mock_js.posted) == 1
    assert mock_js.posted[0]["payload"] == "pong"


def test_atomic_lock_wait(mock_js):
    """Provides semantic functionality and verification."""
    mock_js.Atomics.locked = 1
    original_ce = mock_js.Atomics.compareExchange
    calls = []

    def custom_ce(arr, idx, expected, value):
        """Provides semantic functionality and verification."""
        calls.append(1)
        if len(calls) == 1:
            return 1
        elif len(calls) == 2:
            return 0
        return original_ce(arr, idx, expected, value)

    mock_js.Atomics.compareExchange = custom_ce
    lock = AtomicLock()
    lock.lock()
    assert len(calls) == 2
