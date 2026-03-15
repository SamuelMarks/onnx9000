"""Module providing core logic and structural definitions."""

import builtins
import importlib
import sys
import types
import pytest
from onnx9000.backends.web import worker
from onnx9000.backends.web.rpc import RPCMessage


class DummyJSNavigator:
    """Provides semantic functionality and verification."""

    gpu = True
    deviceMemory = 8


class DummyJSPerformanceMemory:
    """Provides semantic functionality and verification."""

    jsHeapSizeLimit = 1000
    totalJSHeapSize = 500
    usedJSHeapSize = 250


class DummyJSPerformance:
    """Provides semantic functionality and verification."""

    memory = DummyJSPerformanceMemory()


class DummyJS:
    """Provides semantic functionality and verification."""

    navigator = DummyJSNavigator()
    performance = DummyJSPerformance()
    SharedArrayBuffer = True

    def __init__(self):
        """Provides semantic functionality and verification."""
        self.events = {}
        self.posted = []
        self.console_logs = []

    def addEventListener(self, event, handler):
        """Provides semantic functionality and verification."""
        self.events[event] = handler

    def postMessage(self, msg):
        """Provides semantic functionality and verification."""
        self.posted.append(msg)


class DummyConsole:
    """Provides semantic functionality and verification."""

    def __init__(self, parent):
        """Provides semantic functionality and verification."""
        self.parent = parent

    def log(self, msg):
        """Provides semantic functionality and verification."""
        self.parent.console_logs.append(msg)


def dummy_to_js(x):
    """Provides semantic functionality and verification."""
    return x


@pytest.fixture
def capture_worker_envs():
    """Provides semantic functionality and verification."""
    dummy_js = DummyJS()
    dummy_js.console = DummyConsole(dummy_js)
    dummy_js_module = types.ModuleType("js")
    dummy_js_module.navigator = dummy_js.navigator
    dummy_js_module.performance = dummy_js.performance
    dummy_js_module.SharedArrayBuffer = dummy_js.SharedArrayBuffer
    dummy_js_module.addEventListener = dummy_js.addEventListener
    dummy_js_module.postMessage = dummy_js.postMessage
    dummy_js_module.console = dummy_js.console
    dummy_pyodide_ffi = types.ModuleType("pyodide.ffi")
    dummy_pyodide_ffi.to_js = dummy_to_js
    dummy_pyodide = types.ModuleType("pyodide")
    dummy_pyodide.ffi = dummy_pyodide_ffi
    sys.modules["js"] = dummy_js_module
    sys.modules["pyodide"] = dummy_pyodide
    sys.modules["pyodide.ffi"] = dummy_pyodide_ffi
    classes = []
    orig_build_class = builtins.__build_class__

    def my_build_class(*args, **kwargs):
        """Provides semantic functionality and verification."""
        cls = orig_build_class(*args, **kwargs)
        if cls.__name__ == "WebWorkerEnv":
            classes.append(cls)
        return cls

    builtins.__build_class__ = my_build_class
    try:
        importlib.reload(worker)
    finally:
        builtins.__build_class__ = orig_build_class
        del sys.modules["js"]
        del sys.modules["pyodide"]
        del sys.modules["pyodide.ffi"]
    return classes[0], classes[1], dummy_js


def test_first_web_worker_env(capture_worker_envs):
    """Provides semantic functionality and verification."""
    FirstEnv, _, dummy_js = capture_worker_envs
    env = FirstEnv()
    env.boot()
    with pytest.raises(TypeError):
        env._on_global_error(Exception("test error"))

    def my_handler(payload):
        """Provides semantic functionality and verification."""
        return {"result": payload * 2}

    env.register("multiply", my_handler)

    class DummyEvent:
        """Provides semantic functionality and verification."""

        def __init__(self, data):
            """Provides semantic functionality and verification."""
            self.data = data

    env.on_message(DummyEvent({"id": "msg1", "type": "multiply", "payload": 5}))
    env.on_message(DummyEvent({"id": "msg2", "type": "get_memory"}))
    env.on_message(DummyEvent({"id": "msg3", "type": "unknown_type"}))

    class ToPyData:
        """Provides semantic functionality and verification."""

        def to_py(self):
            """Provides semantic functionality and verification."""
            return {"id": "msg4", "type": "multiply", "payload": 3}

    env.on_message(DummyEvent(ToPyData()))


def test_second_web_worker_env_global_error(capture_worker_envs):
    """Provides semantic functionality and verification."""
    _, SecondEnv, dummy_js = capture_worker_envs
    env = SecondEnv()
    with pytest.raises(TypeError):
        env._on_global_error(Exception("another error"))


def test_vfs_mock():
    """Provides semantic functionality and verification."""
    from onnx9000.backends.web.worker import VFSMock

    vfs = VFSMock()
    vfs.write("test.txt", b"hello")
    assert vfs.read("test.txt") == b"hello"
    with pytest.raises(FileNotFoundError):
        vfs.read("missing.txt")


def test_second_web_worker_env_remaining(capture_worker_envs):
    """Provides semantic functionality and verification."""
    _, SecondEnv, dummy_js = capture_worker_envs
    env = SecondEnv()
    env.set_debug(True)
    assert env.debug_mode is True
    env.boot()
    msg = dummy_js.posted[-1]
    assert msg["type"] == "handshake"

    class DummyEvent:
        """Provides semantic functionality and verification."""

        def __init__(self, data):
            """Provides semantic functionality and verification."""
            self.data = data

    env.on_message(DummyEvent({"id": "mem", "type": "get_memory"}))
    resp = dummy_js.posted[-1]
    assert resp["type"] == "get_memory_response"
    assert resp["payload"]["deviceMemory"] == 8
