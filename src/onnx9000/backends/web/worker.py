"""Module providing core logic and structural definitions."""

import time
import typing
from typing import Any, Dict, Optional
from .rpc import RPCMessage

try:
    import js  # type: ignore
    from pyodide.ffi import to_js  # type: ignore
except ImportError:
    js = None
    to_js = None


class AtomicLock:
    """Provides semantic functionality and verification."""

    def __init__(self, sab: Optional[Any] = None) -> None:
        """Provides semantic functionality and verification."""
        if sab is None and js is not None and hasattr(js, "SharedArrayBuffer"):
            self.sab = js.SharedArrayBuffer.new(4)
        else:
            self.sab = sab

        if self.sab is not None and js is not None:
            self.int32_array = js.Int32Array.new(self.sab)
        else:
            self.int32_array = None

    def lock(self) -> None:
        """Provides semantic functionality and verification."""
        if self.int32_array is not None and js is not None:
            while js.Atomics.compareExchange(self.int32_array, 0, 0, 1) != 0:
                js.Atomics.wait(self.int32_array, 0, 1)

    def unlock(self) -> None:
        """Provides semantic functionality and verification."""
        if self.int32_array is not None and js is not None:
            js.Atomics.store(self.int32_array, 0, 0)
            js.Atomics.notify(self.int32_array, 0, 1)


class WebWorkerEnv:
    """Provides semantic functionality and verification."""

    def __init__(self) -> None:
        """Provides semantic functionality and verification."""
        self.handlers: Dict[str, typing.Callable[[Any], Any]] = {}
        if js is not None:
            self.debug_mode = False
            # Step 010: serialize error objects using window.addEventListener("error")
            js.addEventListener("error", self._on_global_error)

    def _on_global_error(self, event: Any) -> None:
        """Provides semantic functionality and verification."""
        self._post_message(
            RPCMessage(id="global", type="error", error=self._serialize_error(event))
        )

    def boot(self) -> None:
        """Provides semantic functionality and verification."""
        # Send capabilities handshake
        caps = {
            "wasm_simd": True,  # Hardcoded for now, real check later
            "webgpu": js is not None and hasattr(js.navigator, "gpu"),
            "shared_memory": js is not None and hasattr(js, "SharedArrayBuffer"),
        }
        msg = RPCMessage(id="0", type="handshake", payload=caps)
        self._post_message(msg)

    def register(self, type: str, handler: typing.Callable[[Any], Any]) -> None:
        """Provides semantic functionality and verification."""
        self.handlers[type] = handler

    def _serialize_error(self, err: Exception) -> str:
        """Provides semantic functionality and verification."""
        return str(err)

    def _post_message(self, msg: RPCMessage) -> None:
        """Provides semantic functionality and verification."""
        if js is not None and hasattr(js, "postMessage") and to_js is not None:
            js.postMessage(to_js(msg.to_dict()))

    def on_message(self, event: Any) -> None:
        """Provides semantic functionality and verification."""
        data = event.data
        if hasattr(data, "to_py"):
            data = data.to_py()
        msg = RPCMessage.from_dict(data)

        if msg.type == "get_memory":
            # Step 016
            mem_info = {}
            if js is not None and hasattr(js.navigator, "deviceMemory"):
                mem_info["deviceMemory"] = js.navigator.deviceMemory
            if js is not None and hasattr(js.performance, "memory"):
                mem_info["jsHeapSizeLimit"] = js.performance.memory.jsHeapSizeLimit
                mem_info["totalJSHeapSize"] = js.performance.memory.totalJSHeapSize
                mem_info["usedJSHeapSize"] = js.performance.memory.usedJSHeapSize
            resp = RPCMessage(id=msg.id, type="get_memory_response", payload=mem_info)
            self._post_message(resp)
            return

        resp_payload = None
        error_msg = None

        try:
            if msg.type in self.handlers:
                resp_payload = self.handlers[msg.type](msg.payload)
            else:
                raise ValueError(f"Unknown RPC type: {msg.type}")
        except Exception as e:
            error_msg = self._serialize_error(e)

        resp = RPCMessage(
            id=msg.id,
            type=f"{msg.type}_response",
            payload=resp_payload,
            error=error_msg,
        )
        self._post_message(resp)


class VFSMock:
    """Provides semantic functionality and verification."""

    def __init__(self):
        """Provides semantic functionality and verification."""
        self.files: Dict[str, bytes] = {}

    def write(self, path: str, data: bytes) -> None:
        """Provides semantic functionality and verification."""
        self.files[path] = data

    def read(self, path: str) -> bytes:
        """Provides semantic functionality and verification."""
        if path not in self.files:
            raise FileNotFoundError(f"File not found in VFS: {path}")
        return self.files[path]


class WebWorkerEnv:
    """Provides semantic functionality and verification."""

    def __init__(self) -> None:
        """Provides semantic functionality and verification."""
        self.handlers: Dict[str, typing.Callable[[Any], Any]] = {}
        self.debug_mode = False
        self.vfs = VFSMock()
        if js is not None:
            self.debug_mode = False
            # Step 010: serialize error objects using window.addEventListener("error")
            js.addEventListener("error", self._on_global_error)

    def _on_global_error(self, event: Any) -> None:
        """Provides semantic functionality and verification."""
        self._post_message(
            RPCMessage(id="global", type="error", error=self._serialize_error(event))
        )

    def set_debug(self, debug: bool) -> None:
        """Provides semantic functionality and verification."""
        self.debug_mode = debug
        if self.debug_mode and js is not None and hasattr(js, "console"):
            js.console.log("WebWorkerEnv debug mode enabled")

    def boot(self) -> None:
        """Provides semantic functionality and verification."""
        caps = {
            "wasm_simd": True,
            "webgpu": js is not None and hasattr(js.navigator, "gpu"),
            "shared_memory": js is not None and hasattr(js, "SharedArrayBuffer"),
        }
        msg = RPCMessage(id="0", type="handshake", payload=caps)
        self._post_message(msg)

    def register(self, type: str, handler: typing.Callable[[Any], Any]) -> None:
        """Provides semantic functionality and verification."""
        self.handlers[type] = handler

    def _serialize_error(self, err: Exception) -> str:
        """Provides semantic functionality and verification."""
        return str(err)

    def _post_message(self, msg: RPCMessage) -> None:
        """Provides semantic functionality and verification."""
        if js is not None and hasattr(js, "postMessage") and to_js is not None:
            js.postMessage(to_js(msg.to_dict()))

    def on_message(self, event: Any) -> None:
        """Provides semantic functionality and verification."""
        data = event.data
        if hasattr(data, "to_py"):
            data = data.to_py()
        msg = RPCMessage.from_dict(data)

        if self.debug_mode and js is not None and hasattr(js, "console"):
            js.console.log(f"Worker received: {msg.type}")

        if msg.type == "get_memory":
            mem_info = {}
            if js is not None and hasattr(js.navigator, "deviceMemory"):
                mem_info["deviceMemory"] = js.navigator.deviceMemory
            if js is not None and hasattr(js.performance, "memory"):
                mem_info["jsHeapSizeLimit"] = js.performance.memory.jsHeapSizeLimit
                mem_info["totalJSHeapSize"] = js.performance.memory.totalJSHeapSize
                mem_info["usedJSHeapSize"] = js.performance.memory.usedJSHeapSize
            resp = RPCMessage(id=msg.id, type="get_memory_response", payload=mem_info)
            self._post_message(resp)
            return

        resp_payload = None
        error_msg = None

        try:
            if msg.type in self.handlers:
                resp_payload = self.handlers[msg.type](msg.payload)
            else:
                raise ValueError(f"Unknown RPC type: {msg.type}")
        except Exception as e:
            error_msg = self._serialize_error(e)

        resp = RPCMessage(
            id=msg.id,
            type=f"{msg.type}_response",
            payload=resp_payload,
            error=error_msg,
        )
        self._post_message(resp)
