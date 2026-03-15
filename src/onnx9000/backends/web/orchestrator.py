"""Module providing core logic and structural definitions."""

import uuid
import time
import typing
import asyncio
from typing import Any, Callable, Dict, Optional, List, Tuple
from .rpc import RPCMessage, CancellationToken

try:
    from pyodide.ffi import create_proxy, to_js  # type: ignore
    import js  # type: ignore
except ImportError:
    js = None
    create_proxy = None
    to_js = None


class WebWorkerOrchestrator:
    """Provides semantic functionality and verification."""

    def __init__(self, worker_url: str) -> None:
        """Provides semantic functionality and verification."""
        self.worker_url = worker_url
        self.worker: Optional[Any] = None
        self._pending_requests: Dict[str, Callable[[Any], None]] = {}
        self._stream_callbacks: Dict[str, Callable[[Any], None]] = {}
        self._task_queue: List[
            Tuple[int, str, str, Any, Optional[CancellationToken]]
        ] = []
        self._active_task_id: Optional[str] = None
        self._oom_restarts = 0
        self._last_heartbeat = 0.0
        self._heartbeat_interval = 1.0
        self._timeout_threshold = 5.0
        self._telemetry: List[Dict[str, Any]] = []
        self.capabilities: Dict[str, bool] = {}
        self._init_future: Optional[asyncio.Future] = None

    async def init_async(self) -> None:
        """Provides semantic functionality and verification."""
        loop = asyncio.get_running_loop()
        self._init_future = loop.create_future()
        if js is not None:
            self.worker = js.Worker.new(self.worker_url)
            self._setup_worker_listeners()
            self._last_heartbeat = time.time()
        else:
            self._init_future.set_result(None)
        await self._init_future

    def init(self) -> None:
        """Provides semantic functionality and verification."""
        if js is not None:
            self.worker = js.Worker.new(self.worker_url)
            self._setup_worker_listeners()
            self._last_heartbeat = time.time()

    def _setup_worker_listeners(self) -> None:
        """Provides semantic functionality and verification."""
        if self.worker is not None and create_proxy is not None:
            proxy_msg = create_proxy(self._on_message)
            proxy_err = create_proxy(self._on_error)
            self.worker.addEventListener("message", proxy_msg)
            self.worker.addEventListener("error", proxy_err)

    def terminate(self) -> None:
        """Provides semantic functionality and verification."""
        if self.worker is not None:
            self.worker.terminate()
            self.worker = None
            self._active_task_id = None

    async def soft_kill(self) -> None:
        """Provides semantic functionality and verification."""
        while self._active_task_id is not None or self._task_queue:
            await asyncio.sleep(0.01)
        self.terminate()

    def restart(self) -> None:
        """Provides semantic functionality and verification."""
        self.terminate()
        self.init()

    def _on_error(self, event: Any) -> None:
        """Provides semantic functionality and verification."""
        err_msg = str(getattr(event, "message", "Unknown error")).lower()
        if "out of memory" in err_msg or "oom" in err_msg:
            self._oom_restarts += 1
            self.restart()

    def _on_message(self, event: Any) -> None:
        """Provides semantic functionality and verification."""
        data = event.data
        if hasattr(data, "to_py"):
            data = data.to_py()
        msg = RPCMessage.from_dict(data)

        latency = time.time() - msg.timestamp
        self._telemetry.append({"id": msg.id, "type": msg.type, "latency": latency})

        if msg.type == "heartbeat":
            self._last_heartbeat = time.time()
            return

        if msg.type == "handshake":
            self.capabilities = msg.payload
            if self._init_future is not None and not self._init_future.done():
                self._init_future.set_result(None)
            return

        if msg.id in self._stream_callbacks and msg.type.endswith("_stream"):
            self._stream_callbacks[msg.id](msg.payload)
            return

        if msg.id in self._pending_requests:
            self._pending_requests[msg.id](msg)
            del self._pending_requests[msg.id]

        if msg.id == self._active_task_id:
            self._active_task_id = None

        self._process_queue()

    def check_health(self) -> bool:
        """Provides semantic functionality and verification."""
        if self.worker is None:
            return False
        if time.time() - self._last_heartbeat > self._timeout_threshold:
            self.restart()
            return False
        return True

    def _process_queue(self) -> None:
        """Provides semantic functionality and verification."""
        if self._active_task_id is not None or not self._task_queue:
            return
        self._task_queue.sort(key=lambda x: x[0])
        _, msg_id, type_, payload, token = self._task_queue.pop(0)

        if token is not None and token.is_cancelled:
            self._process_queue()
            return

        self._active_task_id = msg_id
        self._dispatch(msg_id, type_, payload)

    def _dispatch(self, msg_id: str, type_: str, payload: Any) -> None:
        """Provides semantic functionality and verification."""
        msg = RPCMessage(id=msg_id, type=type_, payload=payload)
        if self.worker is not None and js is not None and to_js is not None:
            self.worker.postMessage(to_js(msg.to_dict()))

    def enqueue(
        self,
        type: str,
        payload: Any,
        priority: int = 1,
        token: Optional[CancellationToken] = None,
    ) -> str:
        """Provides semantic functionality and verification."""
        msg_id = str(uuid.uuid4())

        if token is not None:

            def on_cancel():
                """Provides semantic functionality and verification."""
                if self._active_task_id == msg_id:
                    self._dispatch(msg_id, "cancel", None)

            token.on_cancel(on_cancel)

        self._task_queue.append((priority, msg_id, type, payload, token))
        self._process_queue()
        return msg_id

    def send(self, type: str, payload: Any) -> str:
        """Provides semantic functionality and verification."""
        return self.enqueue(type, payload, priority=0)

    async def send_request(self, type: str, payload: Any, priority: int = 1) -> Any:
        """Provides semantic functionality and verification."""
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        def callback(msg: RPCMessage) -> None:
            """Provides semantic functionality and verification."""
            if msg.error:
                future.set_exception(RuntimeError(msg.error))
            else:
                future.set_result(msg.payload)

        msg_id = self.enqueue(type, payload, priority=priority)
        self._pending_requests[msg_id] = callback
        return await future

    async def send_stream(
        self, type: str, payload: Any, on_chunk: Callable[[Any], None]
    ) -> Any:
        """Provides semantic functionality and verification."""
        msg_id = self.enqueue(type, payload)
        self._stream_callbacks[msg_id] = on_chunk

        loop = asyncio.get_running_loop()
        future = loop.create_future()

        def callback(msg: RPCMessage) -> None:
            """Provides semantic functionality and verification."""
            if msg.id in self._stream_callbacks:
                del self._stream_callbacks[msg.id]
            if msg.error:
                future.set_exception(RuntimeError(msg.error))
            else:
                future.set_result(msg.payload)

        self._pending_requests[msg_id] = callback
        return await future


class WorkerPool:
    """Provides semantic functionality and verification."""

    def __init__(self, count: int, url: str):
        """Provides semantic functionality and verification."""
        self.workers = [WebWorkerOrchestrator(url) for _ in range(count)]
        self._rr_index = 0

    def init(self) -> None:
        """Provides semantic functionality and verification."""
        for w in self.workers:
            w.init()

    def enqueue(self, type: str, payload: Any) -> str:
        """Provides semantic functionality and verification."""
        w = self.workers[self._rr_index]
        self._rr_index = (self._rr_index + 1) % len(self.workers)
        return w.enqueue(type, payload)
