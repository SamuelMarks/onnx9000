"""Module providing core logic and structural definitions."""

import json
import time
import typing
from typing import Any, Dict, Optional, Union

try:
    from pyodide.ffi import to_js  # type: ignore
    import js  # type: ignore
except ImportError:
    js = None
    to_js = None


class RPCMessage:
    """Provides semantic functionality and verification."""

    def __init__(
        self,
        id: str,
        type: str,
        payload: Any,
        error: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """Provides semantic functionality and verification."""
        self.id = id
        self.type = type
        self.payload = payload
        self.error = error
        self.timestamp = timestamp if timestamp is not None else time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Provides semantic functionality and verification."""
        return {
            "id": self.id,
            "type": self.type,
            "payload": self.payload,
            "error": self.error,
            "timestamp": self.timestamp,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "RPCMessage":
        """Provides semantic functionality and verification."""
        return RPCMessage(
            id=data.get("id", ""),
            type=data.get("type", ""),
            payload=data.get("payload"),
            error=data.get("error"),
            timestamp=data.get("timestamp"),
        )


class CancellationToken:
    """Provides semantic functionality and verification."""

    def __init__(self) -> None:
        """Provides semantic functionality and verification."""
        self.is_cancelled = False
        self._callbacks = []

    def cancel(self) -> None:
        """Provides semantic functionality and verification."""
        self.is_cancelled = True
        for cb in self._callbacks:
            cb()

    def on_cancel(self, callback: typing.Callable[[], None]) -> None:
        """Provides semantic functionality and verification."""
        if self.is_cancelled:
            callback()
        else:
            self._callbacks.append(callback)


def serialize_fallback(data: Any) -> Any:
    """Step 036: Fallback serialization if structured clone fails"""
    try:
        import msgpack  # type: ignore

        return msgpack.packb(data)
    except ImportError:
        return json.dumps(data)


def deserialize_fallback(data: Any) -> Any:
    """Provides semantic functionality and verification."""
    if isinstance(data, (bytes, bytearray)):
        try:
            import msgpack  # type: ignore

            return msgpack.unpackb(data)
        except ImportError:
            return json.loads(data.decode("utf-8"))
    if isinstance(data, str):
        return json.loads(data)
    return data
