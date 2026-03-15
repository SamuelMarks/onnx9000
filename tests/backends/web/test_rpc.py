import json
import sys
from unittest.mock import patch, MagicMock
from onnx9000.backends.web.rpc import (
    serialize_fallback,
    deserialize_fallback,
    RPCMessage,
    CancellationToken,
)


def test_fallback_without_msgpack():
    with patch("builtins.__import__") as mock_import:

        def import_mock(name, *args, **kwargs):
            if name == "msgpack":
                raise ImportError("No msgpack")
            import importlib

            return importlib.__import__(name, *args, **kwargs)

        mock_import.side_effect = import_mock
        data = {"key": "value"}
        serialized = serialize_fallback(data)
        assert serialized == json.dumps(data)
        deserialized = deserialize_fallback(serialized.encode("utf-8"))
        assert deserialized == data


def test_deserialize_fallback_str():
    data = {"key": "value"}
    serialized = json.dumps(data)
    deserialized = deserialize_fallback(serialized)
    assert deserialized == data


def test_fallback_with_msgpack():
    mock_msgpack = MagicMock()
    mock_msgpack.packb.return_value = b"packed"
    mock_msgpack.unpackb.return_value = {"key": "value"}
    with patch.dict("sys.modules", {"msgpack": mock_msgpack}):
        assert serialize_fallback({"key": "value"}) == b"packed"
        assert deserialize_fallback(b"packed") == {"key": "value"}


def test_deserialize_fallback_other():
    data = {"key": "value"}
    assert deserialize_fallback(data) == data


def test_rpc_message():
    msg = RPCMessage("1", "test", "data")
    assert msg.to_dict()["id"] == "1"
    msg2 = RPCMessage.from_dict({"id": "2", "type": "t", "payload": "p"})
    assert msg2.id == "2"


def test_cancellationToken():
    c = CancellationToken()
    called = []
    c.on_cancel(lambda: called.append(1))
    c.cancel()
    assert c.is_cancelled
    assert called == [1]
    c.on_cancel(lambda: called.append(2))
    assert called == [1, 2]


def test_pyodide_import():
    import importlib
    import onnx9000.backends.web.rpc
    import sys

    mock_js = MagicMock()
    mock_pyodide = MagicMock()
    mock_pyodide.ffi.to_js = MagicMock()
    with patch.dict(
        "sys.modules",
        {"js": mock_js, "pyodide": mock_pyodide, "pyodide.ffi": mock_pyodide.ffi},
    ):
        importlib.reload(onnx9000.backends.web.rpc)
        assert onnx9000.backends.web.rpc.js is not None
