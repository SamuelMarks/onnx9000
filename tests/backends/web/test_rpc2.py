import sys
import json
from unittest.mock import patch
from onnx9000.backends.web.rpc import serialize_fallback, deserialize_fallback


def test_fallback_without_msgpack():
    with patch.dict("sys.modules", {"msgpack": None}):
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
