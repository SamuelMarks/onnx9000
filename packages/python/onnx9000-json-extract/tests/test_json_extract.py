import json

from onnx9000.json_extract import _default_serializer, extract_json


def test_json_extract():
    class DummyObj:
        def __init__(self):
            self.a = 1
            self._b = 2

    assert _default_serializer(b"abc") == "[Buffer: 3 bytes]"
    assert _default_serializer(DummyObj()) == {"a": 1}
    assert _default_serializer({1}) == [1]
    assert _default_serializer(1.23) == "1.23"

    res = extract_json(DummyObj(), indent=0)
    assert '"a":1' in res.replace(" ", "").replace("\n", "")
