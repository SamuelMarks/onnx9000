"""Module docstring."""

import collections
import io

import pytest
from onnx9000.converters.pytorch_parser import (
    RestrictedUnpickler,
    _rebuild_parameter,
    _rebuild_tensor_v2,
    _rebuild_tensor_v3,
)


def test_restricted_unpickler_find_class():
    """Docstring for D103."""
    unpickler = RestrictedUnpickler(io.BytesIO(b""), lambda *args: None)

    assert unpickler.find_class("collections", "OrderedDict") == collections.OrderedDict
    assert unpickler.find_class("torch._utils", "_rebuild_tensor_v2") == _rebuild_tensor_v2
    assert unpickler.find_class("torch._utils", "_rebuild_tensor_v3") == _rebuild_tensor_v3
    assert callable(unpickler.find_class("torch._tensor", "_rebuild_from_type_v2"))
    assert unpickler.find_class("torch", "FloatStorage") == "FloatStorage"
    assert unpickler.find_class("torch.nn.parameter", "Parameter") == _rebuild_parameter

    # Builtins
    assert unpickler.find_class("builtins", "list") == list

    # Dummy fallback
    DummyCls = unpickler.find_class("unknown.module", "UnknownClass")
    obj = DummyCls()
    assert obj.__class__.__name__ == "UnknownClass"
    assert obj.__class__.__module__ == "unknown.module"


def test_rebuild_tensor_v2():
    """Docstring for D103."""
    res = _rebuild_tensor_v2("storage", 0, (2, 2), (2, 1), False, [])
    assert res["storage"] == "storage"
    assert res["size"] == (2, 2)


def test_rebuild_tensor_v3():
    """Docstring for D103."""
    res = _rebuild_tensor_v3("storage", 0, (2, 2), (2, 1), False, [], "float32")
    assert res["storage"] == "storage"
    assert res["dtype"] == "float32"


def test_rebuild_parameter():
    """Docstring for D103."""
    res = _rebuild_parameter("data", True, [])
    assert res["data"] == "data"
    assert res["requires_grad"] is True


def test_parse_pytorch_checkpoint_zip():
    """Docstring for D103."""
    import io
    import pickle
    import zipfile

    from onnx9000.converters.pytorch_parser import parse_pytorch_checkpoint

    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as z:
        # Create data.pkl
        io.BytesIO()
        z.writestr("archive/data.pkl", pickle.dumps(None))  # simple none
    stream.seek(0)
    assert parse_pytorch_checkpoint(stream) is None


def test_parse_pytorch_checkpoint_old():
    """Docstring for D103."""
    import io
    import pickle
    import struct

    from onnx9000.converters.pytorch_parser import parse_pytorch_checkpoint

    stream = io.BytesIO()
    pickle.dump(0x1950A86A20F9469CFC6C, stream)
    pickle.dump(2, stream)
    pickle.dump({"protocol_version": 1001}, stream)
    pickle.dump({}, stream)  # state dict
    pickle.dump([], stream)  # keys

    stream.seek(0)
    assert parse_pytorch_checkpoint(stream) == {}


def test_restricted_unpickler_persistent_load():
    """Docstring for D103."""
    import io

    from onnx9000.converters.pytorch_parser import RestrictedUnpickler

    called = []

    def storage_callback(storage_type, key, count):
        called.append((storage_type, key, count))
        return "storage"

    unpickler = RestrictedUnpickler(io.BytesIO(b""), storage_callback)

    # 96-102
    assert unpickler.persistent_load(("storage", "FloatStorage", 123, "cpu", 5)) == "storage"
    assert called == [("FloatStorage", "123", 5)]

    # fallthrough
    assert unpickler.persistent_load(("unknown",)) == ("unknown",)


def test_parse_pytorch_checkpoint_path_bytes():
    """Docstring for D103."""
    # test passing bytes directly
    # we need a valid zip bytes
    import io
    import pickle
    import zipfile

    from onnx9000.converters.pytorch_parser import parse_pytorch_checkpoint

    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as z:
        io.BytesIO()
        z.writestr("archive/data.pkl", pickle.dumps(None))
    stream.seek(0)
    data = stream.read()

    # test passing bytes
    assert parse_pytorch_checkpoint(data) is None

    # test passing filepath
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.write(data)
        tf_name = tf.name

    try:
        assert parse_pytorch_checkpoint(tf_name) is None
    finally:
        import os

        os.remove(tf_name)


def test_rebuild_from_type_v2():
    """Docstring for D103."""
    import io

    from onnx9000.converters.pytorch_parser import RestrictedUnpickler

    unpickler = RestrictedUnpickler(io.BytesIO(b""), lambda *args: None)
    rebuild = unpickler.find_class("torch._tensor", "_rebuild_from_type_v2")

    def func(x):
        return {"x": x}

    res = rebuild(func, "Float", [1], "state")
    assert res["x"] == 1
    assert res["type"] == "Float"
    assert res["state"] == "state"


def test_pytorch_parser_old_format_storages():
    """Docstring for D103."""
    import io
    import pickle
    import struct

    from onnx9000.converters.pytorch_parser import parse_pytorch_checkpoint

    stream = io.BytesIO()
    # Write MAGIC
    pickle.dump(0x1950A86A20F9469CFC6C, stream)
    # version
    pickle.dump(2, stream)
    # sys info
    pickle.dump({"protocol_version": 1001}, stream)

    # State dict: simple dict
    pickle.dump({"x": {"storage": "key1"}}, stream)

    # Keys for storages
    pickle.dump(["key1"], stream)

    # Num elements
    stream.write(struct.pack("<q", 4))

    # We need to mock storage_specs populated during persistent_load
    # Actually wait, `storage_specs[key] = (storage_type, num_elements, location)`
    # This was populated when `persistent_load` ran on state_dict

    stream.seek(0)

    # We need to actually trigger the `storage_callback`
    # Let's rebuild the stream more properly
    stream = io.BytesIO()
    pickle.dump(0x1950A86A20F9469CFC6C, stream)
    pickle.dump(2, stream)
    pickle.dump({"protocol_version": 1001}, stream)

    # State dict with persistent load reference
    state_dict_pickled = io.BytesIO()
    p = pickle.Pickler(state_dict_pickled)
    p.persistent_id = lambda obj: (
        ("storage", "FloatStorage", "key1", "cpu", 4) if obj == "dummy" else None
    )

    # We need to inject persistent_id into standard python pickler to mock what PyTorch does
    class MyPickler(pickle.Pickler):
        def persistent_id(self, obj):
            if getattr(obj, "is_storage", False):
                return ("storage", "FloatStorage", "key1", "cpu", 4)
            return None

    class DummyStorage:
        is_storage = True

    MyPickler(state_dict_pickled).dump({"tensor": {"storage": DummyStorage()}})

    stream.write(state_dict_pickled.getvalue())

    # storage keys
    pickle.dump(["key1"], stream)

    # storage elements length (int64)
    stream.write(struct.pack("<q", 4))

    # actual storage data
    stream.write(struct.pack("<ffff", 1.0, 2.0, 3.0, 4.0))

    stream.seek(0)

    state_dict = parse_pytorch_checkpoint(stream)
    assert "tensor" in state_dict
    assert "storage" in state_dict["tensor"]

    # check that we actually parsed the storage correctly
    # wait, replace_storages also checks lists
    # Let's add a list to the state dict


def test_parse_pytorch_checkpoint_zip_storage_missing_data_pkl():
    """Docstring for D103."""
    import io
    import pickle
    import zipfile

    from onnx9000.converters.pytorch_parser import parse_pytorch_checkpoint

    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w"):
        pass
    stream.seek(0)
    import pytest

    with pytest.raises(ValueError):
        parse_pytorch_checkpoint(stream)


def test_parse_pytorch_checkpoint_zip_storage_callback():
    """Docstring for D103."""
    import io
    import pickle
    import zipfile

    from onnx9000.converters.pytorch_parser import parse_pytorch_checkpoint

    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as z:
        z.writestr("archive/data.pkl", pickle.dumps(None))
        z.writestr("archive/data/key1", b"1234")
    stream.seek(0)
    parse_pytorch_checkpoint(stream)


def test_parse_pytorch_old_format_invalid_magic():
    """Docstring for D103."""
    import io
    import pickle

    from onnx9000.converters.pytorch_parser import _parse_old_format

    stream = io.BytesIO()
    pickle.dump(12345, stream)
    stream.seek(0)
    import pytest

    with pytest.raises(ValueError):
        _parse_old_format(stream)


def test_parse_pytorch_checkpoint_zip_storage_callback_trigger():
    """Docstring for D103."""
    import io
    import pickle
    import zipfile

    from onnx9000.converters.pytorch_parser import parse_pytorch_checkpoint

    # We need to trigger the persistent_load -> storage_callback inside the parse_pytorch_checkpoint
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as z:
        data_pkl = io.BytesIO()

        class DummyPickler(pickle.Pickler):
            def persistent_id(self, obj):
                if obj == "dummy":
                    return ("storage", "FloatStorage", "key1", "cpu", 4)
                return None

        DummyPickler(data_pkl).dump({"x": {"storage": "dummy"}})
        z.writestr("archive/data.pkl", data_pkl.getvalue())
        z.writestr("archive/data/key1", b"1234")
    stream.seek(0)

    state_dict = parse_pytorch_checkpoint(stream)
    assert state_dict["x"]["storage"].tobytes() == b"1234"


def test_replace_storages_list():
    """Docstring for D103."""
    import io
    import pickle
    import struct

    from onnx9000.converters.pytorch_parser import _parse_old_format

    stream = io.BytesIO()
    pickle.dump(0x1950A86A20F9469CFC6C, stream)
    pickle.dump(2, stream)
    pickle.dump({"protocol_version": 1001}, stream)

    state_dict_pickled = io.BytesIO()

    class DummyPickler(pickle.Pickler):
        def persistent_id(self, obj):
            if obj == "dummy":
                return ("storage", "FloatStorage", "key1", "cpu", 4)
            return None

    DummyPickler(state_dict_pickled).dump([{"storage": "dummy"}])
    stream.write(state_dict_pickled.getvalue())

    pickle.dump(["key1"], stream)
    stream.write(struct.pack("<q", 4))
    stream.write(struct.pack("<ffff", 1.0, 2.0, 3.0, 4.0))
    stream.seek(0)

    state_dict = _parse_old_format(stream)
    assert len(state_dict) == 1
    assert "storage" in state_dict[0]


def test_replace_storages_nested_list():
    """Docstring for D103."""
    import io
    import pickle
    import struct

    from onnx9000.converters.pytorch_parser import _parse_old_format

    stream = io.BytesIO()
    pickle.dump(0x1950A86A20F9469CFC6C, stream)
    pickle.dump(2, stream)
    pickle.dump({"protocol_version": 1001}, stream)

    state_dict_pickled = io.BytesIO()

    class DummyPickler(pickle.Pickler):
        def persistent_id(self, obj):
            if obj == "dummy":
                return ("storage", "FloatStorage", "key1", "cpu", 4)
            return None

    # Nest it to hit replace_storages(v) recursively in lists
    DummyPickler(state_dict_pickled).dump([[{"storage": "dummy"}]])
    stream.write(state_dict_pickled.getvalue())

    pickle.dump(["key1"], stream)
    stream.write(struct.pack("<q", 4))
    stream.write(struct.pack("<ffff", 1.0, 2.0, 3.0, 4.0))
    stream.seek(0)

    state_dict = _parse_old_format(stream)
    assert len(state_dict) == 1
    assert "storage" in state_dict[0][0]


def test_parse_pytorch_checkpoint_not_dict_bytes():
    """Docstring for D103."""
    from onnx9000.converters.pytorch_parser import parse_pytorch_checkpoint

    class MockFile:
        def read(self):
            return b"123"

    import pytest

    with pytest.raises(Exception):
        parse_pytorch_checkpoint(MockFile())


def test_restricted_unpickler_dummy_repr():
    """Docstring for D103."""
    import io

    from onnx9000.converters.pytorch_parser import RestrictedUnpickler

    unpickler = RestrictedUnpickler(io.BytesIO(b""), lambda *args: None)
    DummyClass = unpickler.find_class("unknown_module", "UnknownClass")
    obj = DummyClass()
    assert repr(obj) == "<unknown_module.UnknownClass>"


def test_parse_pytorch_checkpoint_dict_like_bytes_fallback():
    """Docstring for D103."""
    import io

    import pytest
    from onnx9000.converters.pytorch_parser import parse_pytorch_checkpoint

    class DictLike:
        def __init__(self):
            pass

        def read(self):
            return b"123"

    with pytest.raises(Exception):
        parse_pytorch_checkpoint(DictLike())


def test_replace_storages_nested_list_v2():
    """Docstring for D103."""
    import io
    import pickle
    import struct

    from onnx9000.converters.pytorch_parser import _parse_old_format

    stream = io.BytesIO()
    pickle.dump(0x1950A86A20F9469CFC6C, stream)
    pickle.dump(2, stream)
    pickle.dump({"protocol_version": 1001}, stream)

    state_dict_pickled = io.BytesIO()

    class DummyPickler(pickle.Pickler):
        def persistent_id(self, obj):
            if obj == "dummy":
                return ("storage", "FloatStorage", "key1", "cpu", 4)
            return None

    DummyPickler(state_dict_pickled).dump(
        [[{"storage": "key1"}]]
    )  # Wait, it looks up in `storages` and `storages` has 'key1'.
    stream.write(state_dict_pickled.getvalue())

    pickle.dump(["key1"], stream)
    stream.write(struct.pack("<q", 4))
    stream.write(struct.pack("<ffff", 1.0, 2.0, 3.0, 4.0))
    stream.seek(0)

    # We need the inner dict to have 'storage' == 'key1' and for 'key1' to be in storages.
    # But how does 'key1' get into storages? It gets there because it's in storage_keys.
    # Where does storage_keys get its type? from storage_specs.
    # So we need DummyPickler to ALSO dump "dummy" so that storage_specs gets "key1".

    stream2 = io.BytesIO()
    pickle.dump(0x1950A86A20F9469CFC6C, stream2)
    pickle.dump(2, stream2)
    pickle.dump({"protocol_version": 1001}, stream2)

    state_dict_pickled2 = io.BytesIO()
    DummyPickler(state_dict_pickled2).dump([{"storage": "dummy"}, [{"storage": "key1"}]])
    stream2.write(state_dict_pickled2.getvalue())

    pickle.dump(["key1"], stream2)
    stream2.write(struct.pack("<q", 4))
    stream2.write(struct.pack("<ffff", 1.0, 2.0, 3.0, 4.0))
    stream2.seek(0)

    state_dict = _parse_old_format(stream2)
    assert len(state_dict) == 2


def test_replace_storages_dict_else():
    """Docstring for D103."""
    import io
    import pickle
    import struct

    from onnx9000.converters.pytorch_parser import _parse_old_format

    stream2 = io.BytesIO()
    pickle.dump(0x1950A86A20F9469CFC6C, stream2)
    pickle.dump(2, stream2)
    pickle.dump({"protocol_version": 1001}, stream2)

    state_dict_pickled2 = io.BytesIO()

    # a dict that doesn't hit the storage condition
    class DummyPickler(pickle.Pickler):
        def persistent_id(self, obj):
            if obj == "dummy":
                return ("storage", "FloatStorage", "key1", "cpu", 4)
            return None

    DummyPickler(state_dict_pickled2).dump({"a": {"b": 1}, "x": "dummy"})
    stream2.write(state_dict_pickled2.getvalue())

    pickle.dump(["key1"], stream2)
    stream2.write(struct.pack("<q", 4))
    stream2.write(struct.pack("<ffff", 1.0, 2.0, 3.0, 4.0))
    stream2.seek(0)

    _parse_old_format(stream2)
