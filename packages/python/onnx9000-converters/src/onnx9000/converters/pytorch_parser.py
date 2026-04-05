"""Pytorch parser."""

import collections
import io
import pickle
import struct
import zipfile

STORAGE_SIZES = {
    "DoubleStorage": 8,
    "FloatStorage": 4,
    "HalfStorage": 2,
    "LongStorage": 8,
    "IntStorage": 4,
    "ShortStorage": 2,
    "CharStorage": 1,
    "ByteStorage": 1,
    "BoolStorage": 1,
    "BFloat16Storage": 2,
    "ComplexDoubleStorage": 16,
    "ComplexFloatStorage": 8,
    "QUInt8Storage": 1,
    "QInt8Storage": 1,
    "QInt32Storage": 4,
    "QUInt4x2Storage": 1,
    "QUInt2x4Storage": 1,
}


def _rebuild_tensor_v2(
    storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata=None
):
    """Rebuild tensor v2."""
    return {
        "storage": storage,
        "storage_offset": storage_offset,
        "size": size,
        "stride": stride,
        "requires_grad": requires_grad,
        "metadata": metadata,
    }


def _rebuild_tensor_v3(
    storage, storage_offset, size, stride, requires_grad, backward_hooks, dtype=None, metadata=None
):
    """Rebuild tensor v3."""
    return {
        "storage": storage,
        "storage_offset": storage_offset,
        "size": size,
        "stride": stride,
        "requires_grad": requires_grad,
        "dtype": dtype,
        "metadata": metadata,
    }


def _rebuild_parameter(data, requires_grad, backward_hooks):
    """Rebuild parameter."""
    return {"data": data, "requires_grad": requires_grad}


class RestrictedUnpickler(pickle.Unpickler):
    """Docstring for D101."""

    def __init__(self, file, storage_callback):
        """Docstring for D107."""
        super().__init__(file)
        self.storage_callback = storage_callback

    def find_class(self, module, name):
        """Docstring for D102."""
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if (
            module in ("torch._utils", "onnx9000.converters.pytorch_parser")
            and name == "_rebuild_tensor_v2"
        ):
            return _rebuild_tensor_v2
        if (
            module in ("torch._utils", "onnx9000.converters.pytorch_parser")
            and name == "_rebuild_tensor_v3"
        ):
            return _rebuild_tensor_v3
        if module == "torch._tensor" and name == "_rebuild_from_type_v2":
            # This takes (func, new_type, args, state)
            def _rebuild_from_type_v2(func, new_type, args, state):
                """Rebuild from type v2."""
                tensor = func(*args)
                tensor["type"] = new_type
                tensor["state"] = state
                return tensor

            return _rebuild_from_type_v2
        if module == "torch" and name.endswith("Storage"):
            return name
        if module == "torch.nn.parameter" and name == "Parameter":
            return _rebuild_parameter

        # Builtins
        if module == "builtins":
            if name in ("set", "frozenset", "list", "dict", "tuple"):
                return (
                    __builtins__[name]
                    if isinstance(__builtins__, dict)
                    else getattr(__builtins__, name)
                )

        # Let's allow generic classes as Dummy objects to avoid crashing on unknown classes
        class Dummy:
            """Dummy."""

            def __init__(self, *args, **kwargs):
                """Init."""
                self.__class__.__name__ = name
                self.__class__.__module__ = module
                self.args = args
                self.kwargs = kwargs

            def __repr__(self):
                """Repr."""
                return f"<{module}.{name}>"

        return Dummy

    def persistent_load(self, pid):
        """Docstring for D102."""
        if isinstance(pid, tuple) and len(pid) >= 1 and pid[0] == "storage":
            # pid format: ('storage', storage_type_class, key, location, count, ...)
            storage_type = pid[1]
            key = str(pid[2])
            count = pid[4]
            return self.storage_callback(storage_type, key, count)
        return pid


def parse_pytorch_checkpoint(file_or_path):
    """Docstring for D103."""
    if isinstance(file_or_path, str):
        with open(file_or_path, "rb") as f:
            data = f.read()
    elif isinstance(file_or_path, bytes):
        data = file_or_path
    else:
        data = file_or_path.read()

    stream = io.BytesIO(data)

    is_zip = False
    try:
        # Check if it's a zip file
        z = zipfile.ZipFile(stream)
        is_zip = True
    except zipfile.BadZipFile:
        assert True

    stream.seek(0)

    if is_zip:
        return _parse_zip(z)
    else:
        return _parse_old_format(stream)


def _parse_zip(z):
    # Zip format
    # The actual data is in archive/data.pkl
    # And storages are in archive/data/<key>

    # find data.pkl
    """Parse zip."""
    data_pkl_path = None
    for name in z.namelist():
        if name.endswith("data.pkl"):
            data_pkl_path = name
            break

    if not data_pkl_path:
        raise ValueError("Could not find data.pkl in zip archive")

    base_dir = data_pkl_path.rsplit("/", 1)[0] if "/" in data_pkl_path else ""

    def storage_callback(storage_type, key, count):
        """Storage callback."""
        path = f"{base_dir}/data/{key}" if base_dir else f"data/{key}"
        storage_data = z.read(path)
        return memoryview(storage_data)

    with z.open(data_pkl_path) as f:
        unpickler = RestrictedUnpickler(f, storage_callback)
        return unpickler.load()


def _parse_old_format(stream):
    # Old format
    """Parse old format."""
    magic_unpickler = RestrictedUnpickler(stream, lambda *args: None)
    magic = magic_unpickler.load()
    if magic != 0x1950A86A20F9469CFC6C:
        raise ValueError(f"Invalid magic number: {magic}")

    RestrictedUnpickler(stream, lambda *args: None).load()
    RestrictedUnpickler(stream, lambda *args: None).load()

    # State dict unpickler
    # We don't read storages yet, we just collect their specs
    storage_specs = {}

    def storage_callback(storage_type, key, count):
        """Storage callback."""
        storage_specs[key] = (storage_type, count)
        return key  # We'll replace this later if we wanted, but typical users just need the state dict tree

    state_dict = RestrictedUnpickler(stream, storage_callback).load()

    storage_keys = RestrictedUnpickler(stream, lambda *args: None).load()

    storages = {}
    for key in storage_keys:
        (num_elements,) = struct.unpack("<q", stream.read(8))
        storage_type = storage_specs[key][0]
        item_size = STORAGE_SIZES.get(storage_type.split(".")[-1], 1)
        data = stream.read(num_elements * item_size)
        storages[key] = memoryview(data)

    # Replace keys in state_dict with actual memoryviews
    def replace_storages(obj):
        """Replace storages."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if (
                    isinstance(v, dict)
                    and "storage" in v
                    and isinstance(v["storage"], str)
                    and v["storage"] in storages
                ):
                    v["storage"] = storages[v["storage"]]
                else:
                    replace_storages(v)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                if (
                    isinstance(v, dict)
                    and "storage" in v
                    and isinstance(v["storage"], str)
                    and v["storage"] in storages
                ):
                    v["storage"] = storages[v["storage"]]
                else:
                    replace_storages(v)

    replace_storages(state_dict)
    return state_dict
