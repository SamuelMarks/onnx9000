import ctypes

from onnx9000.tensorrt.enums import DataType
from onnx9000.tensorrt.ffi import ffi
from onnx9000.tensorrt.structs import Dims


class ITensor:
    def __init__(self, ptr: int, name: str):
        self.ptr = ptr
        self.name = name

    def __repr__(self):
        return f"<ITensor name='{self.name}'>"


class INetworkDefinition:
    def __init__(self, ptr: int):
        self.ptr = ptr
        self.tensors = {}
        ffi.register_pointer(self.ptr, self)

    def add_input(self, name: str, dtype: DataType, dims: Dims) -> ITensor:
        add_input_func = getattr(ffi.lib, "addInput", None)
        if not add_input_func:
            raise RuntimeError("addInput not found")
        add_input_func.restype = ctypes.c_void_p
        add_input_func.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int32,
            ctypes.POINTER(Dims),
        ]

        name_bytes = name.encode("utf-8")
        ptr = add_input_func(
            ctypes.c_void_p(self.ptr), name_bytes, ctypes.c_int32(dtype.value), ctypes.pointer(dims)
        )
        if not ptr:
            raise RuntimeError(f"Failed to add input {name}")

        tensor = ITensor(ptr, name)
        self.tensors[name] = tensor
        return tensor

    def mark_output(self, tensor: ITensor):
        mark_output_func = getattr(ffi.lib, "markOutput", None)
        if not mark_output_func:
            raise RuntimeError("markOutput not found")
        mark_output_func.restype = None
        mark_output_func.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        mark_output_func(ctypes.c_void_p(self.ptr), ctypes.c_void_p(tensor.ptr))

    def destroy(self):
        if not self.ptr:
            return
        destroy_net = getattr(ffi.lib, "destroyNetworkDefinition", None)
        if destroy_net:
            destroy_net(ctypes.c_void_p(self.ptr))
        ffi.unregister_pointer(self.ptr)
        self.ptr = None

    def __del__(self):
        self.destroy()


class IBuilderConfig:
    def __init__(self, ptr: int):
        self.ptr = ptr
        ffi.register_pointer(self.ptr, self)

    def destroy(self):
        if not self.ptr:
            return
        destroy_cfg = getattr(ffi.lib, "destroyBuilderConfig", None)
        if destroy_cfg:
            destroy_cfg(ctypes.c_void_p(self.ptr))
        ffi.unregister_pointer(self.ptr)
        self.ptr = None

    def __del__(self):
        self.destroy()
