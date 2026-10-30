"""Network."""

import ctypes

from onnx9000.core.exceptions import BackendExecutionError
from onnx9000.tensorrt.enums import DataType
from onnx9000.tensorrt.ffi import ffi
from onnx9000.tensorrt.structs import Dims


class ITensor:
    """ITensor class."""

    def __init__(self, ptr: int, name: str) -> None:
        """Initialize ITensor with configured parameters and state.

        Args:
            ptr: Parameter specifying ptr (int).
            name: Parameter specifying name (str).
        """
        self.ptr = ptr
        self.name = name

    def __repr__(self) -> str:
        """Return a formatted string representation of ITensor.

        Returns:
            Result of repr.
        """
        return f"<ITensor name='{self.name}'>"


class INetworkDefinition:
    """INetworkDefinition class."""

    def __init__(self, ptr: int) -> None:
        """Initialize INetworkDefinition with configured parameters and state.

        Args:
            ptr: Parameter specifying ptr (int).
        """
        self.ptr = ptr
        self.tensors = {}
        ffi.register_pointer(self.ptr, self)

    def add_input(self, name: str, dtype: DataType, dims: Dims) -> ITensor:
        """Execute add_input."""
        add_input_func = getattr(ffi.lib, "addInput", None)
        if not add_input_func:
            raise BackendExecutionError("addInput not found")
        add_input_func.restype = ctypes.c_void_p
        add_input_func.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int32,
            ctypes.POINTER(Dims),
        ]

        name_bytes = name.encode("utf-8")
        ptr = add_input_func(
            ctypes.c_void_p(self.ptr),
            name_bytes,
            ctypes.c_int32(dtype.value),
            ctypes.pointer(dims),
        )
        if not ptr:
            raise BackendExecutionError(f"Failed to add input {name}")

        tensor = ITensor(ptr, name)
        self.tensors[name] = tensor
        return tensor

    def mark_output(self, tensor: ITensor) -> None:
        """Execute mark_output."""
        mark_output_func = getattr(ffi.lib, "markOutput", None)
        if not mark_output_func:
            raise BackendExecutionError("markOutput not found")
        mark_output_func.restype = None
        mark_output_func.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        mark_output_func(ctypes.c_void_p(self.ptr), ctypes.c_void_p(tensor.ptr))

    def destroy(self) -> None:
        """Execute destroy."""
        if not self.ptr or getattr(ffi, "lib", None) is None:
            return
        destroy_net = getattr(ffi.lib, "destroyNetworkDefinition", None)
        if destroy_net:
            destroy_net(ctypes.c_void_p(self.ptr))
        ffi.unregister_pointer(self.ptr)
        self.ptr = None

    def __del__(self) -> None:
        """Clean up resources and native handles held by INetworkDefinition."""
        self.destroy()


class IBuilderConfig:
    """IBuilderConfig class."""

    def __init__(self, ptr: int) -> None:
        """Initialize IBuilderConfig with configured parameters and state.

        Args:
            ptr: Parameter specifying ptr (int).
        """
        self.ptr = ptr
        ffi.register_pointer(self.ptr, self)

    def destroy(self) -> None:
        """Execute destroy."""
        if not self.ptr or getattr(ffi, "lib", None) is None:
            return
        destroy_cfg = getattr(ffi.lib, "destroyBuilderConfig", None)
        if destroy_cfg:
            destroy_cfg(ctypes.c_void_p(self.ptr))
        ffi.unregister_pointer(self.ptr)
        self.ptr = None

    def __del__(self) -> None:
        """Clean up resources and native handles held by IBuilderConfig."""
        self.destroy()
