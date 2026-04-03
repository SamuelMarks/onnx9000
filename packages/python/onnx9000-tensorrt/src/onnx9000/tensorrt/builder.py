"""Module docstring."""

import ctypes

from onnx9000.tensorrt.enums import MemoryPoolType
from onnx9000.tensorrt.ffi import ffi


class BuilderConfig:
    """BuilderConfig class."""

    def __init__(self, ptr: int):
        """Initialize."""
        self.ptr = ptr

    def set_memory_pool_limit(self, pool_type: MemoryPoolType, size: int):
        """Execute set_memory_pool_limit."""
        if hasattr(ffi.lib, "setMemoryPoolLimit"):
            ffi.lib.setMemoryPoolLimit(
                ctypes.c_void_p(self.ptr), ctypes.c_int(pool_type.value), ctypes.c_size_t(size)
            )
        elif hasattr(ffi.lib, "setMaxWorkspaceSize"):
            ffi.lib.setMaxWorkspaceSize(ctypes.c_void_p(self.ptr), ctypes.c_size_t(size))


class NetworkDefinition:
    """NetworkDefinition class."""

    def __init__(self, ptr: int):
        """Initialize."""
        self.ptr = ptr
        self.tensors = {}

    def destroy(self):
        """Execute destroy."""
        destroy_func = getattr(ffi.lib, "destroyNetworkDefinition", None)
        if destroy_func:
            destroy_func(ctypes.c_void_p(self.ptr))


class Builder:
    """Builder class."""

    def __init__(self, logger_callback=None):
        """Initialize."""
        if not ffi.lib:
            raise RuntimeError("TensorRT library not loaded")

        create_builder = getattr(ffi.lib, "createInferBuilder_INTERNAL", None)
        if not create_builder:
            raise RuntimeError("createInferBuilder_INTERNAL not found in TensorRT lib")

        create_builder.restype = ctypes.c_void_p
        create_builder.argtypes = [ctypes.c_void_p, ctypes.c_int]

        # We need a proper C++ logger object here.
        # NvInfer C++ API needs an ILogger derived object. TRT 8 doesn't provide a direct C API for this.
        # A full FFI implementation requires either a small C shim or implementing the exact vtable layout of ILogger.
        # But wait! TRT 10+ has `getPluginRegistry()`, and many functions but still uses classes for ILogger.
        # Actually, if we just pass a null pointer for logger, will it work? No, `createInferBuilder` requires a valid ILogger pointer.
        # Let's write a mock vtable for ILogger for Linux x64 or use standard fallback policies if we are building a zero-build parser.

        class ILoggerVTable(ctypes.Structure):
            """ILoggerVTable class."""

            _fields_ = [
                ("log", ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p))
            ]

        class ILogger(ctypes.Structure):
            """ILogger class."""

            _fields_ = [("vtable", ctypes.POINTER(ILoggerVTable))]

        self.vtable = ILoggerVTable(ffi._c_log_callback)
        self.logger = ILogger(ctypes.pointer(self.vtable))

        # TRT versions. E.g. 10.0.0
        # Usually NYI = NV_TENSORRT_VERSION
        trt_version = ffi.version[0] * 10000 + ffi.version[1] * 100 + ffi.version[2]
        if trt_version == 0:
            trt_version = 80600  # default to 8.6

        self.ptr = create_builder(ctypes.pointer(self.logger), ctypes.c_int(trt_version))
        if not self.ptr:
            raise RuntimeError("Failed to create TensorRT Builder")

        ffi.register_pointer(self.ptr, self)

    def create_network(self) -> NetworkDefinition:
        """Execute create_network."""
        create_network = getattr(ffi.lib, "createNetworkV2", None)
        if not create_network:
            raise RuntimeError("createNetworkV2 not found")

        # 1 << 0 for Explicit Batch
        flags = 1 << 0
        create_network.restype = ctypes.c_void_p
        create_network.argtypes = [ctypes.c_void_p, ctypes.c_int32]

        net_ptr = create_network(ctypes.c_void_p(self.ptr), ctypes.c_int32(flags))
        if not net_ptr:
            raise RuntimeError("Failed to create NetworkDefinition")
        return NetworkDefinition(net_ptr)

    def create_builder_config(self) -> BuilderConfig:
        """Execute create_builder_config."""
        create_config = getattr(ffi.lib, "createBuilderConfig", None)
        if not create_config:
            raise RuntimeError("createBuilderConfig not found")
        create_config.restype = ctypes.c_void_p
        create_config.argtypes = [ctypes.c_void_p]
        cfg_ptr = create_config(ctypes.c_void_p(self.ptr))
        return BuilderConfig(cfg_ptr)

    def destroy(self):
        """Execute destroy."""
        if not self.ptr:
            return
        destroy_builder = getattr(ffi.lib, "destroyInferBuilder", None)
        if destroy_builder:
            destroy_builder(ctypes.c_void_p(self.ptr))
        ffi.unregister_pointer(self.ptr)
        self.ptr = None

    def __del__(self):
        """Initialize."""
        self.destroy()
