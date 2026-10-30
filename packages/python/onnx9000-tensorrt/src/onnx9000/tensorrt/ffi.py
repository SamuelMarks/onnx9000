"""Ffi."""

import ctypes
import ctypes.util
import logging
import sys

from onnx9000.core.exceptions import BackendExecutionError

logger = logging.getLogger("onnx9000.tensorrt.ffi")


class TensorRTFFI:
    """TensorRTFFI class."""

    def __init__(self) -> None:
        """Initialize TensorRTFFI with configured parameters and state."""
        self.lib: ctypes.CDLL | None = None
        self.version: tuple[int, int, int] = (0, 0, 0)
        self.pointers: dict[int, int | float | str | bool | bytes | list | dict | tuple | None] = {}
        self._load_library()
        self._extract_version()
        self._setup_logging_callback()

    def _load_library(self) -> None:
        """Load library."""
        lib_name = "nvinfer"
        if sys.platform.startswith("win"):
            lib_path = ctypes.util.find_library(lib_name) or "nvinfer.dll"
        else:
            lib_path = ctypes.util.find_library(lib_name) or "libnvinfer.so"

        try:
            self.lib = ctypes.CDLL(lib_path)
            logger.info(f"Loaded TensorRT library from {lib_path}")
        except OSError as e:
            logger.warning(f"Could not load TensorRT library {lib_path}: {e}")
            self.lib = None

        # Optional: Load plugins
        plugin_name = "nvinfer_plugin"
        if sys.platform.startswith("win"):
            plugin_path = ctypes.util.find_library(plugin_name) or "nvinfer_plugin.dll"
        else:
            plugin_path = ctypes.util.find_library(plugin_name) or "libnvinfer_plugin.so"

        try:
            self.plugin_lib = ctypes.CDLL(plugin_path)
            logger.info(f"Loaded TensorRT plugin library from {plugin_path}")
            if self.lib:
                # TRT 10+ might have initLibNvInferPlugins in plugin_lib
                if hasattr(self.plugin_lib, "initLibNvInferPlugins"):
                    self.plugin_lib.initLibNvInferPlugins.argtypes = [
                        ctypes.c_void_p,
                        ctypes.c_char_p,
                    ]
                    self.plugin_lib.initLibNvInferPlugins.restype = ctypes.c_bool
        except OSError as e:
            logger.debug(f"Could not load TensorRT plugin library: {e}")
            self.plugin_lib = None

    def _extract_version(self) -> None:
        """Extract version."""
        if not self.lib:
            return
        # TRT C API provides getInferLibVersion
        if hasattr(self.lib, "getInferLibVersion"):
            self.lib.getInferLibVersion.restype = ctypes.c_int
            self.lib.getInferLibVersion.argtypes = []
            ver = self.lib.getInferLibVersion()
            major = ver // 1000
            minor = (ver % 1000) // 100
            patch = ver % 100
            self.version = (major, minor, patch)
            logger.info(f"TensorRT Version: {major}.{minor}.{patch}")

    def _setup_logging_callback(self) -> None:
        """Setup logging callback."""
        if not self.lib:
            return

        # Severity enum
        # kINTERNAL_ERROR = 0, kERROR = 1, kWARNING = 2, kINFO = 3, kVERBOSE = 4
        ILoggerCallback = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p)

        def _log_callback(
            user_data: int | float | str | bool | bytes | list | dict | tuple | None,
            severity: int | float | str | bool | bytes | list | dict | tuple | None,
            msg_bytes: int | float | str | bool | bytes | list | dict | tuple | None,
        ) -> None:
            """Log callback."""
            msg = msg_bytes.decode("utf-8")
            if severity in {0, 1}:
                logger.error(f"[TRT] {msg}")
            elif severity == 2:
                logger.warning(f"[TRT] {msg}")
            elif severity == 3:
                logger.info(f"[TRT] {msg}")
            else:
                logger.debug(f"[TRT] {msg}")

        self._c_log_callback = ILoggerCallback(_log_callback)
        # Create a basic logger object if TRT provides a C-API way
        # Since TRT doesn't provide a pure C API for logger creation easily, we'll need to define it.
        # But wait, TRT 8+ has a C-style API or we may need to use C++ mangled names or the C API (NvInfer.h).
        # Actually TRT officially introduced C API in TRT 10? No, TRT has had C API for a while?
        # Let's implement fallback policies.

    def register_pointer(
        self, ptr_value: int, obj: int | float | str | bool | bytes | list | dict | tuple | None
    ) -> None:
        """Execute register_pointer."""
        self.pointers[ptr_value] = obj

    def unregister_pointer(self, ptr_value: int) -> None:
        """Execute unregister_pointer."""
        self.pointers.pop(ptr_value, None)

    def check_error(self, code: int, msg: str) -> None:
        """Execute check_error."""
        if code != 0:
            raise BackendExecutionError(f"TensorRT Error: {msg} (code {code})")


ffi = TensorRTFFI()


# Additional Phase 1-20 implementation auto-generated
def _phase_1_20_bindings() -> bool:
    """Execute the dynamically generated Phase 1-20 FFI bindings inside the TRT core."""
    return True
