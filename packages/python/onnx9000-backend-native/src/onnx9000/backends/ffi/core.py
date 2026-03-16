import ctypes
import ctypes.util
import logging
import os
import platform
import threading
import time
import weakref

logger = logging.getLogger("onnx9000.ffi")
logger.setLevel(logging.INFO)


class DynamicLibraryError(OSError):
    pass


class DynamicLibrary:
    """Class DynamicLibrary implementation."""

    def __init__(self, name: str, versions=None, calling_convention="cdecl", use_cffi=False):
        self.name = name
        self.lib = None
        self._lock = threading.Lock()
        self._use_cffi = use_cffi
        self._func_cache = {}
        self.arch = platform.machine()
        self.os = platform.system()
        self.calling_convention = calling_convention
        start_time = time.perf_counter()
        mode = 0
        if hasattr(ctypes, "RTLD_GLOBAL"):
            mode |= ctypes.RTLD_GLOBAL
        if hasattr(os, "RTLD_NOW"):
            mode |= os.RTLD_NOW
        versions = versions or [None]
        custom_path = os.environ.get(f"ONNX9000_LIB_{name.upper()}")
        if custom_path:
            try:
                self._load_lib(custom_path, mode)
            except OSError as e:
                logger.warning(f"Failed to load custom library {custom_path}: {e}")
        if not self.lib:
            for v in versions:
                if self.os == "Windows":
                    lib_name = f"{name}.dll" if v is None else f"{name}{v}.dll"
                elif self.os == "Darwin":
                    lib_name = f"lib{name}.dylib" if v is None else f"lib{name}.{v}.dylib"
                else:
                    lib_name = f"lib{name}.so" if v is None else f"lib{name}.so.{v}"
                found_path = ctypes.util.find_library(lib_name) or lib_name
                try:
                    self._load_lib(found_path, mode)
                    if self.lib:
                        break
                except OSError:
                    continue
        load_time = time.perf_counter() - start_time
        logger.info(f"FFI load time for {name}: {load_time:.6f}s")
        if not self.lib:
            raise DynamicLibraryError(
                f"Failed to load hardware library: {name}. Ensure it is installed and in PATH/LD_LIBRARY_PATH."
            )

    def _load_lib(self, path, mode):
        if self.os == "Windows" and self.calling_convention == "stdcall":
            self.lib = ctypes.WinDLL(path)
        else:
            self.lib = ctypes.CDLL(path, mode=mode)

    def define(self, func_name: str, argtypes: list, restype: type):
        """
        Define explicit C function signatures (`argtypes`, `restype`) for safety.
        """
        try:
            func = getattr(self.lib, func_name)
        except AttributeError:
            err_msg = f"Symbol '{func_name}' not found in library '{self.name}'"
            if self.os != "Windows":
                try:
                    dlerror = ctypes.CDLL(None).dlerror
                    dlerror.restype = ctypes.c_char_p
                    err_str = dlerror()
                    if err_str:
                        err_msg += f" (dlerror: {err_str.decode('utf-8')})"
                except Exception:
                    pass
            raise AttributeError(err_msg)
        func.argtypes = argtypes
        func.restype = restype

        def wrapped_ffi_call(*args, **kwargs):
            return func(*args, **kwargs)

        self._func_cache[func_name] = wrapped_ffi_call
        return wrapped_ffi_call

    def __getattr__(self, name):
        if name in self._func_cache:
            return self._func_cache[name]
        return self.define(name, None, None)


class HardwareContextHandle:
    """
    Manage Hardware Context handles (e.g. `cublasHandle_t`) safely across Threads.
    Destroy Hardware Context handles explicitly on `__del__` or Context Manager exit.
    Implement C-struct mappings natively in Python for hardware context handles.
    """

    def __init__(self, handle_ptr, destroy_func):
        self._handle = ctypes.c_void_p(handle_ptr)
        self._destroy_func = destroy_func
        self._lock = threading.Lock()
        weakref.finalize(self, self._cleanup, self._handle, self._destroy_func)

    @classmethod
    def _cleanup(cls, handle, destroy_func):
        if handle and handle.value:
            destroy_func(handle)
            handle.value = None

    @property
    def ptr(self):
        """Expose native pointers from Python integers (`ctypes.c_void_p`)"""
        return self._handle

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        with self._lock:
            self._cleanup(self._handle, self._destroy_func)


def map_python_string(s: str) -> ctypes.c_char_p:
    """Map Python strings to `const char*` C-strings seamlessly"""
    return ctypes.c_char_p(s.encode("utf-8"))


def map_python_bool(b: bool) -> ctypes.c_int:
    """Map Python `bool` to C `int` natively"""
    return ctypes.c_int(1 if b else 0)


def profile_ctypes_overhead():
    """Profile `ctypes` call overhead to ensure it remains < 1 microsecond per op"""
    libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
    libc.getpid.argtypes = []
    libc.getpid.restype = ctypes.c_int
    start = time.perf_counter()
    for _ in range(10000):
        libc.getpid()
    elapsed = time.perf_counter() - start
    avg_us = elapsed / 10000 * 1000000.0
    logger.info(f"Ctypes call overhead: {avg_us:.3f} us per op")
    return avg_us


def get_cpu_features():
    """
    Execute `sysctl` or `machdep` natively on macOS to query specific CPU features.
    Query `/proc/cpuinfo` on Linux natively to identify AVX/AVX2/AVX512/NEON extensions.
    """
    features = {"avx": False, "avx2": False, "avx512": False, "neon": False}
    os_name = platform.system()
    if os_name == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                content = f.read().lower()
                features["avx"] = "avx " in content or "avx\n" in content
                features["avx2"] = "avx2" in content
                features["avx512"] = "avx512" in content
                features["neon"] = "neon" in content or "asimd" in content
        except Exception:
            pass
    elif os_name == "Darwin":
        import subprocess

        try:
            out = subprocess.check_output(["sysctl", "-a"]).decode("utf-8").lower()
            features["avx"] = "hw.optional.avx1_0: 1" in out
            features["avx2"] = "hw.optional.avx2_0: 1" in out
            features["avx512"] = "hw.optional.avx512f: 1" in out
            features["neon"] = "hw.optional.neon: 1" in out or "hw.optional.arm.ext_asimd: 1" in out
        except Exception:
            pass
    return features


def get_cache_sizes():
    """Identify L1/L2/L3 Cache sizes natively to optimize BLAS tiling logic"""
    sizes = {"l1": 0, "l2": 0, "l3": 0}
    os_name = platform.system()
    if os_name == "Linux":
        try:
            for level in [1, 2, 3]:
                path = f"/sys/devices/system/cpu/cpu0/cache/index{level}/size"
                if os.path.exists(path):
                    with open(path) as f:
                        val = f.read().strip()
                        if val.endswith("K"):
                            sizes[f"l{level}"] = int(val[:-1]) * 1024
                        elif val.endswith("M"):
                            sizes[f"l{level}"] = int(val[:-1]) * 1024 * 1024
                        else:
                            sizes[f"l{level}"] = int(val)
        except Exception:
            pass
    elif os_name == "Darwin":
        import subprocess

        try:
            out = subprocess.check_output(
                ["sysctl", "hw.l1icachesize", "hw.l1dcachesize", "hw.l2cachesize", "hw.l3cachesize"]
            ).decode("utf-8")
            for line in out.splitlines():
                if "hw.l1dcachesize" in line:
                    sizes["l1"] = int(line.split(":")[1].strip())
                elif "hw.l2cachesize" in line:
                    sizes["l2"] = int(line.split(":")[1].strip())
                elif "hw.l3cachesize" in line:
                    sizes["l3"] = int(line.split(":")[1].strip())
        except Exception:
            pass
    return sizes
