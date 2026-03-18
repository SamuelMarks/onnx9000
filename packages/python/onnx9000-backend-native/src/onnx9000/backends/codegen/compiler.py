"""
C++ Compiler Utilities

Provides functionality to compile generated C++ code into shared libraries
and load them dynamically into Python via ctypes or pybind11.
"""

import ctypes
import importlib.util
import os
import subprocess
import sys
import tempfile


def compile_cpp(
    cpp_code: str, output_path: str = None, use_pybind: bool = True, extra_flags: list[str] = None
) -> str:
    """
    Compiles C++ source code into a shared library.

    Args:
        cpp_code: The generated C++ source code.
        output_path: The path where the shared library will be saved.
        use_pybind: Whether to include pybind11 headers during compilation.
        extra_flags: Additional compiler flags (e.g., ['-O3', '-ffast-math']).

    Returns:
        The path to the compiled shared library.
    """
    if extra_flags is None:
        extra_flags = []

    if use_pybind:
        ext = ".pyd" if sys.platform == "win32" else ".so"
    else:
        if sys.platform == "win32":
            ext = ".dll"
        elif sys.platform == "darwin":
            ext = ".dylib"
        else:
            ext = ".so"

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=ext, prefix="_model_")
        os.close(fd)

    fd_cpp, cpp_path = tempfile.mkstemp(suffix=".cpp")
    with os.fdopen(fd_cpp, "w") as f:
        f.write(cpp_code)

    compiler = os.environ.get("CXX", "clang++" if sys.platform == "darwin" else "g++")

    flags = [
        "-std=c++23",
        "-shared",
        "-fPIC",
        "-O3",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-Wno-sign-compare",
    ] + extra_flags

    include_dir = os.path.join(os.path.dirname(__file__), "..", "..", "include")
    flags.append(f"-I{os.path.abspath(include_dir)}")

    if use_pybind:
        try:
            import pybind11

            flags.extend([f"-I{pybind11.get_include()}", f"-I{pybind11.get_include(user=True)}"])
        except ImportError:
            pass

        import sysconfig

        flags.append(f"-I{sysconfig.get_path('include')}")

    if sys.platform == "darwin":
        flags.extend(["-undefined", "dynamic_lookup", "-framework", "Accelerate"])

    cmd = [compiler] + flags + ["-o", output_path, cpp_path]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Compilation failed:\n{e.stderr}")

    return output_path


def compile_wasm(
    cpp_code: str,
    output_path: str = None,
    extra_flags: list[str] = None,
    opt_level: str = "-O3",
    allow_memory_growth: bool = True,
    standalone_wasm: bool = False,
    enable_simd: bool = False,
    initial_memory: int = None,
    maximum_memory: int = None,
    environment: str = "web,worker,node",
    use_pthreads: bool = False,
    emit_tsd: bool = False,
) -> str:
    """
    Compiles C++ source code into a WebAssembly payload (.wasm) using Emscripten.
    """
    if extra_flags is None:
        extra_flags = []

    if output_path is None:
        suffix = ".wasm" if standalone_wasm else ".js"
        fd, output_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)

    fd_cpp, cpp_path = tempfile.mkstemp(suffix=".cpp")
    with os.fdopen(fd_cpp, "w") as f:
        f.write(cpp_code)

    compiler = os.environ.get("EMCC", "emcc")

    flags = [
        "-std=c++23",
        opt_level,
        "--no-entry",
        "-s",
        "EXPORTED_FUNCTIONS=['_malloc','_free']",
        "-s",
        "EXPORTED_RUNTIME_METHODS=['ccall','cwrap']",
        "-s",
        f"ALLOW_MEMORY_GROWTH={1 if allow_memory_growth else 0}",
    ]

    if emit_tsd:
        flags.extend(["--emit-tsd", output_path.replace(".js", ".d.ts").replace(".wasm", ".d.ts")])

    if not standalone_wasm:
        flags.extend(["-s", f"ENVIRONMENT={environment}"])

    if standalone_wasm:
        flags.extend(["-s", "STANDALONE_WASM=1"])

    if enable_simd:
        flags.append("-msimd128")
        # Ensure that code uses explicit intrinsic includes if needed by defining a macro
        flags.append("-DWASM_SIMD128_ENABLE=1")

    if use_pthreads:
        flags.extend(["-s", "USE_PTHREADS=1"])

    if initial_memory is not None:
        flags.extend(["-s", f"INITIAL_MEMORY={initial_memory}"])

    if maximum_memory is not None:
        flags.extend(["-s", f"MAXIMUM_MEMORY={maximum_memory}"])

    flags.extend(extra_flags)

    include_dir = os.path.join(os.path.dirname(__file__), "..", "..", "include")
    flags.append(f"-I{os.path.abspath(include_dir)}")

    cmd = [compiler] + flags + ["-o", output_path, cpp_path]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"WASM Compilation failed:\n{e.stderr}")
    except FileNotFoundError:
        raise RuntimeError(
            "emcc not found. Please install Emscripten and ensure it's in your PATH."
        )

    wasm_path = output_path.replace(".js", ".wasm")
    if os.path.exists(wasm_path):
        size_mb = os.path.getsize(wasm_path) / (1024 * 1024)
        if size_mb > 4.0:
            import warnings

            warnings.warn(
                f"Generated WASM payload is {size_mb:.2f}MB, which exceeds standard 4MB WebAssembly.instantiate limits. Use instantiateStreaming in the browser."
            )

    # Post-process to expose JS typed-array bridges for zero-copy input/output evaluation
    if not standalone_wasm and output_path.endswith(".js"):
        with open(output_path, "a") as f:
            f.write("""
// High-level TypedArray to C++ Pointer Bridge
Module.createBuffer = function(size, type = 'float32') {
    const bytes = type === 'float32' ? 4 : (type === 'int32' ? 4 : 8);
    const ptr = Module._malloc(size * bytes);
    let view;
    if (type === 'float32') view = new Float32Array(Module.HEAPF32.buffer, ptr, size);
    else if (type === 'int32') view = new Int32Array(Module.HEAP32.buffer, ptr, size);
    else view = new Float64Array(Module.HEAPF64.buffer, ptr, size);
    return { ptr, view };
};
Module.freeBuffer = function(ptr) {
    Module._free(ptr);
};
""")

    return output_path


def load_pybind_module(so_path: str, module_name: str = "_model"):
    """
    Dynamically loads a compiled Pybind11 shared library.
    """
    spec = importlib.util.spec_from_file_location(module_name, so_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {so_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_ctypes_library(so_path: str):
    """
    Dynamically loads a compiled C++ shared library via ctypes.
    """
    return ctypes.cdll.LoadLibrary(so_path)


def compile_static_lib(
    cpp_code: str, output_path: str = None, extra_flags: list[str] = None
) -> str:
    """
    Compiles C++ source code into a static library (.a / .lib).
    """
    if extra_flags is None:
        extra_flags = []

    ext = ".lib" if sys.platform == "win32" else ".a"

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=ext, prefix="lib_model_")
        os.close(fd)

    fd_cpp, cpp_path = tempfile.mkstemp(suffix=".cpp")
    with os.fdopen(fd_cpp, "w") as f:
        f.write(cpp_code)

    compiler = os.environ.get("CXX", "clang++" if sys.platform == "darwin" else "g++")

    # First compile to object file
    obj_path = cpp_path.replace(".cpp", ".o")
    flags = [
        "-std=c++23",
        "-c",
        "-fPIC",
        "-O3",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-Wno-sign-compare",
    ] + extra_flags

    include_dir = os.path.join(os.path.dirname(__file__), "..", "..", "include")
    flags.append(f"-I{os.path.abspath(include_dir)}")

    cmd = [compiler] + flags + ["-o", obj_path, cpp_path]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Object compilation failed:\\n{e.stderr}")

    # Now archive it
    if sys.platform == "win32":
        ar_cmd = ["lib", f"/OUT:{output_path}", obj_path]
    else:
        ar_cmd = ["ar", "rcs", output_path, obj_path]

    try:
        subprocess.run(ar_cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Archiving failed:\\n{e.stderr}")
    finally:
        if os.path.exists(obj_path):
            os.remove(obj_path)

    return output_path
