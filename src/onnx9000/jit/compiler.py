"""Module docstring."""

import importlib.resources
import importlib.util
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import pybind11
from jinja2 import Environment, PackageLoader

from onnx9000 import config
from onnx9000.codegen.generator import Generator
from onnx9000.dtypes import to_cpp_type
from onnx9000.exceptions import CompilationError
from onnx9000.ir import Graph
from onnx9000.jit.hasher import hash_graph
from onnx9000.utils.logger import logger


def _get_compiler() -> str:
    """Detects available C++ compiler."""
    if config.ONNX9000_COMPILER:
        return config.ONNX9000_COMPILER  # pragma: no cover
    if sys.platform == "win32":  # pragma: no cover
        # Usually requires MSVC environment to be set up  # pragma: no cover
        return "cl.exe"  # pragma: no cover
    else:  # pragma: no cover
        # Prefer clang++ or g++  # pragma: no cover
        if shutil.which("clang++"):  # pragma: no cover
            return "clang++"  # pragma: no cover
        elif shutil.which("g++"):  # pragma: no cover
            return "g++"  # pragma: no cover
        elif shutil.which("c++"):  # pragma: no cover
            return "c++"
    raise CompilationError(
        "No C++ compiler found (g++ or clang++)."
    )  # pragma: no cover


def compile_cpp(graph: Graph) -> Path:
    """
    Generates C++ code for the graph and compiles it into a shared library.
    Returns the path to the compiled extension.
    """
    cache_key = hash_graph(graph)
    cache_dir = config.ONNX9000_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    ext = ".pyd" if sys.platform == "win32" else ".so"
    out_path = cache_dir / f"onnx9000_{cache_key}{ext}"

    if out_path.exists():
        logger.info(
            f"Cache hit. Using pre-compiled module at {out_path}"
        )  # pragma: no cover
        return out_path  # pragma: no cover

    logger.info(f"Compiling new module to {out_path}")

    # Generate the C++ Model Class
    generator = Generator(graph, class_name=f"Model_{cache_key}")
    model_code = generator.generate()

    # Determine param types for pybind wrapper
    params_types = []
    for name in graph.initializers:
        dtype = graph.tensors[name].dtype
        params_types.append(to_cpp_type(dtype))

    # Render the complete file
    env = Environment(loader=PackageLoader("onnx9000", "templates"))

    header_template = env.get_template("base_header.hpp.j2")
    wrapper_template = env.get_template("pybind_wrapper.cpp.j2")

    full_code = []
    full_code.append("#include <pybind11/pybind11.h>")
    full_code.append("#include <pybind11/numpy.h>")
    full_code.append(header_template.render())
    full_code.append(model_code)
    full_code.append(
        wrapper_template.render(
            module_name=f"onnx9000_{cache_key}",
            class_name=f"Model_{cache_key}",
            params_types=params_types,
        )
    )

    cpp_path = cache_dir / f"onnx9000_{cache_key}.cpp"
    with open(cpp_path, "w") as f:
        f.write("\n".join(full_code))

    # Compile
    compiler = _get_compiler()
    pybind_inc = pybind11.get_include()
    py_inc = sysconfig.get_path("include")

    # We may also need numpy includes
    import numpy as np

    np_inc = np.get_include()

    cmd = [
        compiler,
        "-O3",
        "-shared",
        "-std=c++23",
        "-fPIC",
        f"-I{pybind_inc}",
        f"-I{py_inc}",
        f"-I{np_inc}",
        str(cpp_path),
        "-o",
        str(out_path),
    ]

    # If darwin, we need to handle undefined symbols from python dynamically
    if sys.platform == "darwin":  # pragma: no cover
        cmd.extend(["-undefined", "dynamic_lookup"])
        if config.ONNX9000_USE_ACCELERATE:
            cmd.extend(["-DUSE_ACCELERATE=1", "-framework", "Accelerate"])

    # Add OpenMP if supported/requested
    if sys.platform != "darwin":  # pragma: no cover
        cmd.append("-fopenmp")  # Default clang on mac doesn't have omp by default

    try:
        logger.debug(f"Running compilation: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:  # pragma: no cover
        raise CompilationError(
            f"C++ Compilation failed:\n{e.stderr}\n{e.stdout}"
        ) from e  # pragma: no cover

    if not config.ONNX9000_DEBUG:
        cpp_path.unlink(missing_ok=True)

    return out_path


def compile_wasm(graph: Graph, out_dir: Path) -> Path:  # pragma: no cover
    """
    Generates C++ code and compiles it to WASM using Emscripten.
    Returns the path to the generated .js file.
    """
    cache_key = hash_graph(graph)
    out_dir.mkdir(parents=True, exist_ok=True)

    js_path = out_dir / f"onnx9000_{cache_key}.js"
    wasm_path = out_dir / f"onnx9000_{cache_key}.wasm"

    if js_path.exists() and wasm_path.exists():
        logger.info(f"Cache hit. Using pre-compiled WASM at {js_path}")
        return js_path

    logger.info(f"Compiling new WASM module to {js_path}")

    generator = Generator(graph, class_name=f"Model_{cache_key}")
    model_code = generator.generate()

    env = Environment(loader=PackageLoader("onnx9000", "templates"))

    header_template = env.get_template("base_header.hpp.j2")
    wrapper_template = env.get_template("embind_wrapper.cpp.j2")

    full_code = []
    full_code.append(header_template.render())
    full_code.append(model_code)
    full_code.append(
        wrapper_template.render(
            module_name=f"onnx9000_{cache_key}", class_name=f"Model_{cache_key}"
        )
    )

    cpp_path = out_dir / f"onnx9000_{cache_key}.cpp"
    with open(cpp_path, "w") as f:
        f.write("\n".join(full_code))

    compiler = config.ONNX9000_WASM_COMPILER

    if not shutil.which(compiler):
        raise CompilationError(
            f"Emscripten compiler '{compiler}' not found. Please install Emscripten or activate emsdk."
        )

    cmd = [
        compiler,
        "-O3",
        "-std=c++23",
        "--bind",
        "-msimd128",
        "-s",
        "WASM=1",
        "-s",
        "ALLOW_MEMORY_GROWTH=1",
        str(cpp_path),
        "-o",
        str(js_path),
    ]

    try:
        logger.debug(f"Running WASM compilation: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise CompilationError(
            f"WASM Compilation failed:\n{e.stderr}\n{e.stdout}"
        ) from e

    if not config.ONNX9000_DEBUG:
        cpp_path.unlink(missing_ok=True)

    return js_path


def load_module(module_path: Path):
    """Loads the compiled shared library into Python."""
    module_name = module_path.stem
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise CompilationError(
            f"Failed to load module {module_path}"
        )  # pragma: no cover

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
