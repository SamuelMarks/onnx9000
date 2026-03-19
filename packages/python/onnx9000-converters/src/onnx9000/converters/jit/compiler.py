"""Module providing core logic and structural definitions."""

import importlib.resources
import importlib.util
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import pybind11
from jinja2 import Environment, PackageLoader
from onnx9000.backends.codegen.generator import Generator
from onnx9000.converters.jit.hasher import hash_graph
from onnx9000.core import config
from onnx9000.core.dtypes import to_cpp_type
from onnx9000.core.exceptions import CompilationError
from onnx9000.core.ir import Graph
from onnx9000.core.logger import get_logger

logger = get_logger(__name__)


def _get_compiler() -> str:
    """Detects available C++ compiler."""
    if config.ONNX9000_COMPILER:
        return config.ONNX9000_COMPILER
    if sys.platform == "win32":
        return "cl.exe"
    elif shutil.which("clang++"):
        return "clang++"
    elif shutil.which("g++"):
        return "g++"
    elif shutil.which("c++"):
        return "c++"
    raise CompilationError("No C++ compiler found (g++ or clang++).")


def compile_cpp(graph: Graph) -> Path:
    """Generate C++ code for the graph and compiles it into a shared library.

    Returns the path to the compiled extension.
    """
    cache_key = hash_graph(graph)
    cache_dir = config.ONNX9000_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    ext = ".pyd" if sys.platform == "win32" else ".so"
    out_path = cache_dir / f"onnx9000_{cache_key}{ext}"
    if out_path.exists():
        logger.info(f"Cache hit. Using pre-compiled module at {out_path}")
        return out_path
    logger.info(f"Compiling new module to {out_path}")
    generator = Generator(graph, class_name=f"Model_{cache_key}")
    model_code = generator.generate()
    params_types = []
    for name in graph.initializers:
        dtype = graph.tensors[name].dtype
        params_types.append(to_cpp_type(dtype))
    env = Environment(loader=PackageLoader("onnx9000", "backends/templates"))
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
    compiler = _get_compiler()
    pybind_inc = pybind11.get_include()
    py_inc = sysconfig.get_path("include")
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
    if sys.platform == "darwin":
        cmd.extend(["-undefined", "dynamic_lookup"])
        if config.ONNX9000_USE_ACCELERATE:
            cmd.extend(["-DUSE_ACCELERATE=1", "-framework", "Accelerate"])
    if sys.platform != "darwin":
        cmd.append("-fopenmp")
    try:
        logger.debug(f"Running compilation: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise CompilationError(f"C++ Compilation failed:\n{e.stderr}\n{e.stdout}") from e
    if not config.ONNX9000_DEBUG:
        cpp_path.unlink(missing_ok=True)
    return out_path


def compile_wasm(graph: Graph, out_dir: Path) -> Path:
    """Generate C++ code and compiles it to WASM using Emscripten.

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
    env = Environment(loader=PackageLoader("onnx9000", "backends/templates"))
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
        raise CompilationError(f"WASM Compilation failed:\n{e.stderr}\n{e.stdout}") from e
    if not config.ONNX9000_DEBUG:
        cpp_path.unlink(missing_ok=True)
    return js_path


def load_module(module_path: Path):
    """Load the compiled shared library into Python."""
    module_name = module_path.stem
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise CompilationError(f"Failed to load module {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
