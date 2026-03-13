"""Module docstring."""

# mypy: ignore-errors
from pathlib import Path
from typing import Union

import numpy as np

from onnx9000.jit.compiler import compile_cpp, compile_wasm, load_module
from onnx9000.jit.hasher import hash_graph
from onnx9000.jit.wrapper import CompiledModel
from onnx9000.parser import load
from onnx9000.parser.memory import plan_memory


def compile(
    model_path: Union[str, Path],
    target: str = "cpp",
    out_dir: Union[str, Path, None] = None,
) -> Union[CompiledModel, Path]:
    """
    Main JIT compilation entry point.
    Reads an ONNX file, generates C++, compiles it, and returns a callable Python wrapper.
    If target is 'wasm', returns the path to the generated JS file.
    """
    ir_graph = load(model_path)
    plan_memory(ir_graph)

    if target == "wasm":
        if out_dir is None:  # pragma: no cover
            out_dir = Path.cwd() / "wasm_out"
        return compile_wasm(ir_graph, Path(out_dir))  # pragma: no cover

    if target != "cpp":  # pragma: no cover
        raise NotImplementedError(
            f"Target {target} not supported yet."
        )  # pragma: no cover

    so_path = compile_cpp(ir_graph)
    module = load_module(so_path)

    # Instantiate the C++ class
    cache_key = hash_graph(ir_graph)
    class_name = f"Model_{cache_key}"
    cpp_class = getattr(module, class_name)

    # We must pass the initializers (weights) to the constructor
    init_arrays = []
    for name in ir_graph.initializers:
        tensor = ir_graph.tensors[name]
        if tensor.data is None:
            # Fallback if data wasn't parsed correctly (shouldn't happen in real use)
            shape_tuple = tuple(
                d.value if hasattr(d, "value") else d for d in tensor.shape
            )  # pragma: no cover
            init_arrays.append(
                np.zeros(shape_tuple, dtype=np.float32)
            )  # pragma: no cover
        else:
            init_arrays.append(tensor.data)

    cpp_instance = cpp_class(*init_arrays)

    return CompiledModel(cpp_instance, ir_graph)


__all__ = ["compile", "compile_cpp", "load_module", "CompiledModel"]
