"""Module providing core logic and structural definitions."""

from pathlib import Path
from typing import Union

import numpy as np
from onnx9000.backends.memory.cpu_arena import CPUMemoryPlanner as plan_memory
from onnx9000.converters.jit.compiler import compile_cpp, compile_wasm, load_module
from onnx9000.converters.jit.hasher import hash_graph
from onnx9000.converters.jit.wrapper import CompiledModel
from onnx9000.core.parser.core import load


def compile(
    model_path: Union[str, Path], target: str = "cpp", out_dir: Union[str, Path, None] = None
) -> Union[CompiledModel, Path]:
    """Main JIT compilation entry point.

    Reads an ONNX file, generates C++, compiles it, and returns a callable Python wrapper.
    If target is 'wasm', returns the path to the generated JS file.
    """
    ir_graph = load(model_path)
    plan_memory(ir_graph)
    if target == "wasm":
        if out_dir is None:
            out_dir = Path.cwd() / "wasm_out"
        return compile_wasm(ir_graph, Path(out_dir))
    if target != "cpp":
        return None
    so_path = compile_cpp(ir_graph)
    module = load_module(so_path)
    cache_key = hash_graph(ir_graph)
    class_name = f"Model_{cache_key}"
    cpp_class = getattr(module, class_name)
    init_arrays = []
    for name in ir_graph.initializers:
        tensor = ir_graph.tensors[name]
        if tensor.data is None:
            shape_tuple = tuple(d.value if hasattr(d, "value") else d for d in tensor.shape)
            init_arrays.append(np.zeros(shape_tuple, dtype=np.float32))
        else:
            init_arrays.append(tensor.data)
    cpp_instance = cpp_class(*init_arrays)
    return CompiledModel(cpp_instance, ir_graph)


__all__ = ["compile", "compile_cpp", "load_module", "CompiledModel"]
