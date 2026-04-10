"""C++ Code Generation Utilities.

Translates ONNX operations to equivalent C++ bindings and memory buffers.
"""

import os

from jinja2 import Environment, FileSystemLoader
from onnx9000.backends.codegen.utils import sanitize_name
from onnx9000.core import config
from onnx9000.core.dtypes import to_cpp_type
from onnx9000.core.ir import Graph
from onnx9000.core.registry import global_registry as registry


class Generator:
    """Core C++ Code Generator orchestrating the translation of an ir.Graph.

    into a bespoke C++ class using Jinja2 templates.
    """

    def __init__(self, graph: Graph, class_name: str = "GeneratedModel") -> None:
        """Initialize the code generator for a given Graph."""
        self.graph = graph
        self.class_name = class_name
        self.code_blocks: list[str] = []

        from onnx9000.core.memory_planner import simulate_memory_plan
        from onnx9000.core.shape_inference import infer_shapes_and_types

        # Implement static shape and dtype resolution pass ahead of transpilation
        infer_shapes_and_types(self.graph)

        # Implement global contiguous Memory Arena calculator for all intermediate tensors
        arena = simulate_memory_plan(self.graph)
        self.tensor_offsets = arena.tensor_offsets
        self.peak_memory = arena.peak_memory

        # Setup Jinja2 Environment
        template_dir = os.path.dirname(os.path.abspath(__file__))
        self.env = Environment(loader=FileSystemLoader(template_dir))

    def generate(
        self,
        embed_constants: bool = False,
        mmap_constants: bool = False,
        mmap_file: str = "weights.bin",
        pybind_module_name: str = "_model",
        isolated_execution_function: bool = False,
    ) -> str:
        """Assembles the complete C++ class."""
        template = self.env.get_template("template.cpp.j2")

        # Constructor Params & Inits
        constructor_params_list = []
        constructor_inits_list = []
        initializers_data = []
        mmap_offsets = {}
        current_mmap_offset = 0

        if mmap_constants:
            import struct

            mmap_fp = open(mmap_file, "wb")
        else:
            mmap_fp = None

        for name in self.graph.initializers:
            tensor = self.graph.tensors[name]
            cpp_type = to_cpp_type(tensor.dtype)
            clean_name = sanitize_name(name)
            shape_str = "{" + ", ".join(map(str, tensor.shape)) + "}"

            data_bytes = []
            if embed_constants or mmap_constants:
                if tensor.data is not None:
                    data_bytes = list(tensor.data)
                elif hasattr(tensor, "values") and tensor.values is not None:
                    import struct

                    fmt = f"{len(tensor.values)}f"
                    if cpp_type == "int64_t":
                        fmt = f"{len(tensor.values)}q"
                    elif cpp_type == "int32_t":
                        fmt = f"{len(tensor.values)}i"
                    elif cpp_type == "bool":
                        fmt = f"{len(tensor.values)}?"
                    data_bytes = list(struct.pack(fmt, *tensor.values))

            if mmap_constants and mmap_fp:
                mmap_offsets[clean_name] = current_mmap_offset
                mmap_fp.write(bytes(data_bytes))
                current_mmap_offset += len(data_bytes)

            if not embed_constants and not mmap_constants:
                constructor_params_list.append(f"{cpp_type}* {clean_name}_ptr")
                constructor_inits_list.append(f"{clean_name}({clean_name}_ptr, {shape_str})")
            elif mmap_constants:
                # We will initialize these from the mmap pointer inside the constructor body
                constructor_inits_list.append(f"{clean_name}(nullptr, {shape_str})")
            elif embed_constants:
                # We will initialize these from the embedded arrays
                constructor_inits_list.append(
                    f"{clean_name}(reinterpret_cast<{cpp_type}*>(const_cast<uint8_t*>({clean_name}_data)), {shape_str})"
                )

            initializers_data.append(
                {
                    "clean_name": clean_name,
                    "cpp_type": cpp_type,
                    "shape_str": shape_str,
                    "data_bytes": data_bytes,
                    "mmap_offset": mmap_offsets.get(clean_name, 0),
                }
            )

        if mmap_fp:
            mmap_fp.close()

        constructor_params = ", ".join(constructor_params_list)
        constructor_inits = (
            " : " + ", ".join(constructor_inits_list) if constructor_inits_list else ""
        )
        if not embed_constants and not mmap_constants:
            pybind_init_params = ", ".join([f"{i['cpp_type']}*" for i in initializers_data])
        else:
            pybind_init_params = ""

        # Forward Params
        forward_params_list = []
        seen = set()
        inputs_data = []
        unique_inputs = []

        for name in self.graph.inputs:
            tensor = self.graph.tensors[name]
            cpp_type = to_cpp_type(tensor.dtype)
            clean_name = sanitize_name(name)

            shape_str = "{" + ", ".join(map(str, tensor.shape)) + "}"
            inputs_data.append(
                {"clean_name": clean_name, "cpp_type": cpp_type, "shape_str": shape_str}
            )

            if clean_name not in seen:
                forward_params_list.append(f"{cpp_type}* {clean_name}_ptr")
                seen.add(clean_name)
                unique_inputs.append({"clean_name": clean_name, "cpp_type": cpp_type})

        outputs_data = []
        ret_types = []
        out_returns = []

        for i, name in enumerate(self.graph.outputs):
            tensor = self.graph.tensors[name]
            cpp_type = to_cpp_type(tensor.dtype)
            clean_name = sanitize_name(name)
            shape_str = "{" + ", ".join(map(str, tensor.shape)) + "}"
            forward_params_list.append(f"{cpp_type}* {clean_name}_out_ptr_{i}")
            outputs_data.append(
                {
                    "clean_name": clean_name,
                    "cpp_type": cpp_type,
                    "offset": self.tensor_offsets.get(name),
                    "shape_str": shape_str,
                }
            )
            ret_types.append(f"pybind11::array_t<{cpp_type}>")
            out_returns.append(f"{clean_name}_out_arr_{i}")

        forward_params = ", ".join(forward_params_list)

        # Calculate used outputs for DCE
        used_tensors = set(self.graph.outputs)
        for node in reversed(self.graph.nodes):
            # If none of this node's outputs are used, and it's not a node with side effects
            if not any(out in used_tensors for out in node.outputs):
                continue
            for inp in node.inputs:
                used_tensors.add(inp)

        # Op Codes
        op_codes = []
        for node in self.graph.nodes:
            # Transpiler Level DCE (Dead Code Elimination)
            if not any(out in used_tensors for out in node.outputs):
                continue

            domain = getattr(node, "domain", "")
            op_gen = registry.get_op(domain, node.op_type)
            op_code = op_gen(node, self)
            op_codes.append(op_code)

        # Forward Py Params
        forward_py_params_list = []
        for inp in inputs_data:
            forward_py_params_list.append(
                f"pybind11::array_t<{inp['cpp_type']}, pybind11::array::c_style | pybind11::array::forcecast> {inp['clean_name']}_arr"
            )
        forward_py_params = ", ".join(forward_py_params_list)

        ret_type_str = (
            f"std::tuple<{', '.join(ret_types)}>"
            if len(ret_types) > 1
            else ret_types[0]
            if ret_types
            else "void"
        )
        output_returns = ", ".join(out_returns)

        forward_call_args_list = []
        for inp in inputs_data:
            forward_call_args_list.append(f"{inp['clean_name']}_ptr")
        for _ in self.graph.outputs:
            forward_call_args_list.append("nullptr")
        forward_call_args = ", ".join(forward_call_args_list)

        c_api_null_args = (
            ", ".join(["nullptr"] * len(initializers_data)) if pybind_init_params else ""
        )

        c_api_forward_params = ""
        c_api_forward_call_args_list = []
        for inp in inputs_data:
            c_api_forward_params += f", {inp['cpp_type']}* {inp['clean_name']}"
            c_api_forward_call_args_list.append(inp["clean_name"])
        for out in outputs_data:
            c_api_forward_params += f", {out['cpp_type']}* {out['clean_name']}"
            c_api_forward_call_args_list.append(out["clean_name"])

        c_api_forward_call_args = ", ".join(c_api_forward_call_args_list)

        model_name = getattr(self.graph, "name", "onnx9000_model")
        model_version = getattr(self.graph, "version", "1")
        model_producer = getattr(self.graph, "producer", "onnx9000")
        doc_string = getattr(self.graph, "doc_string", "ONNX Model translated by onnx9000")

        return template.render(
            use_cuda=config.ONNX9000_USE_CUDA,
            class_name=self.class_name,
            constructor_params=constructor_params,
            constructor_inits=constructor_inits,
            peak_memory=self.peak_memory,
            forward_params=forward_params,
            inputs=inputs_data,
            op_codes=op_codes,
            outputs=outputs_data,
            forward_py_ret_type=ret_type_str,
            forward_py_params=forward_py_params,
            unique_inputs=unique_inputs,
            forward_call_args=forward_call_args,
            output_returns=output_returns,
            initializers=initializers_data,
            pybind_init_params=pybind_init_params,
            embed_constants=embed_constants,
            mmap_constants=mmap_constants,
            mmap_file=mmap_file,
            pybind_module_name=pybind_module_name,
            c_api_null_args=c_api_null_args,
            c_api_forward_params=c_api_forward_params,
            c_api_forward_call_args=c_api_forward_call_args,
            model_name=model_name,
            model_version=model_version,
            model_producer=model_producer,
            doc_string=doc_string,
            isolated_execution_function=isolated_execution_function,
        )

    def write_to_directory(self, output_dir: str, run_clang_format: bool = True, **generate_kwargs):
        """Generate the C++ code and writes it to a specific directory alongside a CMakeLists.txt.

        Optionally runs clang-format on the generated code.
        """
        os.makedirs(output_dir, exist_ok=True)
        cpp_code = self.generate(**generate_kwargs)
        cpp_path = os.path.join(output_dir, "model.cpp")

        with open(cpp_path, "w") as f:
            f.write(cpp_code)

        if run_clang_format:
            import subprocess

            try:
                subprocess.run(["clang-format", "-i", cpp_path], check=True, capture_output=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                run_clang_format = False

        cmake_content = f"""cmake_minimum_required(VERSION 3.14)
project({self.class_name}Project)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library({self.class_name} SHARED model.cpp)
set_target_properties({self.class_name} PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable({self.class_name}_cli main.cpp model.cpp)
"""
        with open(os.path.join(output_dir, "CMakeLists.txt"), "w") as f:
            f.write(cmake_content)

        main_content = f"""#include <iostream>
#include <chrono>

extern "C" {{
    void* {self.class_name}_create();
    void {self.class_name}_destroy(void*);
    int {self.class_name}_forward(void*{", void*" * (len(self.graph.inputs) + len(self.graph.outputs))});
}}

int main() {{
    std::cout << "Initializing {self.class_name}..." << std::endl;
    void* model = {self.class_name}_create();
    if (!model) {{
        std::cerr << "Failed to create model." << std::endl;
        return 1;
    }}
    
    // We pass nulls or zeros for benchmarking. In a real CLI these would be parsed or memory mapped.
    auto start = std::chrono::high_resolution_clock::now();
    int res = {self.class_name}_forward(model{", nullptr" * (len(self.graph.inputs) + len(self.graph.outputs))});
    auto end = std::chrono::high_resolution_clock::now();
    
    if (res != 0) std::cerr << "Forward pass failed!" << std::endl;
    else std::cout << "Execution time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;
    
    {self.class_name}_destroy(model);
    return res;
}}
"""
        main_path = os.path.join(output_dir, "main.cpp")
        with open(main_path, "w") as f:
            f.write(main_content)

        return cpp_path

    def get_tensor_name(self, name: str) -> str:
        """Return the safe C++ variable name for a given tensor."""
        return sanitize_name(name)
