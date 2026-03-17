"""
C++ Code Generation Utilities

Translates ONNX operations to equivalent C++ bindings and memory buffers.
"""

from onnx9000.backends.codegen.utils import sanitize_name
from onnx9000.core.dtypes import to_cpp_type
from onnx9000.core.ir import Graph
from onnx9000.core.registry import global_registry as registry


class Generator:
    """
    Core C++ Code Generator orchestrating the translation of an ir.Graph
    into a bespoke C++ class.
    """

    def __init__(self, graph: Graph, class_name: str = "GeneratedModel") -> None:
        """Initializes the code generator for a given Graph."""
        self.graph = graph
        self.class_name = class_name
        self.code_blocks: list[str] = []
        self.max_buffer_id = -1
        for tensor in self.graph.tensors.values():
            if tensor.buffer_id is not None:
                self.max_buffer_id = max(self.max_buffer_id, tensor.buffer_id)
        self.num_buffers = self.max_buffer_id + 1 if self.max_buffer_id >= 0 else 0

    def _generate_constructor(self) -> str:
        """Generates the C++ constructor accepting initializers."""
        params = []
        inits = []
        for name in self.graph.initializers:
            tensor = self.graph.tensors[name]
            cpp_type = to_cpp_type(tensor.dtype)
            clean_name = sanitize_name(name)
            params.append(f"{cpp_type}* {clean_name}_ptr")
            shape_str = "{" + ", ".join(map(str, tensor.shape)) + "}"
            inits.append(f"{clean_name}({clean_name}_ptr, {shape_str})")
        params_str = ", ".join(params)
        inits_str = " : " + ", ".join(inits) if inits else ""
        return f"    {self.class_name}({params_str}){inits_str} {{\n        // Initialize centralized memory arenas\n        _arena.resize({self.num_buffers});\n    }}"

    def _generate_members(self) -> str:
        """Generates class member variables."""
        members = []
        from onnx9000.core import config

        if config.ONNX9000_USE_CUDA:
            members.append("    std::vector<onnx9000::CudaBuffer> _arena;")
        else:
            members.append("    std::vector<std::vector<uint8_t>> _arena;")
        for name in self.graph.initializers:
            tensor = self.graph.tensors[name]
            cpp_type = to_cpp_type(tensor.dtype)
            clean_name = sanitize_name(name)
            members.append(f"    onnx9000::Tensor<{cpp_type}> {clean_name};")
        return "\n".join(members)

    def _generate_forward_signature(self) -> str:
        """Generates the forward() method signature."""
        params = []
        seen = set()
        for name in self.graph.inputs:
            tensor = self.graph.tensors[name]
            cpp_type = to_cpp_type(tensor.dtype)
            clean_name = sanitize_name(name)
            if clean_name not in seen:
                params.append(f"{cpp_type}* {clean_name}_ptr")
                seen.add(clean_name)
        for i, name in enumerate(self.graph.outputs):
            tensor = self.graph.tensors[name]
            cpp_type = to_cpp_type(tensor.dtype)
            clean_name = sanitize_name(name)
            params.append(f"{cpp_type}* {clean_name}_out_ptr_{i}")
        params_str = ", ".join(params)
        return f"    std::expected<void, std::string> forward({params_str}) noexcept {{"

    def _generate_forward_body(self) -> str:
        """Generates the main execution loop mapping IR nodes to C++ code."""
        body = []
        body.append("        try {")
        for name in self.graph.inputs:
            tensor = self.graph.tensors[name]
            cpp_type = to_cpp_type(tensor.dtype)
            clean_name = sanitize_name(name)
            shape_str = "{" + ", ".join(map(str, tensor.shape)) + "}"
            body.append(
                f"            onnx9000::Tensor<{cpp_type}> {clean_name}({clean_name}_ptr, {shape_str});"
            )
        for node in self.graph.nodes:
            domain = getattr(node, "domain", "")
            op_gen = registry.get_op(node.op_type, domain=domain)
            op_code = op_gen(node, self)
            body.append(op_code)
        for i, name in enumerate(self.graph.outputs):
            tensor = self.graph.tensors[name]
            clean_name = sanitize_name(name)
            cpp_type = to_cpp_type(tensor.dtype)
            body.append(f"            // Mock fill {clean_name}_out_ptr_{i}")
            idx = tensor.buffer_id
            if idx is not None:
                body.append(
                    f"            if ({clean_name}_out_ptr_{i} != nullptr && _arena.size() > {idx}) {{"
                )
                body.append(
                    f"                size_t c_size = _arena[{idx}].size() / sizeof({cpp_type});"
                )
                body.append(
                    f"                if (c_size > 0) std::copy(reinterpret_cast<{cpp_type}*>(_arena[{idx}].data()), reinterpret_cast<{cpp_type}*>(_arena[{idx}].data()) + c_size, {clean_name}_out_ptr_{i});"
                )
                body.append("            }")
            else:
                body.append(
                    f"            if ({clean_name}_out_ptr_{i} != nullptr && {clean_name}_ptr != nullptr) {{"
                )
                body.append(f"                size_t c_size = {clean_name}.size();")
                body.append(
                    f"                std::copy({clean_name}.data, {clean_name}.data + c_size, {clean_name}_out_ptr_{i});"
                )
                body.append("            }")
        body.append("            return {};")
        body.append("        } catch (const std::exception& e) {")
        body.append("            return std::unexpected(std::string(e.what()));")
        body.append("        } catch (...) {")
        body.append('            return std::unexpected(std::string("Unknown C++ error"));')
        body.append("        }")
        body.append("    }")
        return "\n".join(body)

    def _generate_forward_py(self) -> str:
        """Generates a Pybind11 specific forward wrapper that handles NumPy arrays."""
        lines = []
        params = []
        for name in self.graph.inputs:
            tensor = self.graph.tensors[name]
            cpp_type = to_cpp_type(tensor.dtype)
            clean_name = sanitize_name(name)
            params.append(f"pybind11::array_t<{cpp_type}> {clean_name}_arr")
        params_str = ", ".join(params)
        ret_types = []
        for name in self.graph.outputs:
            tensor = self.graph.tensors[name]
            cpp_type = to_cpp_type(tensor.dtype)
            ret_types.append(f"pybind11::array_t<{cpp_type}>")
        ret_type_str = (
            f"std::tuple<{', '.join(ret_types)}>"
            if len(ret_types) > 1
            else ret_types[0]
            if ret_types
            else "void"
        )
        lines.append(f"    {ret_type_str} forward_py({params_str}) {{")
        seen_ins = set()
        for name in self.graph.inputs:
            tensor = self.graph.tensors[name]
            cpp_type = to_cpp_type(tensor.dtype)
            clean_name = sanitize_name(name)
            if clean_name not in seen_ins:
                lines.append(
                    f"        {cpp_type}* {clean_name}_ptr = static_cast<{cpp_type}*>({clean_name}_arr.request().ptr);"
                )
                seen_ins.add(clean_name)
        forward_call_args = []
        for name in self.graph.inputs:
            forward_call_args.append(f"{sanitize_name(name)}_ptr")
        for _ in self.graph.outputs:
            forward_call_args.append("nullptr")
        lines.append("        // Release GIL during pure C++ compute")
        lines.append("        std::expected<void, std::string> res;")
        lines.append("        {")
        lines.append("            pybind11::gil_scoped_release release;")
        lines.append(f"            res = forward({', '.join(forward_call_args)});")
        lines.append("        }")
        lines.append("        if (!res) {")
        lines.append("            throw std::runtime_error(res.error());")
        lines.append("        }")
        out_returns = []
        for i, name in enumerate(self.graph.outputs):
            tensor = self.graph.tensors[name]
            cpp_type = to_cpp_type(tensor.dtype)
            clean_name = sanitize_name(name)
            idx = tensor.buffer_id
            lines.append(f"        size_t c_size_{i} = _arena[{idx}].size() / sizeof({cpp_type});")
            lines.append(
                f"        pybind11::array_t<{cpp_type}> {clean_name}_out_arr_{i}({{ (ssize_t)c_size_{i} }});"
            )
            lines.append(
                f"        {cpp_type}* ptr_{i} = static_cast<{cpp_type}*>({clean_name}_out_arr_{i}.request().ptr);"
            )
            lines.append(
                f"        if (c_size_{i} > 0) std::copy(reinterpret_cast<{cpp_type}*>(_arena[{idx}].data()), reinterpret_cast<{cpp_type}*>(_arena[{idx}].data()) + c_size_{i}, ptr_{i});"
            )
            out_returns.append(f"{clean_name}_out_arr_{i}")
        if len(out_returns) > 1:
            lines.append(f"        return std::make_tuple({', '.join(out_returns)});")
        elif out_returns:
            lines.append(f"        return {out_returns[0]};")
        lines.append("    }")
        return "\n".join(lines)

    def generate(self) -> str:
        """Assembles the complete C++ class."""
        c = [
            f"class {self.class_name} {{",
            "public:",
            self._generate_constructor(),
            self._generate_forward_signature(),
            self._generate_forward_body(),
            self._generate_forward_py(),
            "private:",
            self._generate_members(),
            "};",
        ]
        return "\n".join(c)

    def get_tensor_name(self, name: str) -> str:
        """Returns the safe C++ variable name for a given tensor."""
        return sanitize_name(name)
