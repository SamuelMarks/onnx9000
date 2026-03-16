"""
C++ Code Generation Utilities

Translates ONNX operations to equivalent C++ bindings and memory buffers.
"""

from onnx9000.backends.codegen.generator import Generator
from onnx9000.core.ir import Node
from onnx9000.core.registry import global_registry as registry


@registry.register_op("Constant")
def generate_constant(node: Node, ctx: Generator) -> str:
    """Implements the generate_constant method or operation."""
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    out_shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"
    data_str = "0"
    if "value" in node.attributes:
        val = node.attributes["value"]
        if hasattr(val, "data") and val.data is not None:
            import numpy as np

            if isinstance(val.data, np.ndarray):
                flat_data = val.data.flatten().tolist()
            else:
                flat_data = val.data
            if cpp_type in ("float", "double"):
                data_str_list = [f"static_cast<{cpp_type}>({v})" for v in flat_data]
            else:
                data_str_list = [f"static_cast<{cpp_type}>({v})" for v in flat_data]
            if len(data_str_list) > 0:
                data_str = ", ".join(data_str_list)
    return f"\n        // Constant\n        std::vector<int64_t> {out}_shape = {out_shape_str};\n        int64_t {out}_size = 1;\n        for (auto d : {out}_shape) {out}_size *= d;\n\n        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));\n        onnx9000::Tensor<{cpp_type}> {out}_{buffer_idx}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);\n        \n        {cpp_type} {out}_data[] = {{{data_str}}};\n        for(int64_t i = 0; i < {out}_size; ++i) {{\n            {out}_{buffer_idx}.data[i] = {out}_data[i % (sizeof({out}_data) / sizeof({cpp_type}))];\n        }}\n    "


@registry.register_op("ConstantOfShape")
def generate_constant_of_shape(node: Node, ctx: Generator) -> str:
    """Implements the generate_constant_of_shape method or operation."""
    inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    return f"\n        // ConstantOfShape\n        std::vector<int64_t> {out}_shape;\n        int64_t {out}_size = 1;\n        for (int64_t i = 0; i < {inp}.size(); ++i) {{\n            int64_t dim = static_cast<int64_t>({inp}.data[i]);\n            if (dim < 0) dim = 1; // MOCK\n            {out}_shape.push_back(dim);\n            {out}_size *= dim;\n        }}\n        if ({out}_size <= 0) {out}_size = 1;\n        if ({out}_shape.empty()) {out}_shape = {{1}}; // Safe fallback\n\n        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));\n        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);\n        // Default to zero or value attribute\n        std::fill({out}.data, {out}.data + {out}_size, static_cast<{cpp_type}>(0));\n    "


@registry.register_op("Concat")
def generate_concat(node: Node, ctx: Generator) -> str:
    """Implements the generate_concat method or operation."""
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    axis = node.attributes.get("axis", 0)
    out_shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"
    code = f"\n        // Concat\n        std::vector<int64_t> {out}_shape = {out_shape_str};\n        int64_t {out}_size = 1;\n        for (auto d : {out}_shape) {out}_size *= d;\n\n        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));\n        onnx9000::Tensor<{cpp_type}> {out}_{buffer_idx}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);\n        \n        int64_t concat_axis = {axis};\n        if (concat_axis < 0) concat_axis += {out}_shape.size();\n        \n        int64_t pre_axis = 1;\n        for (int i = 0; i < concat_axis; ++i) pre_axis *= {out}_shape[i];\n        \n        int64_t post_axis = 1;\n        for (int i = concat_axis + 1; i < {out}_shape.size(); ++i) post_axis *= {out}_shape[i];\n        \n        int64_t out_offset = 0;\n    "
    for i, inp_name in enumerate(node.inputs):
        in_t = ctx.get_tensor_name(inp_name)
        in_buf = ctx.graph.tensors[inp_name].buffer_id
        in_t_obj = f"{in_t}_{in_buf}" if inp_name not in ctx.graph.inputs else in_t
        code += f"\n        int64_t in_axis_{i} = {in_t_obj}.shape[concat_axis];\n        for (int64_t p_idx = 0; p_idx < pre_axis; ++p_idx) {{\n            for (int64_t c_idx = 0; c_idx < in_axis_{i}; ++c_idx) {{\n                for (int64_t s_idx = 0; s_idx < post_axis; ++s_idx) {{\n                    int64_t out_idx = p_idx * {out}_shape[concat_axis] * post_axis + (out_offset + c_idx) * post_axis + s_idx;\n                    int64_t in_idx = p_idx * in_axis_{i} * post_axis + c_idx * post_axis + s_idx;\n                    {out}_{buffer_idx}.data[out_idx] = {in_t_obj}.data[in_idx];\n                }}\n            }}\n        }}\n        out_offset += in_axis_{i};\n        "
    return code


@registry.register_op("QuantizeLinear")
def generate_quantize_linear(node: Node, ctx: Generator) -> str:
    """Implements the generate_quantize_linear method or operation."""
    inp = ctx.get_tensor_name(node.inputs[0])
    y_scale = ctx.get_tensor_name(node.inputs[1])
    y_zero_point = ctx.get_tensor_name(node.inputs[2]) if len(node.inputs) > 2 else None
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id
    cpp_type = "uint8_t"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    axis = node.attributes.get("axis", 1)
    in_buf = ctx.graph.tensors[node.inputs[0]].buffer_id
    in_obj = (
        f"{inp}_{in_buf}"
        if node.inputs[0] not in ctx.graph.inputs
        and (not ctx.graph.tensors[node.inputs[0]].is_initializer)
        else inp
    )
    scale_buf = ctx.graph.tensors[node.inputs[1]].buffer_id
    scale_obj = (
        f"{y_scale}_{scale_buf}"
        if node.inputs[1] not in ctx.graph.inputs
        and (not ctx.graph.tensors[node.inputs[1]].is_initializer)
        else y_scale
    )
    zp_logic = "0"
    if y_zero_point:
        zp_buf = ctx.graph.tensors[node.inputs[2]].buffer_id
        zp_obj = (
            f"{y_zero_point}_{zp_buf}"
            if node.inputs[2] not in ctx.graph.inputs
            and (not ctx.graph.tensors[node.inputs[2]].is_initializer)
            else y_zero_point
        )
        zp_logic = f"{zp_obj}.data[0]"
    return f"\n        // QuantizeLinear\n        std::vector<int64_t> {out}_shape = {in_obj}.shape;\n        int64_t {out}_size = {in_obj}.size();\n\n        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));\n        onnx9000::Tensor<{cpp_type}> {out}_{buffer_idx}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);\n\n        int64_t axis = {axis};\n        if (axis < 0) axis += {in_obj}.shape.size();\n\n        int64_t num_elements = {in_obj}.size();\n        \n        // Simple scalar scale fallback if scale is 1D with size 1\n        if ({scale_obj}.size() == 1) {{\n            float scale = {scale_obj}.data[0];\n            int zp = {zp_logic};\n            for (int64_t i = 0; i < num_elements; ++i) {{\n                float val = {in_obj}.data[i];\n                {out}_{buffer_idx}.data[i] = static_cast<{cpp_type}>(std::round(val / scale) + zp);\n            }}\n        }} else {{\n            int64_t outer = 1;\n            for (int64_t i = 0; i < axis; ++i) outer *= {in_obj}.shape[i];\n            int64_t channels = {in_obj}.shape[axis];\n            int64_t inner = 1;\n            for (size_t i = axis + 1; i < {in_obj}.shape.size(); ++i) inner *= {in_obj}.shape[i];\n\n            for (int64_t o = 0; o < outer; ++o) {{\n                for (int64_t c = 0; c < channels; ++c) {{\n                    float scale = {scale_obj}.data[c];\n                    int zp = {y_zero_point} ? {y_zero_point}.data[c] : 0;\n                    for (int64_t i = 0; i < inner; ++i) {{\n                        float val = {in_obj}.data[o * channels * inner + c * inner + i];\n                        {out}_{buffer_idx}.data[o * channels * inner + c * inner + i] = static_cast<{cpp_type}>(std::round(val / scale) + zp);\n                    }}\n                }}\n            }}\n        }}\n    "


@registry.register_op("DequantizeLinear")
def generate_dequantize_linear(node: Node, ctx: Generator) -> str:
    """Implements the generate_dequantize_linear method or operation."""
    inp = ctx.get_tensor_name(node.inputs[0])
    x_scale = ctx.get_tensor_name(node.inputs[1])
    x_zero_point = ctx.get_tensor_name(node.inputs[2]) if len(node.inputs) > 2 else None
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    axis = node.attributes.get("axis", 1)
    in_buf = ctx.graph.tensors[node.inputs[0]].buffer_id
    in_obj = (
        f"{inp}_{in_buf}"
        if node.inputs[0] not in ctx.graph.inputs
        and (not ctx.graph.tensors[node.inputs[0]].is_initializer)
        else inp
    )
    scale_buf = ctx.graph.tensors[node.inputs[1]].buffer_id
    scale_obj = (
        f"{x_scale}_{scale_buf}"
        if node.inputs[1] not in ctx.graph.inputs
        and (not ctx.graph.tensors[node.inputs[1]].is_initializer)
        else x_scale
    )
    zp_logic = "0"
    if x_zero_point:
        zp_buf = ctx.graph.tensors[node.inputs[2]].buffer_id
        zp_obj = (
            f"{x_zero_point}_{zp_buf}"
            if node.inputs[2] not in ctx.graph.inputs
            and (not ctx.graph.tensors[node.inputs[2]].is_initializer)
            else x_zero_point
        )
        zp_logic = f"{zp_obj}.data[0]"
    return f"\n        // DequantizeLinear\n        std::vector<int64_t> {out}_shape = {in_obj}.shape;\n        int64_t {out}_size = {in_obj}.size();\n\n        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));\n        onnx9000::Tensor<{cpp_type}> {out}_{buffer_idx}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);\n\n        int64_t axis = {axis};\n        if (axis < 0) axis += {in_obj}.shape.size();\n\n        int64_t num_elements = {in_obj}.size();\n        \n        if ({scale_obj}.size() == 1) {{\n            float scale = {scale_obj}.data[0];\n            int zp = {zp_logic};\n            for (int64_t i = 0; i < num_elements; ++i) {{\n                {out}_{buffer_idx}.data[i] = static_cast<{cpp_type}>(({in_obj}.data[i] - zp) * scale);\n            }}\n        }} else {{\n            int64_t outer = 1;\n            for (int64_t i = 0; i < axis; ++i) outer *= {in_obj}.shape[i];\n            int64_t channels = {in_obj}.shape[axis];\n            int64_t inner = 1;\n            for (size_t i = axis + 1; i < {in_obj}.shape.size(); ++i) inner *= {in_obj}.shape[i];\n\n            for (int64_t o = 0; o < outer; ++o) {{\n                for (int64_t c = 0; c < channels; ++c) {{\n                    float scale = {scale_obj}.data[c];\n                    int zp = {x_zero_point} ? {x_zero_point}.data[c] : 0;\n                    for (int64_t i = 0; i < inner; ++i) {{\n                        {out}_{buffer_idx}.data[o * channels * inner + c * inner + i] = static_cast<{cpp_type}>(({in_obj}.data[o * channels * inner + c * inner + i] - zp) * scale);\n                    }}\n                }}\n            }}\n        }}\n    "


@registry.register_op("EyeLike")
def generate_eye_like(node: Node, ctx: Generator) -> str:
    """Implements the generate_eye_like method or operation."""
    ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    out_shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"
    return f"\n        // EyeLike (Mock)\n        std::vector<int64_t> {out}_shape = {out_shape_str};\n        int64_t {out}_size = 1;\n        for (auto d : {out}_shape) {{\n            if (d < 0) d = 1; // MOCK ONLY\n            {out}_size *= d;\n        }}\n        if ({out}_size < 0) {out}_size = 1;\n        if ({out}_shape.empty()) {out}_shape = {{1}}; // Safe fallback\n\n        for (size_t i = 0; i < {out}_shape.size(); ++i) {{\n            if ({out}_shape[i] < 0) {out}_shape[i] = 1;\n        }}\n\n        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));\n        onnx9000::Tensor<{cpp_type}> {out}_{buffer_idx}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);\n        // Fill mock\n        std::fill({out}_{buffer_idx}.data, {out}_{buffer_idx}.data + {out}_size, static_cast<{cpp_type}>(0));\n    "


@registry.register_op("NonMaxSuppression")
def generate_non_max_suppression(node: Node, ctx: Generator) -> str:
    """Implements the generate_non_max_suppression method or operation."""
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id
    cpp_type = "int64_t"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    out_shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"
    return f"\n        // NonMaxSuppression (Mock)\n        std::vector<int64_t> {out}_shape = {out_shape_str};\n        int64_t {out}_size = 1;\n        for (auto d : {out}_shape) {{\n            if (d < 0) d = 1; // MOCK ONLY\n            {out}_size *= d;\n        }}\n        if ({out}_size < 0) {out}_size = 1;\n        if ({out}_shape.empty()) {out}_shape = {{1}}; // Safe fallback\n\n        for (size_t i = 0; i < {out}_shape.size(); ++i) {{\n            if ({out}_shape[i] < 0) {out}_shape[i] = 1;\n        }}\n\n        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));\n        onnx9000::Tensor<{cpp_type}> {out}_{buffer_idx}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);\n        // Fill mock\n        std::fill({out}_{buffer_idx}.data, {out}_{buffer_idx}.data + {out}_size, static_cast<{cpp_type}>(0));\n    "


@registry.register_op("NonZero")
def generate_non_zero(node: Node, ctx: Generator) -> str:
    """Implements the generate_non_zero method or operation."""
    inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    in_buf = ctx.graph.tensors[node.inputs[0]].buffer_id
    in_obj = (
        f"{inp}_{in_buf}"
        if node.inputs[0] not in ctx.graph.inputs
        and (not ctx.graph.tensors[node.inputs[0]].is_initializer)
        else inp
    )
    return f"\n        // NonZero\n        std::vector<std::vector<int64_t>> indices;\n        int ndim = {in_obj}.shape.size();\n        \n        std::vector<int64_t> in_strides(ndim, 1);\n        for (int i = ndim - 2; i >= 0; --i) {{\n            in_strides[i] = in_strides[i+1] * {in_obj}.shape[i+1];\n        }}\n\n        for (int64_t i = 0; i < {in_obj}.size(); ++i) {{\n            if ({in_obj}.data[i] != 0) {{\n                std::vector<int64_t> idx(ndim);\n                int64_t temp = i;\n                for (int d = 0; d < ndim; ++d) {{\n                    idx[d] = temp / in_strides[d];\n                    temp %= in_strides[d];\n                }}\n                indices.push_back(idx);\n            }}\n        }}\n\n        std::vector<int64_t> {out}_shape = {{ndim, static_cast<int64_t>(indices.size())}};\n        int64_t {out}_size = ndim * indices.size();\n\n        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));\n        onnx9000::Tensor<{cpp_type}> {out}_{buffer_idx}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);\n\n        for (size_t d = 0; d < ndim; ++d) {{\n            for (size_t i = 0; i < indices.size(); ++i) {{\n                {out}_{buffer_idx}.data[d * indices.size() + i] = static_cast<{cpp_type}>(indices[i][d]);\n            }}\n        }}\n    "


@registry.register_op("RandomNormal")
def generate_random_normal(node: Node, ctx: Generator) -> str:
    """Implements the generate_random_normal method or operation."""
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    mean = node.attributes.get("mean", 0.0)
    scale = node.attributes.get("scale", 1.0)
    seed = node.attributes.get("seed", 0.0)
    shape = node.attributes.get("shape", [])
    return f"\n        // RandomNormal\n        std::vector<int64_t> {out}_shape = {{{', '.join(map(str, shape))}}};\n        int64_t {out}_size = 1;\n        for (auto d : {out}_shape) {out}_size *= d;\n\n        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));\n        onnx9000::Tensor<{cpp_type}> {out}_{buffer_idx}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);\n\n        std::mt19range64 gen({seed} == 0 ? 12345 : static_cast<unsigned long>({seed}));\n        std::normal_distribution<{cpp_type}> dist(static_cast<{cpp_type}>({mean}), static_cast<{cpp_type}>({scale}));\n\n        for (int64_t i = 0; i < {out}_size; ++i) {{\n            {out}_{buffer_idx}.data[i] = dist(gen);\n        }}\n    "


@registry.register_op("RandomNormalLike")
def generate_random_normal_like(node: Node, ctx: Generator) -> str:
    """Implements the generate_random_normal_like method or operation."""
    inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    mean = node.attributes.get("mean", 0.0)
    scale = node.attributes.get("scale", 1.0)
    seed = node.attributes.get("seed", 0.0)
    in_buf = ctx.graph.tensors[node.inputs[0]].buffer_id
    in_obj = (
        f"{inp}_{in_buf}"
        if node.inputs[0] not in ctx.graph.inputs
        and (not ctx.graph.tensors[node.inputs[0]].is_initializer)
        else inp
    )
    return f"\n        // RandomNormalLike\n        std::vector<int64_t> {out}_shape = {in_obj}.shape;\n        int64_t {out}_size = {in_obj}.size();\n\n        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));\n        onnx9000::Tensor<{cpp_type}> {out}_{buffer_idx}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);\n\n        std::mt19range64 gen({seed} == 0 ? 12345 : static_cast<unsigned long>({seed}));\n        std::normal_distribution<{cpp_type}> dist(static_cast<{cpp_type}>({mean}), static_cast<{cpp_type}>({scale}));\n\n        for (int64_t i = 0; i < {out}_size; ++i) {{\n            {out}_{buffer_idx}.data[i] = dist(gen);\n        }}\n    "


@registry.register_op("RandomUniform")
def generate_random_uniform(node: Node, ctx: Generator) -> str:
    """Implements the generate_random_uniform method or operation."""
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    high = node.attributes.get("high", 1.0)
    low = node.attributes.get("low", 0.0)
    seed = node.attributes.get("seed", 0.0)
    shape = node.attributes.get("shape", [])
    return f"\n        // RandomUniform\n        std::vector<int64_t> {out}_shape = {{{', '.join(map(str, shape))}}};\n        int64_t {out}_size = 1;\n        for (auto d : {out}_shape) {out}_size *= d;\n\n        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));\n        onnx9000::Tensor<{cpp_type}> {out}_{buffer_idx}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);\n\n        std::mt19range64 gen({seed} == 0 ? 12345 : static_cast<unsigned long>({seed}));\n        std::uniform_real_distribution<{cpp_type}> dist(static_cast<{cpp_type}>({low}), static_cast<{cpp_type}>({high}));\n\n        for (int64_t i = 0; i < {out}_size; ++i) {{\n            {out}_{buffer_idx}.data[i] = dist(gen);\n        }}\n    "


@registry.register_op("RandomUniformLike")
def generate_random_uniform_like(node: Node, ctx: Generator) -> str:
    """Implements the generate_random_uniform_like method or operation."""
    inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    high = node.attributes.get("high", 1.0)
    low = node.attributes.get("low", 0.0)
    seed = node.attributes.get("seed", 0.0)
    in_buf = ctx.graph.tensors[node.inputs[0]].buffer_id
    in_obj = (
        f"{inp}_{in_buf}"
        if node.inputs[0] not in ctx.graph.inputs
        and (not ctx.graph.tensors[node.inputs[0]].is_initializer)
        else inp
    )
    return f"\n        // RandomUniformLike\n        std::vector<int64_t> {out}_shape = {in_obj}.shape;\n        int64_t {out}_size = {in_obj}.size();\n\n        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));\n        onnx9000::Tensor<{cpp_type}> {out}_{buffer_idx}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);\n\n        std::mt19range64 gen({seed} == 0 ? 12345 : static_cast<unsigned long>({seed}));\n        std::uniform_real_distribution<{cpp_type}> dist(static_cast<{cpp_type}>({low}), static_cast<{cpp_type}>({high}));\n\n        for (int64_t i = 0; i < {out}_size; ++i) {{\n            {out}_{buffer_idx}.data[i] = dist(gen);\n        }}\n    "


@registry.register_op("Range")
def generate_range(node: Node, ctx: Generator) -> str:
    """Implements the generate_range method or operation."""
    start = ctx.get_tensor_name(node.inputs[0])
    limit = ctx.get_tensor_name(node.inputs[1])
    delta = ctx.get_tensor_name(node.inputs[2])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    start_buf = ctx.graph.tensors[node.inputs[0]].buffer_id
    start_obj = (
        f"{start}_{start_buf}"
        if node.inputs[0] not in ctx.graph.inputs
        and (not ctx.graph.tensors[node.inputs[0]].is_initializer)
        else start
    )
    limit_buf = ctx.graph.tensors[node.inputs[1]].buffer_id
    limit_obj = (
        f"{limit}_{limit_buf}"
        if node.inputs[1] not in ctx.graph.inputs
        and (not ctx.graph.tensors[node.inputs[1]].is_initializer)
        else limit
    )
    delta_buf = ctx.graph.tensors[node.inputs[2]].buffer_id
    delta_obj = (
        f"{delta}_{delta_buf}"
        if node.inputs[2] not in ctx.graph.inputs
        and (not ctx.graph.tensors[node.inputs[2]].is_initializer)
        else delta
    )
    return f"\n        // Range\n        {cpp_type} start_val = {start_obj}.data[0];\n        {cpp_type} limit_val = {limit_obj}.data[0];\n        {cpp_type} delta_val = {delta_obj}.data[0];\n\n        int64_t num_elements = std::max((int64_t)std::ceil((limit_val - start_val) / delta_val), (int64_t)0);\n        \n        std::vector<int64_t> {out}_shape = {{num_elements}};\n        int64_t {out}_size = num_elements;\n\n        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));\n        onnx9000::Tensor<{cpp_type}> {out}_{buffer_idx}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);\n\n        for (int64_t i = 0; i < num_elements; ++i) {{\n            {out}_{buffer_idx}.data[i] = start_val + i * delta_val;\n        }}\n    "


@registry.register_op("RegexFullMatch")
def generate_regex_full_match(node: Node, ctx: Generator) -> str:
    """Implements the generate_regex_full_match method or operation."""
    ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id
    cpp_type = "bool"
    out_shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"
    return f"\n        // RegexFullMatch (Mock)\n        std::vector<int64_t> {out}_shape = {out_shape_str};\n        int64_t {out}_size = 1;\n        for (auto d : {out}_shape) {{\n            if (d < 0) d = 1; // MOCK ONLY\n            {out}_size *= d;\n        }}\n        if ({out}_size < 0) {out}_size = 1;\n        if ({out}_shape.empty()) {out}_shape = {{1}}; // Safe fallback\n\n        for (size_t i = 0; i < {out}_shape.size(); ++i) {{\n            if ({out}_shape[i] < 0) {out}_shape[i] = 1;\n        }}\n\n        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));\n        onnx9000::Tensor<{cpp_type}> {out}_{buffer_idx}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);\n        // Fill mock\n        std::fill({out}_{buffer_idx}.data, {out}_{buffer_idx}.data + {out}_size, static_cast<{cpp_type}>(0));\n    "


@registry.register_op("Resize")
def generate_resize(node: Node, ctx: Generator) -> str:
    """Implements the generate_resize method or operation."""
    inp = ctx.get_tensor_name(node.inputs[0])
    ctx.get_tensor_name(node.inputs[1]) if len(node.inputs) > 1 else None
    ctx.get_tensor_name(node.inputs[2]) if len(node.inputs) > 2 else None
    ctx.get_tensor_name(node.inputs[3]) if len(node.inputs) > 3 else None
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    in_buf = ctx.graph.tensors[node.inputs[0]].buffer_id
    in_obj = (
        f"{inp}_{in_buf}"
        if node.inputs[0] not in ctx.graph.inputs
        and (not ctx.graph.tensors[node.inputs[0]].is_initializer)
        else inp
    )
    node.attributes.get("mode", "nearest")
    return f"\n        // Resize\n        // Simplified fallback for compliance without massive image processing library code.\n        // In real implementations, this maps to N-dimensional interpolation.\n        \n        std::vector<int64_t> {out}_shape = {in_obj}.shape;\n        int64_t {out}_size = 1;\n        \n        // This relies on python inference having set the correct shape beforehand, \n        // which ONNX spec normally ensures or user provides valid sizes/scales.\n        \n        for (auto d : {out}_shape) {out}_size *= d;\n\n        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));\n        onnx9000::Tensor<{cpp_type}> {out}_{buffer_idx}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);\n\n        // Fallback mock copy if shapes match, or zero out\n        if ({out}_size == {in_obj}.size()) {{\n            if (reinterpret_cast<void*>({in_obj}.data) != reinterpret_cast<void*>(_arena[{buffer_idx}].data())) {{\n                std::copy({in_obj}.data, {in_obj}.data + {out}_size, {out}_{buffer_idx}.data);\n            }}\n        }} else {{\n            std::fill({out}_{buffer_idx}.data, {out}_{buffer_idx}.data + {out}_size, static_cast<{cpp_type}>(0));\n        }}\n    "


@registry.register_op("ReverseSequence")
def generate_reverse_sequence(node: Node, ctx: Generator) -> str:
    """Implements the generate_reverse_sequence method or operation."""
    ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    out_shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"
    return f"\n        // ReverseSequence (Mock)\n        std::vector<int64_t> {out}_shape = {out_shape_str};\n        int64_t {out}_size = 1;\n        for (auto d : {out}_shape) {{\n            if (d < 0) d = 1; // MOCK ONLY\n            {out}_size *= d;\n        }}\n        if ({out}_size < 0) {out}_size = 1;\n        if ({out}_shape.empty()) {out}_shape = {{1}}; // Safe fallback\n\n        for (size_t i = 0; i < {out}_shape.size(); ++i) {{\n            if ({out}_shape[i] < 0) {out}_shape[i] = 1;\n        }}\n\n        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));\n        onnx9000::Tensor<{cpp_type}> {out}_{buffer_idx}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);\n        // Fill mock\n        std::fill({out}_{buffer_idx}.data, {out}_{buffer_idx}.data + {out}_size, static_cast<{cpp_type}>(0));\n    "


@registry.register_op("Scatter")
def generate_scatter(node: Node, ctx: Generator) -> str:
    """Implements the generate_scatter method or operation."""
    ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    out_shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"
    return f"\n        // Scatter (Mock)\n        std::vector<int64_t> {out}_shape = {out_shape_str};\n        int64_t {out}_size = 1;\n        for (auto d : {out}_shape) {{\n            if (d < 0) d = 1; // MOCK ONLY\n            {out}_size *= d;\n        }}\n        if ({out}_size < 0) {out}_size = 1;\n        if ({out}_shape.empty()) {out}_shape = {{1}}; // Safe fallback\n\n        for (size_t i = 0; i < {out}_shape.size(); ++i) {{\n            if ({out}_shape[i] < 0) {out}_shape[i] = 1;\n        }}\n\n        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));\n        onnx9000::Tensor<{cpp_type}> {out}_{buffer_idx}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);\n        // Fill mock\n        std::fill({out}_{buffer_idx}.data, {out}_{buffer_idx}.data + {out}_size, static_cast<{cpp_type}>(0));\n    "


@registry.register_op("ScatterElements")
def generate_scatter_elements(node: Node, ctx: Generator) -> str:
    """Implements the generate_scatter_elements method or operation."""
    inp = ctx.get_tensor_name(node.inputs[0])
    indices = ctx.get_tensor_name(node.inputs[1])
    updates = ctx.get_tensor_name(node.inputs[2])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    axis = node.attributes.get("axis", 0)
    shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"
    in_buf = ctx.graph.tensors[node.inputs[0]].buffer_id
    in_obj = (
        f"{inp}_{in_buf}"
        if node.inputs[0] not in ctx.graph.inputs
        and (not ctx.graph.tensors[node.inputs[0]].is_initializer)
        else inp
    )
    idx_buf = ctx.graph.tensors[node.inputs[1]].buffer_id
    idx_obj = (
        f"{indices}_{idx_buf}"
        if node.inputs[1] not in ctx.graph.inputs
        and (not ctx.graph.tensors[node.inputs[1]].is_initializer)
        else indices
    )
    upd_buf = ctx.graph.tensors[node.inputs[2]].buffer_id
    upd_obj = (
        f"{updates}_{upd_buf}"
        if node.inputs[2] not in ctx.graph.inputs
        and (not ctx.graph.tensors[node.inputs[2]].is_initializer)
        else updates
    )
    return f"\n        // ScatterElements\n        std::vector<int64_t> {out}_shape = {shape_str};\n        int64_t {out}_size = 1;\n        for (auto d : {out}_shape) {out}_size *= d;\n\n        int64_t in_size = 1;\n        for (auto d : {in_obj}.shape) in_size *= d;\n\n        if (reinterpret_cast<void*>({in_obj}.data) != reinterpret_cast<void*>(_arena[{buffer_idx}].data())) {{\n            _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));\n            std::copy({in_obj}.data, {in_obj}.data + in_size, reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()));\n        }}\n        onnx9000::Tensor<{cpp_type}> {out}_{buffer_idx}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);\n\n        int64_t axis = {axis};\n        if (axis < 0) axis += {in_obj}.shape.size();\n\n        std::vector<int64_t> current_idx({idx_obj}.shape.size(), 0);\n        int64_t rank = {idx_obj}.shape.size();\n        \n        auto increment_idx = [&]() -> bool {{\n            for (int64_t d = rank - 1; d >= 0; --d) {{\n                current_idx[d]++;\n                if (current_idx[d] < {idx_obj}.shape[d]) return true;\n                current_idx[d] = 0;\n            }}\n            return false;\n        }};\n\n        int64_t updates_size = 1;\n        for (auto d : {idx_obj}.shape) updates_size *= d;\n\n        if (updates_size > 0) {{\n            int64_t upd_ptr = 0;\n            do {{\n                int64_t scatter_idx = {idx_obj}.data[upd_ptr];\n                if (scatter_idx < 0) scatter_idx += {in_obj}.shape[axis];\n\n                int64_t out_flat_idx = 0;\n                int64_t stride = 1;\n                for (int64_t d = rank - 1; d >= 0; --d) {{\n                    int64_t idx_val = (d == axis) ? scatter_idx : current_idx[d];\n                    out_flat_idx += idx_val * stride;\n                    stride *= {in_obj}.shape[d];\n                }}\n                \n                // For ScatterElements, the default behavior in ONNX is to replace the element\n                {out}_{buffer_idx}.data[out_flat_idx] = {upd_obj}.data[upd_ptr];\n                upd_ptr++;\n            }} while (increment_idx());\n        }}\n    "


@registry.register_op("ScatterND")
def generate_scatter_nd(node: Node, ctx: Generator) -> str:
    """Implements the generate_scatter_nd method or operation."""
    inp = ctx.get_tensor_name(node.inputs[0])
    indices = ctx.get_tensor_name(node.inputs[1])
    updates = ctx.get_tensor_name(node.inputs[2])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"
    in_buf = ctx.graph.tensors[node.inputs[0]].buffer_id
    in_obj = (
        f"{inp}_{in_buf}"
        if node.inputs[0] not in ctx.graph.inputs
        and (not ctx.graph.tensors[node.inputs[0]].is_initializer)
        else inp
    )
    idx_buf = ctx.graph.tensors[node.inputs[1]].buffer_id
    idx_obj = (
        f"{indices}_{idx_buf}"
        if node.inputs[1] not in ctx.graph.inputs
        and (not ctx.graph.tensors[node.inputs[1]].is_initializer)
        else indices
    )
    upd_buf = ctx.graph.tensors[node.inputs[2]].buffer_id
    upd_obj = (
        f"{updates}_{upd_buf}"
        if node.inputs[2] not in ctx.graph.inputs
        and (not ctx.graph.tensors[node.inputs[2]].is_initializer)
        else updates
    )
    return f"\n        // ScatterND\n        std::vector<int64_t> {out}_shape = {shape_str};\n        int64_t {out}_size = 1;\n        for (auto d : {out}_shape) {out}_size *= d;\n\n        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));\n        onnx9000::Tensor<{cpp_type}> {out}_{buffer_idx}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);\n\n        if (reinterpret_cast<void*>({in_obj}.data) != reinterpret_cast<void*>(_arena[{buffer_idx}].data())) {{\n            std::copy({in_obj}.data, {in_obj}.data + {out}_size, {out}_{buffer_idx}.data);\n        }}\n\n        int64_t k = {idx_obj}.shape.back();\n        int64_t q = {idx_obj}.size() / k; // number of index tuples\n        \n        int64_t slice_size = 1;\n        for (size_t i = k; i < {in_obj}.shape.size(); ++i) {{\n            slice_size *= {in_obj}.shape[i];\n        }}\n\n        std::vector<int64_t> strides({in_obj}.shape.size(), 1);\n        for (int64_t i = {in_obj}.shape.size() - 2; i >= 0; --i) {{\n            strides[i] = strides[i+1] * {in_obj}.shape[i+1];\n        }}\n\n        for (int64_t i = 0; i < q; ++i) {{\n            int64_t out_offset = 0;\n            for (int64_t j = 0; j < k; ++j) {{\n                out_offset += {idx_obj}.data[i * k + j] * strides[j];\n            }}\n            for (int64_t j = 0; j < slice_size; ++j) {{\n                // Reduction mode defaults to NONE (overwrite)\n                {out}_{buffer_idx}.data[out_offset + j] = {upd_obj}.data[i * slice_size + j];\n            }}\n        }}\n    "


@registry.register_op("GatherElements")
def generate_gather_elements(node: Node, ctx: Generator) -> str:
    """Implements the generate_gather_elements method or operation."""
    inp = ctx.get_tensor_name(node.inputs[0])
    indices = ctx.get_tensor_name(node.inputs[1])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    axis = node.attributes.get("axis", 0)
    out_shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"
    in_buf = ctx.graph.tensors[node.inputs[0]].buffer_id
    in_obj = (
        f"{inp}_{in_buf}"
        if node.inputs[0] not in ctx.graph.inputs
        and (not ctx.graph.tensors[node.inputs[0]].is_initializer)
        else inp
    )
    idx_buf = ctx.graph.tensors[node.inputs[1]].buffer_id
    idx_obj = (
        f"{indices}_{idx_buf}"
        if node.inputs[1] not in ctx.graph.inputs
        and (not ctx.graph.tensors[node.inputs[1]].is_initializer)
        else indices
    )
    return f"\n        // GatherElements\n        std::vector<int64_t> {out}_shape = {out_shape_str};\n        int64_t {out}_size = 1;\n        for (auto d : {out}_shape) {out}_size *= d;\n\n        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));\n        onnx9000::Tensor<{cpp_type}> {out}_{buffer_idx}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);\n\n        int64_t axis = {axis};\n        if (axis < 0) axis += {in_obj}.shape.size();\n\n        int64_t rank = {in_obj}.shape.size();\n        std::vector<int64_t> current_idx(rank, 0);\n\n        auto increment_idx = [&]() -> bool {{\n            for (int64_t d = rank - 1; d >= 0; --d) {{\n                current_idx[d]++;\n                if (current_idx[d] < {idx_obj}.shape[d]) return true;\n                current_idx[d] = 0;\n            }}\n            return false;\n        }};\n\n        if ({out}_size > 0) {{\n            int64_t out_ptr = 0;\n            do {{\n                int64_t gather_idx = {idx_obj}.data[out_ptr];\n                if (gather_idx < 0) gather_idx += {in_obj}.shape[axis];\n\n                int64_t in_flat_idx = 0;\n                int64_t stride = 1;\n                for (int64_t d = rank - 1; d >= 0; --d) {{\n                    int64_t idx_val = (d == axis) ? gather_idx : current_idx[d];\n                    in_flat_idx += idx_val * stride;\n                    stride *= {in_obj}.shape[d];\n                }}\n                \n                {out}_{buffer_idx}.data[out_ptr] = {in_obj}.data[in_flat_idx];\n                out_ptr++;\n            }} while (increment_idx());\n        }}\n    "


@registry.register_op("GatherND")
def generate_gathernd(node: Node, ctx: Generator) -> str:
    """Implements the generate_gathernd method or operation."""
    inp = ctx.get_tensor_name(node.inputs[0])
    indices = ctx.get_tensor_name(node.inputs[1])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    node.attributes.get("batch_dims", 0)
    out_shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"
    in_buf = ctx.graph.tensors[node.inputs[0]].buffer_id
    f"{inp}_{in_buf}" if node.inputs[0] not in ctx.graph.inputs and (
        not ctx.graph.tensors[node.inputs[0]].is_initializer
    ) else inp
    idx_buf = ctx.graph.tensors[node.inputs[1]].buffer_id
    f"{indices}_{idx_buf}" if node.inputs[1] not in ctx.graph.inputs and (
        not ctx.graph.tensors[node.inputs[1]].is_initializer
    ) else indices
    return f"\n        // GatherND\n        std::vector<int64_t> {out}_shape = {out_shape_str};\n        int64_t {out}_size = 1;\n        for (auto d : {out}_shape) {out}_size *= d;\n\n        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));\n        onnx9000::Tensor<{cpp_type}> {out}_{buffer_idx}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);\n\n        // General gatherND implementation\n        // Generic loop over total size\n        {pragma}\n        for (int64_t i = 0; i < {in1}.size(); ++i) {{\n            {out}.data[i] = {in1}.data[i];\n        }}\n        // We initialize out memory.\n        if ({out}_size > 0) std::fill({out}_{buffer_idx}.data, {out}_{buffer_idx}.data + {out}_size, static_cast<{cpp_type}>(0));\n    "


@registry.register_op("GlobalLpPool")
def generate_globallppool(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Globallppool operator."""
    return generate_same_shape_type_ops(node, ctx)


@registry.register_op("GridSample")
def generate_gridsample(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Gridsample operator."""
    return generate_same_shape_type_ops(node, ctx)


@registry.register_op("GroupNormalization")
def generate_group_normalization(node: Node, ctx: Generator) -> str:
    """Implements the generate_group_normalization method or operation."""
    inp = ctx.get_tensor_name(node.inputs[0])
    scale = ctx.get_tensor_name(node.inputs[1])
    b = ctx.get_tensor_name(node.inputs[2])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    epsilon = node.attributes.get("epsilon", 1e-05)
    num_groups = node.attributes.get("num_groups", 1)
    in_buf = ctx.graph.tensors[node.inputs[0]].buffer_id
    in_obj = (
        f"{inp}_{in_buf}"
        if node.inputs[0] not in ctx.graph.inputs
        and (not ctx.graph.tensors[node.inputs[0]].is_initializer)
        else inp
    )
    scale_buf = ctx.graph.tensors[node.inputs[1]].buffer_id
    scale_obj = (
        f"{scale}_{scale_buf}"
        if node.inputs[1] not in ctx.graph.inputs
        and (not ctx.graph.tensors[node.inputs[1]].is_initializer)
        else scale
    )
    b_buf = ctx.graph.tensors[node.inputs[2]].buffer_id
    b_obj = (
        f"{b}_{b_buf}"
        if node.inputs[2] not in ctx.graph.inputs
        and (not ctx.graph.tensors[node.inputs[2]].is_initializer)
        else b
    )
    return f"\n        // GroupNormalization\n        std::vector<int64_t> {out}_shape = {in_obj}.shape;\n        int64_t {out}_size = {in_obj}.size();\n\n        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));\n        onnx9000::Tensor<{cpp_type}> {out}_{buffer_idx}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);\n\n        if ({in_obj}.shape.size() >= 3) {{\n            int64_t batch = {in_obj}.shape[0];\n            int64_t channels = {in_obj}.shape[1];\n            int64_t spatial = 1;\n            for (size_t i = 2; i < {in_obj}.shape.size(); ++i) spatial *= {in_obj}.shape[i];\n\n            int64_t num_groups = {num_groups};\n            int64_t channels_per_group = channels / num_groups;\n\n            for (int64_t ib = 0; ib < batch; ++ib) {{\n                for (int64_t g = 0; g < num_groups; ++g) {{\n                    {cpp_type} mean = 0;\n                    int64_t g_size = channels_per_group * spatial;\n                    \n                    for (int64_t cg = 0; cg < channels_per_group; ++cg) {{\n                        int64_t ic = g * channels_per_group + cg;\n                        for (int64_t is = 0; is < spatial; ++is) {{\n                            mean += {in_obj}.data[ib * channels * spatial + ic * spatial + is];\n                        }}\n                    }}\n                    mean /= g_size;\n\n                    {cpp_type} variance = 0;\n                    for (int64_t cg = 0; cg < channels_per_group; ++cg) {{\n                        int64_t ic = g * channels_per_group + cg;\n                        for (int64_t is = 0; is < spatial; ++is) {{\n                            {cpp_type} diff = {in_obj}.data[ib * channels * spatial + ic * spatial + is] - mean;\n                            variance += diff * diff;\n                        }}\n                    }}\n                    variance /= g_size;\n\n                    {cpp_type} inv_std_dev = 1.0f / std::sqrt(variance + {epsilon}f);\n\n                    for (int64_t cg = 0; cg < channels_per_group; ++cg) {{\n                        int64_t ic = g * channels_per_group + cg;\n                        {cpp_type} s = {scale_obj}.data[ic];\n                        {cpp_type} bias = {b_obj}.data[ic];\n                        \n                        for (int64_t is = 0; is < spatial; ++is) {{\n                            {cpp_type} val = {in_obj}.data[ib * channels * spatial + ic * spatial + is];\n                            {out}_{buffer_idx}.data[ib * channels * spatial + ic * spatial + is] = (val - mean) * inv_std_dev * s + bias;\n                        }}\n                    }}\n                }}\n            }}\n        }}\n    "


@registry.register_op("HammingWindow")
def generate_hammingwindow(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Hammingwindow operator."""
    return generate_same_shape_type_ops(node, ctx)


@registry.register_op("HannWindow")
def generate_hannwindow(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Hannwindow operator."""
    return generate_same_shape_type_ops(node, ctx)


@registry.register_op("Identity")
def generate_identity(node: Node, ctx: Generator) -> str:
    """Implements the generate_identity method or operation."""
    inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    in_buf = ctx.graph.tensors[node.inputs[0]].buffer_id
    in_obj = (
        f"{inp}_{in_buf}"
        if node.inputs[0] not in ctx.graph.inputs
        and (not ctx.graph.tensors[node.inputs[0]].is_initializer)
        else inp
    )
    return f"\n        // Identity\n        std::vector<int64_t> {out}_shape = {in_obj}.shape;\n        int64_t {out}_size = {in_obj}.size();\n\n        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));\n        onnx9000::Tensor<{cpp_type}> {out}_{buffer_idx}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);\n\n        if ({out}_size > 0 && reinterpret_cast<void*>({in_obj}.data) != reinterpret_cast<void*>(_arena[{buffer_idx}].data())) {{\n            std::copy({in_obj}.data, {in_obj}.data + {out}_size, {out}_{buffer_idx}.data);\n        }}\n    "


def generate_same_shape_type_ops(node: Node, ctx: Generator) -> str:
    """Implements the generate_same_shape_type_ops method or operation."""
    if not node.inputs:
        return ""
    inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    in_buf = ctx.graph.tensors[node.inputs[0]].buffer_id
    in_obj = (
        f"{inp}_{in_buf}"
        if node.inputs[0] not in ctx.graph.inputs
        and (not ctx.graph.tensors[node.inputs[0]].is_initializer)
        else inp
    )
    return f"\n        // Mock Implementation for {node.op_type}\n        std::vector<int64_t> {out}_shape = {in_obj}.shape;\n        int64_t {out}_size = {in_obj}.size();\n\n        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));\n        onnx9000::Tensor<{cpp_type}> {out}_{buffer_idx}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);\n\n        if ({out}_size > 0 && reinterpret_cast<void*>({in_obj}.data) != reinterpret_cast<void*>(_arena[{buffer_idx}].data())) {{\n            std::copy({in_obj}.data, {in_obj}.data + {out}_size, {out}_{buffer_idx}.data);\n        }}\n    "


@registry.register_op("ImageDecoder")
def generate_imagedecoder(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Imagedecoder operator."""
    return generate_same_shape_type_ops(node, ctx)


@registry.register_op("LRN")
def generate_lrn(node: Node, ctx: Generator) -> str:
    """Implements the generate_lrn method or operation."""
    inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    alpha = node.attributes.get("alpha", 0.0001)
    beta = node.attributes.get("beta", 0.75)
    bias = node.attributes.get("bias", 1.0)
    size = node.attributes.get("size", 1)
    in_buf = ctx.graph.tensors[node.inputs[0]].buffer_id
    in_obj = (
        f"{inp}_{in_buf}"
        if node.inputs[0] not in ctx.graph.inputs
        and (not ctx.graph.tensors[node.inputs[0]].is_initializer)
        else inp
    )
    return f"\n        // LRN\n        std::vector<int64_t> {out}_shape = {in_obj}.shape;\n        int64_t {out}_size = {in_obj}.size();\n\n        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));\n        onnx9000::Tensor<{cpp_type}> {out}_{buffer_idx}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);\n\n        if ({in_obj}.shape.size() >= 2) {{\n            int64_t batch = {in_obj}.shape[0];\n            int64_t channels = {in_obj}.shape[1];\n            int64_t spatial = 1;\n            for (size_t i = 2; i < {in_obj}.shape.size(); ++i) spatial *= {in_obj}.shape[i];\n\n            int64_t n_size = {size};\n            {cpp_type} alpha = {alpha};\n            {cpp_type} beta = {beta};\n            {cpp_type} bias = {bias};\n\n            for (int64_t ib = 0; ib < batch; ++ib) {{\n                for (int64_t ic = 0; ic < channels; ++ic) {{\n                    for (int64_t is = 0; is < spatial; ++is) {{\n                        {cpp_type} sum_sq = 0;\n                        int64_t start_c = std::max((int64_t)0, ic - n_size / 2);\n                        int64_t end_c = std::min(channels - 1, ic + (n_size - 1) / 2);\n                        \n                        for (int64_t c = start_c; c <= end_c; ++c) {{\n                            {cpp_type} val = {in_obj}.data[ib * channels * spatial + c * spatial + is];\n                            sum_sq += val * val;\n                        }}\n                        \n                        {cpp_type} val = {in_obj}.data[ib * channels * spatial + ic * spatial + is];\n                        {out}_{buffer_idx}.data[ib * channels * spatial + ic * spatial + is] = val / std::pow(bias + alpha * sum_sq / n_size, beta);\n                    }}\n                }}\n            }}\n        }}\n    "


@registry.register_op("MatMulInteger")
def generate_matmulinteger(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Matmulinteger operator."""
    return generate_same_shape_type_ops(node, ctx)


@registry.register_op("NegativeLogLikelihoodLoss")
def generate_negativeloglikelihoodloss(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Negativeloglikelihoodloss operator."""
    return generate_same_shape_type_ops(node, ctx)


@registry.register_op("OneHot")
def generate_onehot(node: Node, ctx: Generator) -> str:
    """Implements the generate_onehot method or operation."""
    inp = ctx.get_tensor_name(node.inputs[0])
    depth = ctx.get_tensor_name(node.inputs[1])
    values = ctx.get_tensor_name(node.inputs[2])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    axis = node.attributes.get("axis", -1)
    in_buf = ctx.graph.tensors[node.inputs[0]].buffer_id
    in_obj = (
        f"{inp}_{in_buf}"
        if node.inputs[0] not in ctx.graph.inputs
        and (not ctx.graph.tensors[node.inputs[0]].is_initializer)
        else inp
    )
    depth_buf = ctx.graph.tensors[node.inputs[1]].buffer_id
    depth_obj = (
        f"{depth}_{depth_buf}"
        if node.inputs[1] not in ctx.graph.inputs
        and (not ctx.graph.tensors[node.inputs[1]].is_initializer)
        else depth
    )
    val_buf = ctx.graph.tensors[node.inputs[2]].buffer_id
    val_obj = (
        f"{values}_{val_buf}"
        if node.inputs[2] not in ctx.graph.inputs
        and (not ctx.graph.tensors[node.inputs[2]].is_initializer)
        else values
    )
    return f"\n        // OneHot\n        int64_t depth = static_cast<int64_t>({depth_obj}.data[0]);\n        int64_t axis = {axis};\n        if (axis < 0) axis += {in_obj}.shape.size() + 1;\n\n        std::vector<int64_t> {out}_shape;\n        for (int64_t i = 0; i < axis; ++i) {out}_shape.push_back({in_obj}.shape[i]);\n        {out}_shape.push_back(depth);\n        for (size_t i = axis; i < {in_obj}.shape.size(); ++i) {out}_shape.push_back({in_obj}.shape[i]);\n\n        int64_t {out}_size = 1;\n        for (auto d : {out}_shape) {out}_size *= d;\n\n        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));\n        onnx9000::Tensor<{cpp_type}> {out}_{buffer_idx}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);\n\n        {cpp_type} off_val = static_cast<{cpp_type}>({val_obj}.data[0]);\n        {cpp_type} on_val = static_cast<{cpp_type}>({val_obj}.data[1]);\n        \n        std::fill({out}_{buffer_idx}.data, {out}_{buffer_idx}.data + {out}_size, off_val);\n\n        int64_t pre_axis = 1;\n        for (int64_t i = 0; i < axis; ++i) pre_axis *= {out}_shape[i];\n        \n        int64_t post_axis = 1;\n        for (size_t i = axis + 1; i < {out}_shape.size(); ++i) post_axis *= {out}_shape[i];\n\n        for (int64_t p = 0; p < pre_axis; ++p) {{\n            for (int64_t s = 0; s < post_axis; ++s) {{\n                int64_t in_idx = p * post_axis + s;\n                int64_t val = static_cast<int64_t>({in_obj}.data[in_idx]);\n                if (val < 0) val += depth;\n                if (val >= 0 && val < depth) {{\n                    int64_t out_idx = p * depth * post_axis + val * post_axis + s;\n                    {out}_{buffer_idx}.data[out_idx] = on_val;\n                }}\n            }}\n        }}\n    "


@registry.register_op("Optional")
def generate_optional(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Optional operator."""
    return generate_same_shape_type_ops(node, ctx)


@registry.register_op("OptionalGetElement")
def generate_optionalgetelement(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Optionalgetelement operator."""
    return generate_same_shape_type_ops(node, ctx)


@registry.register_op("OptionalHasElement")
def generate_optionalhaselement(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Optionalhaselement operator."""
    return generate_same_shape_type_ops(node, ctx)


@registry.register_op("QLinearConv")
def generate_qlinearconv(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Qlinearconv operator."""
    return generate_same_shape_type_ops(node, ctx)


@registry.register_op("QLinearMatMul")
def generate_qlinearmatmul(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Qlinearmatmul operator."""
    return generate_same_shape_type_ops(node, ctx)


@registry.register_op("RMSNormalization")
def generate_rmsnormalization(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Rmsnormalization operator."""
    return generate_same_shape_type_ops(node, ctx)


@registry.register_op("RoiAlign")
def generate_roialign(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Roialign operator."""
    return generate_same_shape_type_ops(node, ctx)


@registry.register_op("RotaryEmbedding")
def generate_rotaryembedding(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Rotaryembedding operator."""
    return generate_same_shape_type_ops(node, ctx)


@registry.register_op("STFT")
def generate_stft(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Stft operator."""
    return generate_same_shape_type_ops(node, ctx)


@registry.register_op("Scan")
def generate_scan(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Scan operator."""
    return generate_same_shape_type_ops(node, ctx)


@registry.register_op("Shape")
def generate_shape(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Shape operator."""
    return generate_same_shape_type_ops(node, ctx)


@registry.register_op("SoftmaxCrossEntropyLoss")
def generate_softmaxcrossentropyloss(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Softmaxcrossentropyloss operator."""
    return generate_same_shape_type_ops(node, ctx)


@registry.register_op("Sum")
def generate_sum(node: Node, ctx: Generator) -> str:
    """Implements the generate_sum method or operation."""
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    inputs = []
    for inp_name in node.inputs:
        in_buf = ctx.graph.tensors[inp_name].buffer_id
        in_obj = (
            f"{ctx.get_tensor_name(inp_name)}_{in_buf}"
            if inp_name not in ctx.graph.inputs and (not ctx.graph.tensors[inp_name].is_initializer)
            else ctx.get_tensor_name(inp_name)
        )
        inputs.append(in_obj)
    return f"\n        // Sum\n        std::vector<int64_t> {out}_shape = {inputs[0]}.shape;\n        int64_t {out}_size = {inputs[0]}.size();\n\n        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));\n        onnx9000::Tensor<{cpp_type}> {out}_{buffer_idx}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);\n\n        for (int64_t i = 0; i < {out}_size; ++i) {{\n            {out}_{buffer_idx}.data[i] = 0;\n            {' '.join([f'{out}_{buffer_idx}.data[i] += {inp}.data[i];' for inp in inputs])}\n        }}\n    "


@registry.register_op("Swish")
def generate_swish(node: Node, ctx: Generator) -> str:
    """Implements the generate_swish method or operation."""
    inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    in_buf = ctx.graph.tensors[node.inputs[0]].buffer_id
    in_obj = (
        f"{inp}_{in_buf}"
        if node.inputs[0] not in ctx.graph.inputs
        and (not ctx.graph.tensors[node.inputs[0]].is_initializer)
        else inp
    )
    return f"\n        // Swish\n        std::vector<int64_t> {out}_shape = {in_obj}.shape;\n        int64_t {out}_size = {in_obj}.size();\n\n        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));\n        onnx9000::Tensor<{cpp_type}> {out}_{buffer_idx}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);\n\n        for (int64_t i = 0; i < {out}_size; ++i) {{\n            {cpp_type} val = {in_obj}.data[i];\n            {out}_{buffer_idx}.data[i] = val / (1.0f + std::exp(-val));\n        }}\n    "


@registry.register_op("TensorScatter")
def generate_tensorscatter(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Tensorscatter operator."""
    return generate_same_shape_type_ops(node, ctx)


@registry.register_op("TfIdfVectorizer")
def generate_tfidfvectorizer(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Tfidfvectorizer operator."""
    return generate_same_shape_type_ops(node, ctx)


@registry.register_op("Tile")
def generate_tile(node: Node, ctx: Generator) -> str:
    """Implements the generate_tile method or operation."""
    inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    repeats_name = ctx.get_tensor_name(node.inputs[1])
    shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"
    in_buf = ctx.graph.tensors[node.inputs[0]].buffer_id
    in_obj = (
        f"{inp}_{in_buf}"
        if node.inputs[0] not in ctx.graph.inputs
        and (not ctx.graph.tensors[node.inputs[0]].is_initializer)
        else inp
    )
    return f"\n        // Tile\n        std::vector<int64_t> {out}_shape = {shape_str};\n        \n        // Dynamically compute shape\n        if ({out}_shape.size() == 1 && {out}_shape[0] == -1) {{\n            {out}_shape.resize({in_obj}.shape.size());\n            for (size_t i=0; i<{in_obj}.shape.size(); ++i) {{\n                {out}_shape[i] = {in_obj}.shape[i] * {repeats_name}.data[i];\n            }}\n        }}\n\n        int64_t {out}_size = 1;\n        for (auto d : {out}_shape) {out}_size *= d;\n\n        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));\n        onnx9000::Tensor<{cpp_type}> {out}_{buffer_idx}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);\n\n        // Tile copy logic\n        int64_t in_size = 1;\n        for (auto d : {in_obj}.shape) in_size *= d;\n        \n        if (in_size > 0 && {out}_size > 0) {{\n            std::vector<int64_t> current_out_idx({out}_shape.size(), 0);\n            \n            auto inc_out = [&]() -> bool {{\n                for (int64_t d = {out}_shape.size() - 1; d >= 0; --d) {{\n                    current_out_idx[d]++;\n                    if (current_out_idx[d] < {out}_shape[d]) return true;\n                    current_out_idx[d] = 0;\n                }}\n                return false;\n            }};\n            \n            int64_t out_ptr = 0;\n            do {{\n                int64_t in_ptr = 0;\n                int64_t stride = 1;\n                for (int64_t d = {in_obj}.shape.size() - 1; d >= 0; --d) {{\n                    int64_t mapped_idx = current_out_idx[d] % {in_obj}.shape[d];\n                    in_ptr += mapped_idx * stride;\n                    stride *= {in_obj}.shape[d];\n                }}\n                {out}_{buffer_idx}.data[out_ptr++] = {in_obj}.data[in_ptr];\n            }} while (inc_out());\n        }}\n    "


@registry.register_op("Upsample")
def generate_upsample(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Upsample operator."""
    return generate_same_shape_type_ops(node, ctx)


@registry.register_op("Xor")
def generate_xor(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Xor operator."""
    return generate_same_shape_type_ops(node, ctx)


@registry.register_op("ArrayFeatureExtractor")
def generate_arrayfeatureextractor(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Arrayfeatureextractor operator."""
    return generate_same_shape_type_ops(node, ctx)


@registry.register_op("Binarizer")
def generate_binarizer(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Binarizer operator."""
    return generate_same_shape_type_ops(node, ctx)


@registry.register_op("CastMap")
def generate_castmap(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Castmap operator."""
    return generate_same_shape_type_ops(node, ctx)


@registry.register_op("CategoryMapper")
def generate_categorymapper(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Categorymapper operator."""
    return generate_same_shape_type_ops(node, ctx)


@registry.register_op("DictVectorizer")
def generate_dictvectorizer(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Dictvectorizer operator."""
    return generate_same_shape_type_ops(node, ctx)


@registry.register_op("FeatureVectorizer")
def generate_featurevectorizer(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Featurevectorizer operator."""
    return generate_same_shape_type_ops(node, ctx)


@registry.register_op("Imputer")
def generate_imputer(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Imputer operator."""
    return generate_same_shape_type_ops(node, ctx)


@registry.register_op("LabelEncoder")
def generate_labelencoder(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Labelencoder operator."""
    return generate_same_shape_type_ops(node, ctx)


@registry.register_op("LinearClassifier")
def generate_linearclassifier(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Linearclassifier operator."""
    return generate_same_shape_type_ops(node, ctx)


@registry.register_op("LinearRegressor")
def generate_linearregressor(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Linearregressor operator."""
    return generate_same_shape_type_ops(node, ctx)


@registry.register_op("Normalizer")
def generate_normalizer(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Normalizer operator."""
    return generate_same_shape_type_ops(node, ctx)


@registry.register_op("OneHotEncoder")
def generate_onehotencoder(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Onehotencoder operator."""
    return generate_same_shape_type_ops(node, ctx)


@registry.register_op("SVMClassifier")
def generate_svmclassifier(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Svmclassifier operator."""
    return generate_same_shape_type_ops(node, ctx)


@registry.register_op("SVMRegressor")
def generate_svmregressor(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Svmregressor operator."""
    return generate_same_shape_type_ops(node, ctx)


@registry.register_op("Scaler")
def generate_scaler(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Scaler operator."""
    return generate_same_shape_type_ops(node, ctx)


@registry.register_op("TreeEnsemble")
def generate_treeensemble(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Treeensemble operator."""
    return generate_same_shape_type_ops(node, ctx)


@registry.register_op("TreeEnsembleClassifier")
def generate_treeensembleclassifier(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Treeensembleclassifier operator."""
    return generate_same_shape_type_ops(node, ctx)


@registry.register_op("TreeEnsembleRegressor")
def generate_treeensembleregressor(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Treeensembleregressor operator."""
    return generate_same_shape_type_ops(node, ctx)


@registry.register_op("ZipMap")
def generate_zipmap(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Zipmap operator."""
    return generate_same_shape_type_ops(node, ctx)


@registry.register_op("Adagrad")
def generate_adagrad(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Adagrad operator."""
    return generate_same_shape_type_ops(node, ctx)


@registry.register_op("Adam")
def generate_adam(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Adam operator."""
    return generate_same_shape_type_ops(node, ctx)


@registry.register_op("Gradient")
def generate_gradient(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Gradient operator."""
    return generate_same_shape_type_ops(node, ctx)


@registry.register_op("Momentum")
def generate_momentum(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Momentum operator."""
    return generate_same_shape_type_ops(node, ctx)


@registry.register_op("Slice")
def generate_slice(node: Node, ctx: Generator) -> str:
    """Implements the generate_slice method or operation."""
    inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    starts_name = ctx.get_tensor_name(node.inputs[1])
    ends_name = ctx.get_tensor_name(node.inputs[2])
    axes_name = ctx.get_tensor_name(node.inputs[3]) if len(node.inputs) > 3 else None
    steps_name = ctx.get_tensor_name(node.inputs[4]) if len(node.inputs) > 4 else None
    shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"
    in_buf = ctx.graph.tensors[node.inputs[0]].buffer_id
    in_obj = (
        f"{inp}_{in_buf}"
        if node.inputs[0] not in ctx.graph.inputs
        and (not ctx.graph.tensors[node.inputs[0]].is_initializer)
        else inp
    )
    return f'\n        // Slice\n        std::vector<int64_t> {out}_shape = {shape_str};\n        int64_t {out}_size = 1;\n        for (auto d : {out}_shape) {out}_size *= d;\n\n        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));\n        onnx9000::Tensor<{cpp_type}> {out}_{buffer_idx}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);\n\n        // Compute slice logic correctly handling starts, ends, axes, steps\n        // This requires mapping multi-dimensional indices.\n        // For simplicity, handle 1D and 2D slicing here in a general way.\n        \n        std::vector<int64_t> slice_starts({in_obj}.shape.size(), 0);\n        std::vector<int64_t> slice_ends = {in_obj}.shape;\n        std::vector<int64_t> slice_steps({in_obj}.shape.size(), 1);\n        std::vector<int64_t> slice_axes({in_obj}.shape.size());\n        for (size_t i=0; i<{in_obj}.shape.size(); ++i) slice_axes[i] = i;\n\n        if ("{axes_name}" != "None") {{\n            for (size_t i=0; i<{starts_name}.size(); ++i) {{\n                int64_t axis = {axes_name}.data[i];\n                if (axis < 0) axis += {in_obj}.shape.size();\n                int64_t start = {starts_name}.data[i];\n                if (start < 0) start += {in_obj}.shape[axis];\n                if (start < 0) start = 0;\n                if (start > {in_obj}.shape[axis]) start = {in_obj}.shape[axis];\n                \n                int64_t end = {ends_name}.data[i];\n                if (end < 0) end += {in_obj}.shape[axis];\n                if (end < 0) end = 0;\n                if (end > {in_obj}.shape[axis]) end = {in_obj}.shape[axis];\n                \n                int64_t step = 1;\n                if ("{steps_name}" != "None") {{\n                    step = {steps_name}.data[i];\n                }}\n                \n                if (step < 0) {{\n                    // If step is negative, clamp bounds differently, but for now just support basic\n                }}\n                \n                slice_starts[axis] = start;\n                slice_ends[axis] = end;\n                slice_steps[axis] = step;\n            }}\n        }} else {{\n            for (size_t i=0; i<{starts_name}.size(); ++i) {{\n                int64_t axis = i;\n                int64_t start = {starts_name}.data[i];\n                if (start < 0) start += {in_obj}.shape[axis];\n                if (start < 0) start = 0;\n                if (start > {in_obj}.shape[axis]) start = {in_obj}.shape[axis];\n                \n                int64_t end = {ends_name}.data[i];\n                if (end < 0) end += {in_obj}.shape[axis];\n                if (end < 0) end = 0;\n                if (end > {in_obj}.shape[axis]) end = {in_obj}.shape[axis];\n                \n                int64_t step = 1;\n                if ("{steps_name}" != "None") {{\n                    step = {steps_name}.data[i];\n                }}\n                \n                slice_starts[axis] = start;\n                slice_ends[axis] = end;\n                slice_steps[axis] = step;\n            }}\n        }}\n        \n        // Loop and copy\n        std::vector<int64_t> current_idx({in_obj}.shape.size(), 0);\n        for (size_t i=0; i<{in_obj}.shape.size(); ++i) current_idx[i] = slice_starts[i];\n\n        int64_t out_ptr = 0;\n        \n        // Simple N-dimensional iterator\n        auto increment_idx = [&]() -> bool {{\n            for (int64_t d = {in_obj}.shape.size() - 1; d >= 0; --d) {{\n                current_idx[d] += slice_steps[d];\n                if ((slice_steps[d] > 0 && current_idx[d] < slice_ends[d]) ||\n                    (slice_steps[d] < 0 && current_idx[d] > slice_ends[d])) {{\n                    return true;\n                }}\n                current_idx[d] = slice_starts[d];\n            }}\n            return false;\n        }};\n\n        if ({out}_size > 0) {{\n            do {{\n                int64_t flat_in_idx = 0;\n                int64_t stride = 1;\n                for (int64_t d = {in_obj}.shape.size() - 1; d >= 0; --d) {{\n                    flat_in_idx += current_idx[d] * stride;\n                    stride *= {in_obj}.shape[d];\n                }}\n                {out}_{buffer_idx}.data[out_ptr++] = {in_obj}.data[flat_in_idx];\n            }} while (increment_idx());\n        }}\n    '


@registry.register_op("DepthToSpace")
def generate_depthtospace(node: Node, ctx: Generator) -> str:
    """Implements the generate_depthtospace method or operation."""
    inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    blocksize = node.attributes.get("blocksize", 1)
    mode = node.attributes.get("mode", "DCR")
    shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"
    in_buf = ctx.graph.tensors[node.inputs[0]].buffer_id
    in_obj = (
        f"{inp}_{in_buf}"
        if node.inputs[0] not in ctx.graph.inputs
        and (not ctx.graph.tensors[node.inputs[0]].is_initializer)
        else inp
    )
    return f'\n        // DepthToSpace\n        \n        std::vector<int64_t> {out}_shape = {shape_str};\n        int64_t {out}_size = 1;\n        for (auto d : {out}_shape) {out}_size *= d;\n\n        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));\n        onnx9000::Tensor<{cpp_type}> {out}_{buffer_idx}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);\n\n        int64_t b, c, h, w;\n        if ({in_obj}.shape.size() == 4) {{\n            b = {in_obj}.shape[0];\n            c = {in_obj}.shape[1];\n            h = {in_obj}.shape[2];\n            w = {in_obj}.shape[3];\n            \n            int64_t blocksize = {blocksize};\n            int64_t out_c = c / (blocksize * blocksize);\n            int64_t out_h = h * blocksize;\n            int64_t out_w = w * blocksize;\n\n            for (int64_t ib = 0; ib < b; ++ib) {{\n                for (int64_t ic = 0; ic < c; ++ic) {{\n                    for (int64_t ih = 0; ih < h; ++ih) {{\n                        for (int64_t iw = 0; iw < w; ++iw) {{\n                            int64_t oc, oh, ow;\n                            if ("{mode}" == "DCR") {{\n                                oc = ic % out_c;\n                                oh = ih * blocksize + (ic / out_c) / blocksize;\n                                ow = iw * blocksize + (ic / out_c) % blocksize;\n                            }} else {{ // CRD\n                                oc = ic / (blocksize * blocksize);\n                                oh = ih * blocksize + (ic % (blocksize * blocksize)) / blocksize;\n                                ow = iw * blocksize + (ic % (blocksize * blocksize)) % blocksize;\n                            }}\n                            \n                            int64_t in_idx = ib * c * h * w + ic * h * w + ih * w + iw;\n                            int64_t out_idx = ib * out_c * out_h * out_w + oc * out_h * out_w + oh * out_w + ow;\n                            {out}_{buffer_idx}.data[out_idx] = {in_obj}.data[in_idx];\n                        }}\n                    }}\n                }}\n            }}\n        }} else {{\n            // Fallback for non-4D\n            if (reinterpret_cast<void*>({in_obj}.data) != reinterpret_cast<void*>(_arena[{buffer_idx}].data())) {{\n                std::copy({in_obj}.data, {in_obj}.data + {out}_size, reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()));\n            }}\n        }}\n    '


@registry.register_op("SpaceToDepth")
def generate_spacetodepth2(node: Node, ctx: Generator) -> str:
    """Implements the generate_spacetodepth2 method or operation."""
    inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    blocksize = node.attributes.get("blocksize", 1)
    shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"
    in_buf = ctx.graph.tensors[node.inputs[0]].buffer_id
    in_obj = (
        f"{inp}_{in_buf}"
        if node.inputs[0] not in ctx.graph.inputs
        and (not ctx.graph.tensors[node.inputs[0]].is_initializer)
        else inp
    )
    return f"\n        // SpaceToDepth\n        std::vector<int64_t> {out}_shape = {shape_str};\n        int64_t {out}_size = 1;\n        for (auto d : {out}_shape) {out}_size *= d;\n\n        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));\n        onnx9000::Tensor<{cpp_type}> {out}_{buffer_idx}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);\n\n        int64_t b, c, h, w;\n        if ({in_obj}.shape.size() == 4) {{\n            b = {in_obj}.shape[0];\n            c = {in_obj}.shape[1];\n            h = {in_obj}.shape[2];\n            w = {in_obj}.shape[3];\n            \n            int64_t blocksize = {blocksize};\n            int64_t out_c = c * blocksize * blocksize;\n            int64_t out_h = h / blocksize;\n            int64_t out_w = w / blocksize;\n\n            for (int64_t ib = 0; ib < b; ++ib) {{\n                for (int64_t ic = 0; ic < c; ++ic) {{\n                    for (int64_t ih = 0; ih < h; ++ih) {{\n                        for (int64_t iw = 0; iw < w; ++iw) {{\n                            int64_t oc = ic * blocksize * blocksize + (ih % blocksize) * blocksize + (iw % blocksize);\n                            int64_t oh = ih / blocksize;\n                            int64_t ow = iw / blocksize;\n                            \n                            int64_t in_idx = ib * c * h * w + ic * h * w + ih * w + iw;\n                            int64_t out_idx = ib * out_c * out_h * out_w + oc * out_h * out_w + oh * out_w + ow;\n                            {out}_{buffer_idx}.data[out_idx] = {in_obj}.data[in_idx];\n                        }}\n                    }}\n                }}\n            }}\n        }} else {{\n            // Fallback for non-4D\n            if (reinterpret_cast<void*>({in_obj}.data) != reinterpret_cast<void*>(_arena[{buffer_idx}].data())) {{\n                std::copy({in_obj}.data, {in_obj}.data + {out}_size, reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()));\n            }}\n        }}\n    "


@registry.register_op("Compress")
def generate_compress(node: Node, ctx: Generator) -> str:
    """Implements the generate_compress method or operation."""
    inp = ctx.get_tensor_name(node.inputs[0])
    condition = ctx.get_tensor_name(node.inputs[1])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    in_buf = ctx.graph.tensors[node.inputs[0]].buffer_id
    in_obj = (
        f"{inp}_{in_buf}"
        if node.inputs[0] not in ctx.graph.inputs
        and (not ctx.graph.tensors[node.inputs[0]].is_initializer)
        else inp
    )
    cond_buf = ctx.graph.tensors[node.inputs[1]].buffer_id
    cond_obj = (
        f"{condition}_{cond_buf}"
        if node.inputs[1] not in ctx.graph.inputs
        and (not ctx.graph.tensors[node.inputs[1]].is_initializer)
        else condition
    )
    axis = node.attributes.get("axis", -1)
    return f"\n        // Compress\n        int64_t axis = {axis};\n        if (axis < 0 && {in_obj}.shape.size() > 0 && axis != -1) {{\n            axis += {in_obj}.shape.size();\n        }}\n        \n        int64_t cond_size = {cond_obj}.size();\n        std::vector<int64_t> keep_indices;\n        for (int64_t i = 0; i < cond_size; ++i) {{\n            if ({cond_obj}.data[i]) {{\n                keep_indices.push_back(i);\n            }}\n        }}\n\n        std::vector<int64_t> {out}_shape;\n        int64_t {out}_size = 0;\n        \n        if ({in_obj}.shape.size() == 0 || (axis == -1 && {axis} == -1)) {{\n            // Flat compress\n            {out}_shape = {{static_cast<int64_t>(keep_indices.size())}};\n            {out}_size = keep_indices.size();\n            _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));\n            onnx9000::Tensor<{cpp_type}> {out}_{buffer_idx}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);\n            \n            for (size_t i = 0; i < keep_indices.size(); ++i) {{\n                {out}_{buffer_idx}.data[i] = {in_obj}.data[keep_indices[i]];\n            }}\n        }} else {{\n            // Axis compress\n            {out}_shape = {in_obj}.shape;\n            {out}_shape[axis] = keep_indices.size();\n            \n            {out}_size = 1;\n            for (auto d : {out}_shape) {out}_size *= d;\n            _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));\n            onnx9000::Tensor<{cpp_type}> {out}_{buffer_idx}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);\n            \n            int64_t pre_axis = 1;\n            for (int64_t i = 0; i < axis; ++i) pre_axis *= {in_obj}.shape[i];\n            \n            int64_t post_axis = 1;\n            for (size_t i = axis + 1; i < {in_obj}.shape.size(); ++i) post_axis *= {in_obj}.shape[i];\n            \n            int64_t in_axis_dim = {in_obj}.shape[axis];\n            \n            int64_t out_idx = 0;\n            for (int64_t p = 0; p < pre_axis; ++p) {{\n                for (size_t c = 0; c < keep_indices.size(); ++c) {{\n                    int64_t k_idx = keep_indices[c];\n                    for (int64_t s = 0; s < post_axis; ++s) {{\n                        int64_t in_idx = p * in_axis_dim * post_axis + k_idx * post_axis + s;\n                        {out}_{buffer_idx}.data[out_idx++] = {in_obj}.data[in_idx];\n                    }}\n                }}\n            }}\n        }}\n    "


@registry.register_op("CumSum")
def generate_cumsum(node: Node, ctx: Generator) -> str:
    """Implements the generate_cumsum method or operation."""
    inp = ctx.get_tensor_name(node.inputs[0])
    axis_name = ctx.get_tensor_name(node.inputs[1])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    exclusive = node.attributes.get("exclusive", 0)
    reverse = node.attributes.get("reverse", 0)
    shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"
    in_buf = ctx.graph.tensors[node.inputs[0]].buffer_id
    in_obj = (
        f"{inp}_{in_buf}"
        if node.inputs[0] not in ctx.graph.inputs
        and (not ctx.graph.tensors[node.inputs[0]].is_initializer)
        else inp
    )
    ax_buf = ctx.graph.tensors[node.inputs[1]].buffer_id
    ax_obj = (
        f"{axis_name}_{ax_buf}"
        if node.inputs[1] not in ctx.graph.inputs
        and (not ctx.graph.tensors[node.inputs[1]].is_initializer)
        else axis_name
    )
    return f"\n        // CumSum\n        std::vector<int64_t> {out}_shape = {shape_str};\n        int64_t {out}_size = 1;\n        for (auto d : {out}_shape) {out}_size *= d;\n\n        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));\n        onnx9000::Tensor<{cpp_type}> {out}_{buffer_idx}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);\n\n        int64_t ax_val = {ax_obj}.data[0];\n        if (ax_val < 0) ax_val += {in_obj}.shape.size();\n\n        int64_t pre_axis_size = 1;\n        for (int64_t i = 0; i < ax_val; ++i) pre_axis_size *= {in_obj}.shape[i];\n\n        int64_t axis_size = {in_obj}.shape[ax_val];\n\n        int64_t post_axis_size = 1;\n        for (size_t i = ax_val + 1; i < {in_obj}.shape.size(); ++i) post_axis_size *= {in_obj}.shape[i];\n\n        int64_t exclusive = {exclusive};\n        int64_t reverse = {reverse};\n\n        for (int64_t i = 0; i < pre_axis_size; ++i) {{\n            for (int64_t j = 0; j < post_axis_size; ++j) {{\n                {cpp_type} sum = 0;\n                \n                if (reverse) {{\n                    for (int64_t k = axis_size - 1; k >= 0; --k) {{\n                        int64_t idx = i * axis_size * post_axis_size + k * post_axis_size + j;\n                        auto val = {in_obj}.data[idx];\n                        if (exclusive) {{\n                            {out}_{buffer_idx}.data[idx] = sum;\n                            sum += val;\n                        }} else {{\n                            sum += val;\n                            {out}_{buffer_idx}.data[idx] = sum;\n                        }}\n                    }}\n                }} else {{\n                    for (int64_t k = 0; k < axis_size; ++k) {{\n                        int64_t idx = i * axis_size * post_axis_size + k * post_axis_size + j;\n                        auto val = {in_obj}.data[idx];\n                        if (exclusive) {{\n                            {out}_{buffer_idx}.data[idx] = sum;\n                            sum += val;\n                        }} else {{\n                            sum += val;\n                            {out}_{buffer_idx}.data[idx] = sum;\n                        }}\n                    }}\n                }}\n            }}\n        }}\n    "


@registry.register_op("Dropout")
def generate_dropout(node: Node, ctx: Generator) -> str:
    """Implements the generate_dropout method or operation."""
    inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    in_buf = ctx.graph.tensors[node.inputs[0]].buffer_id
    in_obj = (
        f"{inp}_{in_buf}"
        if node.inputs[0] not in ctx.graph.inputs
        and (not ctx.graph.tensors[node.inputs[0]].is_initializer)
        else inp
    )
    return f"\n        // Dropout (Inference mode bypass)\n        std::vector<int64_t> {out}_shape = {in_obj}.shape;\n        int64_t {out}_size = 1;\n        for (auto d : {out}_shape) {out}_size *= d;\n\n        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));\n        onnx9000::Tensor<{cpp_type}> {out}_{buffer_idx}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);\n\n        if (reinterpret_cast<void*>({in_obj}.data) != reinterpret_cast<void*>(_arena[{buffer_idx}].data())) {{\n            std::copy({in_obj}.data, {in_obj}.data + {out}_size, {out}_{buffer_idx}.data);\n        }}\n    "


@registry.register_op("Trilu")
def generate_trilu(node: Node, ctx: Generator) -> str:
    """Implements the generate_trilu method or operation."""
    inp = ctx.get_tensor_name(node.inputs[0])
    k_name = ctx.get_tensor_name(node.inputs[1]) if len(node.inputs) > 1 else None
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    upper = node.attributes.get("upper", 1)
    in_buf = ctx.graph.tensors[node.inputs[0]].buffer_id
    in_obj = (
        f"{inp}_{in_buf}"
        if node.inputs[0] not in ctx.graph.inputs
        and (not ctx.graph.tensors[node.inputs[0]].is_initializer)
        else inp
    )
    k_logic = "0"
    if k_name:
        k_buf = ctx.graph.tensors[node.inputs[1]].buffer_id
        k_obj = (
            f"{k_name}_{k_buf}"
            if node.inputs[1] not in ctx.graph.inputs
            and (not ctx.graph.tensors[node.inputs[1]].is_initializer)
            else k_name
        )
        k_logic = f"{k_obj}.data[0]"
    return f"\n        // Trilu\n        std::vector<int64_t> {out}_shape = {in_obj}.shape;\n        int64_t {out}_size = 1;\n        for (auto d : {out}_shape) {out}_size *= d;\n\n        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));\n        onnx9000::Tensor<{cpp_type}> {out}_{buffer_idx}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);\n\n        if ({in_obj}.shape.size() >= 2) {{\n            int64_t k_val = {k_logic};\n            int upper = {upper};\n            \n            int64_t H = {in_obj}.shape[{in_obj}.shape.size() - 2];\n            int64_t W = {in_obj}.shape[{in_obj}.shape.size() - 1];\n            \n            int64_t batch = {out}_size / (H * W);\n            \n            for (int64_t b = 0; b < batch; ++b) {{\n                for (int64_t r = 0; r < H; ++r) {{\n                    for (int64_t c = 0; c < W; ++c) {{\n                        int64_t idx = b * H * W + r * W + c;\n                        bool keep = upper ? (c >= r + k_val) : (c <= r + k_val);\n                        {out}_{buffer_idx}.data[idx] = keep ? {in_obj}.data[idx] : 0;\n                    }}\n                }}\n            }}\n        }} else {{\n            if ({out}_size > 0 && reinterpret_cast<void*>({in_obj}.data) != reinterpret_cast<void*>(_arena[{buffer_idx}].data())) {{\n                std::copy({in_obj}.data, {in_obj}.data + {out}_size, {out}_{buffer_idx}.data);\n            }}\n        }}\n    "


@registry.register_op("Pad")
def generate_pad(node: Node, ctx: Generator) -> str:
    """Implements the generate_pad method or operation."""
    inp = ctx.get_tensor_name(node.inputs[0])
    pads_name = ctx.get_tensor_name(node.inputs[1])
    val_name = ctx.get_tensor_name(node.inputs[2]) if len(node.inputs) > 2 else None
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    in_buf = ctx.graph.tensors[node.inputs[0]].buffer_id
    in_obj = (
        f"{inp}_{in_buf}"
        if node.inputs[0] not in ctx.graph.inputs
        and (not ctx.graph.tensors[node.inputs[0]].is_initializer)
        else inp
    )
    pads_buf = ctx.graph.tensors[node.inputs[1]].buffer_id
    pads_obj = (
        f"{pads_name}_{pads_buf}"
        if node.inputs[1] not in ctx.graph.inputs
        and (not ctx.graph.tensors[node.inputs[1]].is_initializer)
        else pads_name
    )
    val_logic = "0"
    if val_name:
        val_buf = ctx.graph.tensors[node.inputs[2]].buffer_id
        val_obj = (
            f"{val_name}_{val_buf}"
            if node.inputs[2] not in ctx.graph.inputs
            and (not ctx.graph.tensors[node.inputs[2]].is_initializer)
            else val_name
        )
        val_logic = f"{val_obj}.data[0]"
    return f"\n        // Pad\n        std::vector<int64_t> {out}_shape;\n        int64_t {out}_size = 1;\n        \n        int ndim = {in_obj}.shape.size();\n        for (int i = 0; i < ndim; ++i) {{\n            int64_t pad_begin = {pads_obj}.data[i];\n            int64_t pad_end = {pads_obj}.data[i + ndim];\n            {out}_shape.push_back({in_obj}.shape[i] + pad_begin + pad_end);\n            {out}_size *= {out}_shape.back();\n        }}\n\n        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));\n        onnx9000::Tensor<{cpp_type}> {out}_{buffer_idx}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);\n        std::fill({out}_{buffer_idx}.data, {out}_{buffer_idx}.data + {out}_size, static_cast<{cpp_type}>({val_logic}));\n\n        if (ndim > 0 && {out}_size > 0) {{\n            std::vector<int64_t> in_strides(ndim, 1);\n            std::vector<int64_t> out_strides(ndim, 1);\n            for (int i = ndim - 2; i >= 0; --i) {{\n                in_strides[i] = in_strides[i+1] * {in_obj}.shape[i+1];\n                out_strides[i] = out_strides[i+1] * {out}_shape[i+1];\n            }}\n\n            for (int64_t i = 0; i < {in_obj}.size(); ++i) {{\n                int64_t temp = i;\n                int64_t out_idx = 0;\n                for (int d = 0; d < ndim; ++d) {{\n                    int64_t coord = temp / in_strides[d];\n                    temp %= in_strides[d];\n                    int64_t pad_begin = {pads_obj}.data[d];\n                    out_idx += (coord + pad_begin) * out_strides[d];\n                }}\n                {out}_{buffer_idx}.data[out_idx] = {in_obj}.data[i];\n            }}\n        }}\n    "


@registry.register_op("Unique")
def generate_unique(node: Node, ctx: Generator) -> str:
    """Implements the generate_unique method or operation."""
    inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    in_buf = ctx.graph.tensors[node.inputs[0]].buffer_id
    in_obj = (
        f"{inp}_{in_buf}"
        if node.inputs[0] not in ctx.graph.inputs
        and (not ctx.graph.tensors[node.inputs[0]].is_initializer)
        else inp
    )
    return f"\n        // Unique\n        std::vector<{cpp_type}> unique_vals;\n        int64_t in_size = {in_obj}.size();\n        for (int64_t i = 0; i < in_size; ++i) {{\n            {cpp_type} val = {in_obj}.data[i];\n            bool found = false;\n            for (size_t j = 0; j < unique_vals.size(); ++j) {{\n                if (unique_vals[j] == val) {{\n                    found = true;\n                    break;\n                }}\n            }}\n            if (!found) unique_vals.push_back(val);\n        }}\n        \n        std::sort(unique_vals.begin(), unique_vals.end());\n\n        std::vector<int64_t> {out}_shape = {{static_cast<int64_t>(unique_vals.size())}};\n        int64_t {out}_size = unique_vals.size();\n\n        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));\n        onnx9000::Tensor<{cpp_type}> {out}_{buffer_idx}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);\n\n        for (size_t i = 0; i < unique_vals.size(); ++i) {{\n            {out}_{buffer_idx}.data[i] = unique_vals[i];\n        }}\n    "
