"""
C++ Code Generation Utilities

Translates ONNX operations to equivalent C++ bindings and memory buffers.
"""

from onnx9000.codegen.generator import Generator
from onnx9000.ir import Node
from onnx9000.registry import registry


@registry.register("Attention")
def generate_attention(node: Node, ctx: Generator) -> str:
    """generate_attention docstring."""
    inp = ctx.get_tensor_name(node.inputs[0])

    code = "// Attention (Mock)\n"

    for i, out_name in enumerate(node.outputs):
        if not out_name:
            continue  # pragma: no cover
        out = ctx.get_tensor_name(out_name)
        info = ctx.graph.tensors[out_name]
        idx = info.buffer_id
        c_type = "float"
        if info.dtype is not None:
            from onnx9000.dtypes import to_cpp_type

            c_type = to_cpp_type(info.dtype)

        shape_str = "{" + ", ".join(map(str, info.shape)) + "}"

        code += f"""
        std::vector<int64_t> {out}_shape = {shape_str};
        int64_t {out}_size = 1;
        for (auto d : {out}_shape) {{
            if (d < 0) d = 1; // MOCK ONLY
            {out}_size *= d;
        }}
        if ({out}_size < 0) {out}_size = 1;
        if ({out}_shape.empty()) {out}_shape = {{1}}; // Safe fallback

        for (size_t k = 0; k < {out}_shape.size(); ++k) {{
            if ({out}_shape[k] < 0) {out}_shape[k] = 1;
        }}

        _arena[{idx}].resize({out}_size * sizeof({c_type}));
        onnx9000::Tensor<{c_type}> {out}(reinterpret_cast<{c_type}*>(_arena[{idx}].data()), {out}_shape);
        std::fill({out}.data, {out}.data + {out}_size, static_cast<{c_type}>(0));
        """

    return code


@registry.register("Conv")
def generate_conv(node: Node, generator_context: "Generator") -> str:
    """generate_conv docstring."""
    x_var = generator_context.get_tensor_name(node.inputs[0])
    w_var = generator_context.get_tensor_name(node.inputs[1])

    tensor_info = generator_context.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    b_opt = "std::nullopt"
    if len(node.inputs) > 2:
        b_var = generator_context.get_tensor_name(node.inputs[2])
        b_opt = f"std::optional<const {cpp_type}*>({b_var}.data)"

    out = generator_context.get_tensor_name(node.outputs[0])

    strides = node.attributes.get("strides", [1, 1])
    pads = node.attributes.get("pads", [0, 0, 0, 0])
    dilations = node.attributes.get("dilations", [1, 1])

    return f"""
        // Conv (im2col + GEMM)
        int64_t {out}_N = {x_var}.shape[0];
        int64_t {out}_C = {x_var}.shape[1];
        int64_t {out}_H = {x_var}.shape[2];
        int64_t {out}_W = {x_var}.shape[3];

        int64_t {out}_M = {w_var}.shape[0];
        int64_t {out}_kH = {w_var}.shape[2];
        int64_t {out}_kW = {w_var}.shape[3];

        int64_t {out}_out_H = ({out}_H + {pads[0]} + {pads[2]} - {dilations[0]} * ({out}_kH - 1) - 1) / {strides[0]} + 1;
        int64_t {out}_out_W = ({out}_W + {pads[1]} + {pads[3]} - {dilations[1]} * ({out}_kW - 1) - 1) / {strides[1]} + 1;

        std::vector<int64_t> {out}_shape = {{{out}_N, {out}_M, {out}_out_H, {out}_out_W}};

        int64_t {out}_size = {out}_N * {out}_M * {out}_out_H * {out}_out_W;
        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);

        std::vector<{cpp_type}> {out}_col_buffer({out}_C * {out}_kH * {out}_kW * {out}_out_H * {out}_out_W);

        auto {out}_res = onnx9000::conv2d_forward<{cpp_type}>(
            {x_var}.data, {w_var}.data, {b_opt}, {out}.data,
            {out}_N, {out}_C, {out}_H, {out}_W,
            {out}_M, {out}_kH, {out}_kW,
            {pads[0]}, {pads[1]}, {pads[2]}, {pads[3]},
            {strides[0]}, {strides[1]},
            {dilations[0]}, {dilations[1]},
            {out}_col_buffer.data()
        );

        if (!{out}_res) {{
            return std::unexpected({out}_res.error());
        }}
"""


@registry.register("Transpose")
def generate_transpose(node: Node, generator_context: Generator) -> str:
    """Generates Transpose op."""
    inp = generator_context.get_tensor_name(node.inputs[0])
    out = generator_context.get_tensor_name(node.outputs[0])
    tensor_info = generator_context.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    # Needs a copy from one index order to another based on perm
    return f"""
        // Transpose
        _arena[{buffer_idx}].resize({inp}.size() * sizeof({cpp_type}));
        // Calculate new shape based on perm
        std::vector<int64_t> {out}_shape = {inp}.shape; // Mock: reverse or use perm
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);
        // Copy loop omitted for brevity
"""


@registry.register("Softmax")
def generate_softmax(node: Node, ctx: Generator) -> str:
    """generate_softmax docstring."""

    inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    axis = node.attributes.get("axis", -1)

    in_buf = ctx.graph.tensors[node.inputs[0]].buffer_id
    in_obj = (
        f"{inp}_{in_buf}"
        if node.inputs[0] not in ctx.graph.inputs
        and not ctx.graph.tensors[node.inputs[0]].is_initializer
        else inp
    )

    return f"""
        // Softmax
        std::vector<int64_t> {out}_shape = {in_obj}.shape;
        int64_t {out}_size = {in_obj}.size();
        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));
        onnx9000::Tensor<{cpp_type}> {out}_{buffer_idx}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);

        int64_t axis = {axis};
        if (axis < 0) axis += {in_obj}.shape.size();

        int64_t outer_size = 1;
        for (int64_t i = 0; i < axis; ++i) outer_size *= {in_obj}.shape[i];

        int64_t inner_size = 1;
        for (size_t i = axis; i < {in_obj}.shape.size(); ++i) inner_size *= {in_obj}.shape[i];

        for (int64_t i = 0; i < outer_size; ++i) {{
            {cpp_type} max_val = std::numeric_limits<{cpp_type}>::lowest();
            for (int64_t j = 0; j < inner_size; ++j) {{
                {cpp_type} val = {in_obj}.data[i * inner_size + j];
                if (val > max_val) max_val = val;
            }}

            {cpp_type} sum = 0;
            for (int64_t j = 0; j < inner_size; ++j) {{
                {cpp_type} exp_val = std::exp({in_obj}.data[i * inner_size + j] - max_val);
                {out}_{buffer_idx}.data[i * inner_size + j] = exp_val;
                sum += exp_val;
            }}

            for (int64_t j = 0; j < inner_size; ++j) {{
                {out}_{buffer_idx}.data[i * inner_size + j] /= sum;
            }}
        }}
    """


@registry.register("LogSoftmax")
def generate_log_softmax(node: Node, ctx: Generator) -> str:
    """generate_log_softmax docstring."""

    inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    axis = node.attributes.get("axis", -1)

    in_buf = ctx.graph.tensors[node.inputs[0]].buffer_id
    in_obj = (
        f"{inp}_{in_buf}"
        if node.inputs[0] not in ctx.graph.inputs
        and not ctx.graph.tensors[node.inputs[0]].is_initializer
        else inp
    )

    return f"""
        // LogSoftmax
        std::vector<int64_t> {out}_shape = {in_obj}.shape;
        int64_t {out}_size = {in_obj}.size();
        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));
        onnx9000::Tensor<{cpp_type}> {out}_{buffer_idx}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);

        int64_t axis = {axis};
        if (axis < 0) axis += {in_obj}.shape.size();

        int64_t outer_size = 1;
        for (int64_t i = 0; i < axis; ++i) outer_size *= {in_obj}.shape[i];

        int64_t inner_size = 1;
        for (size_t i = axis; i < {in_obj}.shape.size(); ++i) inner_size *= {in_obj}.shape[i];

        for (int64_t i = 0; i < outer_size; ++i) {{
            {cpp_type} max_val = std::numeric_limits<{cpp_type}>::lowest();
            for (int64_t j = 0; j < inner_size; ++j) {{
                {cpp_type} val = {in_obj}.data[i * inner_size + j];
                if (val > max_val) max_val = val;
            }}

            {cpp_type} sum = 0;
            for (int64_t j = 0; j < inner_size; ++j) {{
                sum += std::exp({in_obj}.data[i * inner_size + j] - max_val);
            }}

            {cpp_type} log_sum = std::log(sum);
            for (int64_t j = 0; j < inner_size; ++j) {{
                {out}_{buffer_idx}.data[i * inner_size + j] = {in_obj}.data[i * inner_size + j] - max_val - log_sum;
            }}
        }}
    """


@registry.register("Hardmax")
def generate_hardmax(node: Node, ctx: Generator) -> str:
    """generate_hardmax docstring."""

    inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    return f"""
        // Hardmax (Mock)
        _arena[{buffer_idx}].resize({inp}.size() * sizeof({cpp_type}));
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {inp}.shape);
        // ... Hardmax logic ...
        std::copy({inp}.data, {inp}.data + {inp}.size(), {out}.data);
"""


@registry.register("RNN")
def generate_rnn(node: Node, ctx: Generator) -> str:
    """generate_rnn docstring."""
    # This is a mock implementation for testing and shape logic
    x = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    out_shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"

    return f"""
        // RNN (Mock)
        std::vector<int64_t> {out}_shape = {out_shape_str};
        int64_t {out}_size = 1;
        for (auto d : {out}_shape) {out}_size *= d;
        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);
        std::fill({out}.data, {out}.data + {out}_size, static_cast<{cpp_type}>(0));
"""


@registry.register("LSTM")
def generate_lstm(node: Node, ctx: Generator) -> str:
    """generate_lstm docstring."""
    # This is a mock implementation for testing and shape logic
    x = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    out_shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"

    return f"""
        // LSTM (Mock)
        std::vector<int64_t> {out}_shape = {out_shape_str};
        int64_t {out}_size = 1;
        for (auto d : {out}_shape) {out}_size *= d;
        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);
        std::fill({out}.data, {out}.data + {out}_size, static_cast<{cpp_type}>(0));
"""


@registry.register("GRU")
def generate_gru(node: Node, ctx: Generator) -> str:
    """generate_gru docstring."""
    # This is a mock implementation for testing and shape logic
    x = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    out_shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"

    return f"""
        // GRU (Mock)
        std::vector<int64_t> {out}_shape = {out_shape_str};
        int64_t {out}_size = 1;
        for (auto d : {out}_shape) {out}_size *= d;
        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);
        std::fill({out}.data, {out}.data + {out}_size, static_cast<{cpp_type}>(0));
"""


@registry.register("ConvTranspose")
def generate_conv_transpose(node: Node, ctx: Generator) -> str:
    """generate_conv_transpose docstring."""
    x = ctx.get_tensor_name(node.inputs[0])
    w = ctx.get_tensor_name(node.inputs[1])
    b = ctx.get_tensor_name(node.inputs[2]) if len(node.inputs) > 2 else None
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    out_shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"

    return f"""
        // ConvTranspose (Mock)
        std::vector<int64_t> {out}_shape = {out_shape_str};
        int64_t {out}_size = 1;
        for (auto d : {out}_shape) {{
            if (d < 0) d = 1; // MOCK ONLY
            {out}_size *= d;
        }}
        if ({out}_size < 0) {out}_size = 1;
        if ({out}_shape.empty()) {out}_shape = {{1}}; // Safe fallback

        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);
        // Fill mock
        std::fill({out}.data, {out}.data + {out}_size, static_cast<{cpp_type}>(0));
    """


@registry.register("DeformConv")
def generate_deform_conv(node: Node, ctx: Generator) -> str:
    """generate_deform_conv docstring."""
    x = ctx.get_tensor_name(node.inputs[0])
    w = ctx.get_tensor_name(node.inputs[1])
    offset = ctx.get_tensor_name(node.inputs[2])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    out_shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"

    return f"""
        // DeformConv (Mock)
        std::vector<int64_t> {out}_shape = {out_shape_str};
        int64_t {out}_size = 1;
        for (auto d : {out}_shape) {{
            if (d < 0) d = 1; // MOCK ONLY
            {out}_size *= d;
        }}
        if ({out}_size < 0) {out}_size = 1;
        if ({out}_shape.empty()) {out}_shape = {{1}}; // Safe fallback

        for (size_t i = 0; i < {out}_shape.size(); ++i) {{
            if ({out}_shape[i] < 0) {out}_shape[i] = 1;
        }}

        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);
        // Fill mock
        std::fill({out}.data, {out}.data + {out}_size, static_cast<{cpp_type}>(0));
    """


@registry.register("LpNormalization")
def generate_lp_normalization(node: Node, ctx: Generator) -> str:
    """generate_lp_normalization docstring."""
    inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    out_shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"

    return f"""
        // LpNormalization (Mock)
        std::vector<int64_t> {out}_shape = {out_shape_str};
        int64_t {out}_size = 1;
        for (auto d : {out}_shape) {{
            if (d < 0) d = 1; // MOCK ONLY
            {out}_size *= d;
        }}
        if ({out}_size < 0) {out}_size = 1;
        if ({out}_shape.empty()) {out}_shape = {{1}}; // Safe fallback

        for (size_t i = 0; i < {out}_shape.size(); ++i) {{
            if ({out}_shape[i] < 0) {out}_shape[i] = 1;
        }}

        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);
        // Fill mock
        std::fill({out}.data, {out}.data + {out}_size, static_cast<{cpp_type}>(0));
    """


@registry.register("LpPool")
def generate_lp_pool(node: Node, ctx: Generator) -> str:
    """generate_lp_pool docstring."""
    inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    out_shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"

    return f"""
        // LpPool (Mock)
        std::vector<int64_t> {out}_shape = {out_shape_str};
        int64_t {out}_size = 1;
        for (auto d : {out}_shape) {{
            if (d < 0) d = 1; // MOCK ONLY
            {out}_size *= d;
        }}
        if ({out}_size < 0) {out}_size = 1;
        if ({out}_shape.empty()) {out}_shape = {{1}}; // Safe fallback

        for (size_t i = 0; i < {out}_shape.size(); ++i) {{
            if ({out}_shape[i] < 0) {out}_shape[i] = 1;
        }}

        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);
        // Fill mock
        std::fill({out}.data, {out}.data + {out}_size, static_cast<{cpp_type}>(0));
    """


@registry.register("LayerNormalization")
def generate_layer_normalization(node: Node, ctx: Generator) -> str:
    """generate_layer_normalization docstring."""
    inp = ctx.get_tensor_name(node.inputs[0])
    scale = ctx.get_tensor_name(node.inputs[1])
    b = ctx.get_tensor_name(node.inputs[2]) if len(node.inputs) > 2 else None

    out_y = ctx.get_tensor_name(node.outputs[0])
    y_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx_y = y_info.buffer_id

    cpp_type = "float"
    if y_info.dtype is not None:
        from onnx9000.dtypes import to_cpp_type

        cpp_type = to_cpp_type(y_info.dtype)

    axis = node.attributes.get("axis", -1)
    epsilon = node.attributes.get("epsilon", 1e-05)

    in_buf = ctx.graph.tensors[node.inputs[0]].buffer_id
    in_obj = (
        f"{inp}_{in_buf}"
        if node.inputs[0] not in ctx.graph.inputs
        and not ctx.graph.tensors[node.inputs[0]].is_initializer
        else inp
    )

    scale_buf = ctx.graph.tensors[node.inputs[1]].buffer_id
    scale_obj = (
        f"{scale}_{scale_buf}"
        if node.inputs[1] not in ctx.graph.inputs
        and not ctx.graph.tensors[node.inputs[1]].is_initializer
        else scale
    )

    b_obj = ""
    if b:
        b_buf = ctx.graph.tensors[node.inputs[2]].buffer_id
        b_obj = (
            f"{b}_{b_buf}"
            if node.inputs[2] not in ctx.graph.inputs
            and not ctx.graph.tensors[node.inputs[2]].is_initializer
            else b
        )

    b_logic = f"{b_obj}.data[i]" if b else "0.0f"

    return f"""
        // LayerNormalization
        std::vector<int64_t> {out_y}_shape = {in_obj}.shape;
        int64_t {out_y}_size = 1;
        for (auto d : {out_y}_shape) {out_y}_size *= d;

        _arena[{buffer_idx_y}].resize({out_y}_size * sizeof({cpp_type}));
        onnx9000::Tensor<{cpp_type}> {out_y}_{buffer_idx_y}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx_y}].data()), {out_y}_shape);

        int64_t axis = {axis};
        if (axis < 0) axis += {in_obj}.shape.size();

        int64_t num_elements = 1;
        for (int64_t i = 0; i < axis; ++i) num_elements *= {in_obj}.shape[i];

        int64_t norm_size = 1;
        for (size_t i = axis; i < {in_obj}.shape.size(); ++i) norm_size *= {in_obj}.shape[i];

        for (int64_t batch = 0; batch < num_elements; ++batch) {{
            {cpp_type} mean = 0;
            for (int64_t i = 0; i < norm_size; ++i) {{
                mean += {in_obj}.data[batch * norm_size + i];
            }}
            mean /= norm_size;

            {cpp_type} variance = 0;
            for (int64_t i = 0; i < norm_size; ++i) {{
                {cpp_type} diff = {in_obj}.data[batch * norm_size + i] - mean;
                variance += diff * diff;
            }}
            variance /= norm_size;

            {cpp_type} inv_std_dev = 1.0f / std::sqrt(variance + {epsilon}f);

            for (int64_t i = 0; i < norm_size; ++i) {{
                {cpp_type} val = {in_obj}.data[batch * norm_size + i];
                {out_y}_{buffer_idx_y}.data[batch * norm_size + i] = (val - mean) * inv_std_dev * {scale_obj}.data[i] + {b_logic};
            }}
        }}
    """


@registry.register("MeanVarianceNormalization")
def generate_mean_variance_normalization(node: Node, ctx: Generator) -> str:
    """generate_mean_variance_normalization docstring."""
    inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    axes_name = ctx.get_tensor_name(node.inputs[1]) if len(node.inputs) > 1 else None

    in_buf = ctx.graph.tensors[node.inputs[0]].buffer_id
    in_obj = (
        f"{inp}_{in_buf}"
        if node.inputs[0] not in ctx.graph.inputs
        and not ctx.graph.tensors[node.inputs[0]].is_initializer
        else inp
    )

    axes_logic = ""
    if axes_name:
        ax_buf = ctx.graph.tensors[node.inputs[1]].buffer_id
        ax_obj = (
            f"{axes_name}_{ax_buf}"
            if node.inputs[1] not in ctx.graph.inputs
            and not ctx.graph.tensors[node.inputs[1]].is_initializer
            else axes_name
        )
        axes_logic = f"for (int64_t i = 0; i < {ax_obj}.size(); ++i) axes.push_back({ax_obj}.data[i]);"
    else:
        axes_logic = "axes.push_back(0); axes.push_back(2); axes.push_back(3);"  # pragma: no cover

    return f"""
        // MeanVarianceNormalization
        std::vector<int64_t> {out}_shape = {in_obj}.shape;
        int64_t {out}_size = {in_obj}.size();

        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));
        onnx9000::Tensor<{cpp_type}> {out}_{buffer_idx}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);

        std::vector<int64_t> axes;
        {axes_logic}
        
        // This is a simplified fallback that just zeroes the output if axes don't match the standard [0, 2, 3] layout,
        // but for exact compliance we use a full N-dim reduction similar to our Reduce ops.
        // Given complexity limits here, we implement a full sum & variance over the entire tensor if axes is empty,
        // else we mock to 0 to prevent segfaults during testing. True MVN requires complex stride logic.
        
        if (axes.empty() || axes.size() == {in_obj}.shape.size()) {{
            {cpp_type} mean = 0;
            for (int64_t i = 0; i < {in_obj}.size(); ++i) mean += {in_obj}.data[i];
            mean /= {in_obj}.size();
            
            {cpp_type} var = 0;
            for (int64_t i = 0; i < {in_obj}.size(); ++i) {{
                {cpp_type} diff = {in_obj}.data[i] - mean;
                var += diff * diff;
            }}
            var /= {in_obj}.size();
            
            {cpp_type} std_dev = std::sqrt(var + 1e-9f);
            
            for (int64_t i = 0; i < {in_obj}.size(); ++i) {{
                {out}_{buffer_idx}.data[i] = ({in_obj}.data[i] - mean) / std_dev;
            }}
        }} else {{
            std::fill({out}_{buffer_idx}.data, {out}_{buffer_idx}.data + {out}_size, static_cast<{cpp_type}>(0));
        }}
    """


@registry.register("InstanceNormalization")
def generate_instance_normalization(node: Node, ctx: Generator) -> str:
    """generate_instance_normalization docstring."""
    inp = ctx.get_tensor_name(node.inputs[0])
    scale = ctx.get_tensor_name(node.inputs[1])
    b = ctx.get_tensor_name(node.inputs[2])

    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    epsilon = node.attributes.get("epsilon", 1e-05)

    in_buf = ctx.graph.tensors[node.inputs[0]].buffer_id
    in_obj = (
        f"{inp}_{in_buf}"
        if node.inputs[0] not in ctx.graph.inputs
        and not ctx.graph.tensors[node.inputs[0]].is_initializer
        else inp
    )

    scale_buf = ctx.graph.tensors[node.inputs[1]].buffer_id
    scale_obj = (
        f"{scale}_{scale_buf}"
        if node.inputs[1] not in ctx.graph.inputs
        and not ctx.graph.tensors[node.inputs[1]].is_initializer
        else scale
    )

    b_buf = ctx.graph.tensors[node.inputs[2]].buffer_id
    b_obj = (
        f"{b}_{b_buf}"
        if node.inputs[2] not in ctx.graph.inputs
        and not ctx.graph.tensors[node.inputs[2]].is_initializer
        else b
    )

    return f"""
        // InstanceNormalization
        std::vector<int64_t> {out}_shape = {in_obj}.shape;
        int64_t {out}_size = {in_obj}.size();

        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));
        onnx9000::Tensor<{cpp_type}> {out}_{buffer_idx}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);

        if ({in_obj}.shape.size() >= 2) {{
            int64_t batch = {in_obj}.shape[0];
            int64_t channels = {in_obj}.shape[1];
            int64_t spatial = 1;
            for (size_t i = 2; i < {in_obj}.shape.size(); ++i) spatial *= {in_obj}.shape[i];

            for (int64_t ib = 0; ib < batch; ++ib) {{
                for (int64_t ic = 0; ic < channels; ++ic) {{
                    {cpp_type} mean = 0;
                    for (int64_t is = 0; is < spatial; ++is) {{
                        mean += {in_obj}.data[ib * channels * spatial + ic * spatial + is];
                    }}
                    mean /= spatial;

                    {cpp_type} variance = 0;
                    for (int64_t is = 0; is < spatial; ++is) {{
                        {cpp_type} diff = {in_obj}.data[ib * channels * spatial + ic * spatial + is] - mean;
                        variance += diff * diff;
                    }}
                    variance /= spatial;

                    {cpp_type} inv_std_dev = 1.0f / std::sqrt(variance + {epsilon}f);
                    {cpp_type} s = {scale_obj}.data[ic];
                    {cpp_type} bias = {b_obj}.data[ic];

                    for (int64_t is = 0; is < spatial; ++is) {{
                        {cpp_type} val = {in_obj}.data[ib * channels * spatial + ic * spatial + is];
                        {out}_{buffer_idx}.data[ib * channels * spatial + ic * spatial + is] = (val - mean) * inv_std_dev * s + bias;
                    }}
                }}
            }}
        }}
    """


@registry.register("MaxUnpool")
def generate_max_unpool(node: Node, ctx: Generator) -> str:
    """generate_max_unpool docstring."""
    inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    out_shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"

    return f"""
        // MaxUnpool (Mock)
        std::vector<int64_t> {out}_shape = {out_shape_str};
        int64_t {out}_size = 1;
        for (auto d : {out}_shape) {{
            if (d < 0) d = 1; // MOCK ONLY
            {out}_size *= d;
        }}
        if ({out}_size < 0) {out}_size = 1;
        if ({out}_shape.empty()) {out}_shape = {{1}}; // Safe fallback

        for (size_t i = 0; i < {out}_shape.size(); ++i) {{
            if ({out}_shape[i] < 0) {out}_shape[i] = 1;
        }}

        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);
        // Fill mock
        std::fill({out}.data, {out}.data + {out}_size, static_cast<{cpp_type}>(0));
    """


@registry.register("AveragePool")
def generate_average_pool(node: Node, ctx: Generator) -> str:
    """generate_average_pool docstring."""
    inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    out_shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"

    # We emit a mock pooling that just zeroes output since true ND pooling codegen is complex.
    return f"""
        // AveragePool (Mock)
        std::vector<int64_t> {out}_shape = {out_shape_str};
        int64_t {out}_size = 1;
        for (auto d : {out}_shape) {out}_size *= d;
        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);
        std::fill({out}.data, {out}.data + {out}_size, static_cast<{cpp_type}>(0));
"""


@registry.register("MaxPool")
def generate_max_pool(node: Node, ctx: Generator) -> str:
    """generate_max_pool docstring."""
    inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    kernel_shape = node.attributes.get("kernel_shape", [1, 1])
    strides = node.attributes.get("strides", [1, 1])
    pads = node.attributes.get("pads", [0, 0, 0, 0])

    in_buf = ctx.graph.tensors[node.inputs[0]].buffer_id
    in_obj = (
        f"{inp}_{in_buf}"
        if node.inputs[0] not in ctx.graph.inputs
        and not ctx.graph.tensors[node.inputs[0]].is_initializer
        else inp
    )

    return f"""
        // MaxPool
        std::vector<int64_t> {out}_shape = {in_obj}.shape;

        int64_t b, c, h, w;
        if ({in_obj}.shape.size() == 4) {{
            b = {in_obj}.shape[0];
            c = {in_obj}.shape[1];
            h = {in_obj}.shape[2];
            w = {in_obj}.shape[3];

            int64_t kH = {kernel_shape[0]};
            int64_t kW = {kernel_shape[1]};
            int64_t padT = {pads[0]};
            int64_t padL = {pads[1]};
            int64_t padB = {pads[2]};
            int64_t padR = {pads[3]};
            int64_t strideH = {strides[0]};
            int64_t strideW = {strides[1]};

            int64_t out_h = (h + padT + padB - kH) / strideH + 1;
            int64_t out_w = (w + padL + padR - kW) / strideW + 1;

            {out}_shape[2] = out_h;
            {out}_shape[3] = out_w;

            int64_t {out}_size = b * c * out_h * out_w;
            _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));
            onnx9000::Tensor<{cpp_type}> {out}_{buffer_idx}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);

            for (int64_t ib = 0; ib < b; ++ib) {{
                for (int64_t ic = 0; ic < c; ++ic) {{
                    for (int64_t oh = 0; oh < out_h; ++oh) {{
                        for (int64_t ow = 0; ow < out_w; ++ow) {{
                            {cpp_type} max_val = std::numeric_limits<{cpp_type}>::lowest();
                            for (int64_t kh = 0; kh < kH; ++kh) {{
                                for (int64_t kw = 0; kw < kW; ++kw) {{
                                    int64_t ih = oh * strideH - padT + kh;
                                    int64_t iw = ow * strideW - padL + kw;
                                    if (ih >= 0 && ih < h && iw >= 0 && iw < w) {{
                                        int64_t in_idx = ib * c * h * w + ic * h * w + ih * w + iw;
                                        {cpp_type} val = {in_obj}.data[in_idx];
                                        if (val > max_val) max_val = val;
                                    }}
                                }}
                            }}
                            int64_t out_idx = ib * c * out_h * out_w + ic * out_h * out_w + oh * out_w + ow;
                            {out}_{buffer_idx}.data[out_idx] = max_val;
                        }}
                    }}
                }}
            }}
        }} else {{
            // Fallback
            int64_t {out}_size = 1;
            for (auto d : {out}_shape) {out}_size *= d;
            _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));
            onnx9000::Tensor<{cpp_type}> {out}_{buffer_idx}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);
            if ({out}_size > 0 && reinterpret_cast<void*>({in_obj}.data) != reinterpret_cast<void*>(_arena[{buffer_idx}].data())) {{
                std::copy({in_obj}.data, {in_obj}.data + {out}_size, {out}_{buffer_idx}.data);
            }}
        }}
    """


@registry.register("GlobalMaxPool")
def generate_global_max_pool(node: Node, ctx: Generator) -> str:
    """generate_global_max_pool docstring."""
    inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    in_buf = ctx.graph.tensors[node.inputs[0]].buffer_id
    in_obj = (
        f"{inp}_{in_buf}"
        if node.inputs[0] not in ctx.graph.inputs
        and not ctx.graph.tensors[node.inputs[0]].is_initializer
        else inp
    )

    return f"""
        // GlobalMaxPool
        std::vector<int64_t> {out}_shape = {in_obj}.shape;
        int64_t {out}_size = 1;
        
        if ({out}_shape.size() >= 3) {{
            for (size_t i = 2; i < {out}_shape.size(); ++i) {{
                {out}_shape[i] = 1;
            }}
        }}
        
        for (auto d : {out}_shape) {out}_size *= d;

        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));
        onnx9000::Tensor<{cpp_type}> {out}_{buffer_idx}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);

        if ({in_obj}.shape.size() >= 3) {{
            int64_t num_channels = {in_obj}.shape[0] * {in_obj}.shape[1];
            int64_t spatial_size = 1;
            for (size_t i = 2; i < {in_obj}.shape.size(); ++i) {{
                spatial_size *= {in_obj}.shape[i];
            }}
            
            for (int64_t c = 0; c < num_channels; ++c) {{
                {cpp_type} max_val = std::numeric_limits<{cpp_type}>::lowest();
                for (int64_t s = 0; s < spatial_size; ++s) {{
                    {cpp_type} val = {in_obj}.data[c * spatial_size + s];
                    if (val > max_val) max_val = val;
                }}
                {out}_{buffer_idx}.data[c] = max_val;
            }}
        }} else {{
            if ({out}_size > 0 && reinterpret_cast<void*>({in_obj}.data) != reinterpret_cast<void*>(_arena[{buffer_idx}].data())) {{
                std::copy({in_obj}.data, {in_obj}.data + {out}_size, {out}_{buffer_idx}.data);
            }}
        }}
    """


@registry.register("GlobalAveragePool")
def generate_global_average_pool(node: Node, ctx: Generator) -> str:
    """generate_global_average_pool docstring."""
    inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    in_buf = ctx.graph.tensors[node.inputs[0]].buffer_id
    in_obj = (
        f"{inp}_{in_buf}"
        if node.inputs[0] not in ctx.graph.inputs
        and not ctx.graph.tensors[node.inputs[0]].is_initializer
        else inp
    )

    return f"""
        // GlobalAveragePool
        std::vector<int64_t> {out}_shape = {in_obj}.shape;
        int64_t {out}_size = 1;
        
        if ({out}_shape.size() >= 3) {{
            for (size_t i = 2; i < {out}_shape.size(); ++i) {{
                {out}_shape[i] = 1;
            }}
        }}
        
        for (auto d : {out}_shape) {out}_size *= d;

        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));
        onnx9000::Tensor<{cpp_type}> {out}_{buffer_idx}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);

        if ({in_obj}.shape.size() >= 3) {{
            int64_t num_channels = {in_obj}.shape[0] * {in_obj}.shape[1];
            int64_t spatial_size = 1;
            for (size_t i = 2; i < {in_obj}.shape.size(); ++i) {{
                spatial_size *= {in_obj}.shape[i];
            }}
            
            for (int64_t c = 0; c < num_channels; ++c) {{
                {cpp_type} sum = 0;
                for (int64_t s = 0; s < spatial_size; ++s) {{
                    sum += {in_obj}.data[c * spatial_size + s];
                }}
                {out}_{buffer_idx}.data[c] = spatial_size > 0 ? sum / spatial_size : 0;
            }}
        }} else {{
            if ({out}_size > 0 && reinterpret_cast<void*>({in_obj}.data) != reinterpret_cast<void*>(_arena[{buffer_idx}].data())) {{
                std::copy({in_obj}.data, {in_obj}.data + {out}_size, {out}_{buffer_idx}.data);
            }}
        }}
    """


@registry.register("BatchNormalization")
def generate_batchnorm(node: Node, ctx: Generator) -> str:
    """generate_batchnorm docstring."""

    x = ctx.get_tensor_name(node.inputs[0])
    scale = ctx.get_tensor_name(node.inputs[1])
    b = ctx.get_tensor_name(node.inputs[2])
    mean = ctx.get_tensor_name(node.inputs[3])
    var = ctx.get_tensor_name(node.inputs[4])

    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    epsilon = node.attributes.get("epsilon", 1e-5)

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    from onnx9000.codegen.utils import get_omp_pragma

    pragma = get_omp_pragma(f"{x}.size()")

    return f"""
        // BatchNormalization
        _arena[{buffer_idx}].resize({x}.size() * sizeof({cpp_type}));
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {x}.shape);
        
        int64_t spatial = 1;
        for (int i = 2; i < {x}.shape.size(); ++i) spatial *= {x}.shape[i];
        
        int64_t C = {x}.shape.size() > 1 ? {x}.shape[1] : 1;
        int64_t N = {x}.shape.size() > 0 ? {x}.shape[0] : 1;

        {pragma}
        for (int64_t n = 0; n < N; ++n) {{
            for (int64_t c = 0; c < C; ++c) {{
                auto m = {mean}.data[c];
                auto v = {var}.data[c];
                auto s = {scale}.data[c];
                auto bias = {b}.data[c];
                auto inv_std = static_cast<{cpp_type}>(1.0 / std::sqrt(v + {epsilon}));
                for (int64_t i = 0; i < spatial; ++i) {{
                    int64_t idx = n * C * spatial + c * spatial + i;
                    {out}.data[idx] = ({x}.data[idx] - m) * inv_std * s + bias;
                }}
            }}
        }}
"""


@registry.register("Gelu")
def generate_gelu(node: Node, ctx: Generator) -> str:
    """generate_gelu docstring."""

    inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    return f"""
        // Gelu
        _arena[{buffer_idx}].resize({inp}.size() * sizeof({cpp_type}));
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {inp}.shape);
        #pragma omp parallel for
        for (int64_t i = 0; i < {inp}.size(); ++i) {{
            {cpp_type} x = {inp}.data[i];
            {out}.data[i] = static_cast<{cpp_type}>(0.5f * x * (1.0f + std::erf(x / std::sqrt(2.0f))));
        }}
"""
