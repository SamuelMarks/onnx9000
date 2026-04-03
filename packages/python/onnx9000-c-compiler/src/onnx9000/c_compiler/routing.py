"""Tensor Manipulation, Shape, & Routing operation implementations for ONNX to C89 generation."""

from onnx9000.c_compiler.ast_builder import C89Builder
from onnx9000.c_compiler.spatial import get_attribute
from onnx9000.core.ir import Node, Tensor
from onnx9000.core.profiler import resolve_volume


def generate_shape_op(
    b: C89Builder,
    node: Node,
    out_tensor: Tensor,
    in_tensor: Tensor,
    in_name: str,
    out_name: str,
    op_type: str,
):
    """Generate no-op shape updates (Reshape, Flatten, Squeeze, Unsqueeze)."""
    b.emit(f"/* {op_type} (Shape mapping only, memory reused by arena if planned properly) */")
    b.emit("{")
    b.push_indent()
    if out_name != in_name:
        b.emit(f"/* Memory planner didn't alias {in_name} and {out_name}, falling back to copy */")
        volume = resolve_volume(out_tensor.shape)
        b.emit("int i;")
        b.emit(f"for (i = 0; i < {volume}; ++i) {{")
        b.push_indent()
        b.emit(f"{out_name}[i] = {in_name}[i];")
        b.pop_indent()
        b.emit("}")
    b.pop_indent()
    b.emit("}")


def generate_transpose(
    b: C89Builder, node: Node, out_tensor: Tensor, in_tensor: Tensor, in_name: str, out_name: str
):
    """Generate Transpose."""
    b.emit(f"/* {node.op_type} */")
    b.emit("{")
    b.push_indent()
    in_shape = in_tensor.shape
    ndim = len(in_shape)
    perm = get_attribute(node, "perm", list(reversed(range(ndim))))
    if ndim == 2:
        (M, N) = (in_shape[0], in_shape[1])
        b.emit("int i, j;")
        b.emit(f"for (i = 0; i < {M}; ++i) {{")
        b.push_indent()
        b.emit(f"for (j = 0; j < {N}; ++j) {{")
        b.push_indent()
        b.emit(f"{out_name}[j * {M} + i] = {in_name}[i * {N} + j];")
        b.pop_indent()
        b.emit("}")
        b.pop_indent()
        b.emit("}")
    elif ndim == 3:
        (D0, D1, D2) = (in_shape[0], in_shape[1], in_shape[2])
        (P0, P1, P2) = (perm[0], perm[1], perm[2])
        b.emit("int i0, i1, i2;")
        b.emit(f"for (i0 = 0; i0 < {D0}; ++i0) {{")
        b.push_indent()
        b.emit(f"for (i1 = 0; i1 < {D1}; ++i1) {{")
        b.push_indent()
        b.emit(f"for (i2 = 0; i2 < {D2}; ++i2) {{")
        b.push_indent()
        b.emit(f"int in_idx = i0 * {D1 * D2} + i1 * {D2} + i2;")
        out_dims = [0, 0, 0]
        out_dims[P0] = 0
        out_dims[P1] = 1
        out_dims[P2] = 2
        idx_map = {0: "i0", 1: "i1", 2: "i2"}
        out_idx_vars = [
            idx_map[out_dims.index(0)],
            idx_map[out_dims.index(1)],
            idx_map[out_dims.index(2)],
        ]
        out_shape = out_tensor.shape
        (O1, O2) = (out_shape[1], out_shape[2])
        b.emit(
            f"int out_idx = {out_idx_vars[0]} * {O1 * O2} + {out_idx_vars[1]} * {O2} + {out_idx_vars[2]};"
        )
        b.emit(f"{out_name}[out_idx] = {in_name}[in_idx];")
        b.pop_indent()
        b.emit("}")
        b.pop_indent()
        b.emit("}")
        b.pop_indent()
        b.emit("}")
    else:
        b.emit("/* Unsupported Transpose dimensionality in strict C89 fallback for now */")
    b.pop_indent()
    b.emit("}")


def generate_concat(
    b: C89Builder,
    node: Node,
    out_tensor: Tensor,
    in_tensors: list[Tensor],
    in_names: list[str],
    out_name: str,
):
    """Generate Concat loop."""
    b.emit(f"/* {node.op_type} */")
    b.emit("{")
    b.push_indent()
    axis = get_attribute(node, "axis", 0)
    out_shape = out_tensor.shape
    pre_axis_vol = resolve_volume(out_shape[:axis]) if axis > 0 else 1
    post_axis_vol = resolve_volume(out_shape[axis + 1 :]) if axis < len(out_shape) - 1 else 1
    b.emit("int pre, post, i;")
    b.emit(f"for (pre = 0; pre < {pre_axis_vol}; ++pre) {{")
    b.push_indent()
    b.emit("int out_axis_offset = 0;")
    for in_idx, in_tensor in enumerate(in_tensors):
        in_name = in_names[in_idx]
        axis_dim = in_tensor.shape[axis]
        b.emit(f"for (i = 0; i < {axis_dim}; ++i) {{")
        b.push_indent()
        b.emit(f"for (post = 0; post < {post_axis_vol}; ++post) {{")
        b.push_indent()
        out_idx = f"pre * {out_shape[axis] * post_axis_vol} + (out_axis_offset + i) * {post_axis_vol} + post"
        in_idx_str = f"pre * {axis_dim * post_axis_vol} + i * {post_axis_vol} + post"
        b.emit(f"{out_name}[{out_idx}] = {in_name}[{in_idx_str}];")
        b.pop_indent()
        b.emit("}")
        b.pop_indent()
        b.emit("}")
        b.emit(f"out_axis_offset += {axis_dim};")
    b.pop_indent()
    b.emit("}")
    b.pop_indent()
    b.emit("}")


def generate_pad(
    b: C89Builder,
    node: Node,
    out_tensor: Tensor,
    in_tensor: Tensor,
    pads_tensor: Tensor,
    val_tensor: Tensor,
    in_name: str,
    pads_name: str,
    val_name: str,
    out_name: str,
):
    """Generate Pad loop strictly. Supports dynamic padding if arena accommodates it, but usually static."""
    b.emit(f"/* {node.op_type} */")
    b.emit("{")
    b.push_indent()
    in_shape = in_tensor.shape
    out_shape = out_tensor.shape
    import struct

    if (
        pads_tensor
        and (
            getattr(pads_tensor, "is_initializer", False)
            or type(pads_tensor).__name__ == "Constant"
        )
        and getattr(pads_tensor, "data", None)
    ):
        fmt = "q" * len(pads_tensor.shape) * 2
        pads = struct.unpack(f"<{fmt}", pads_tensor.data[: struct.calcsize(f"<{fmt}")])
    else:
        pads = [0] * (len(in_shape) * 2)
    val = 0.0
    if (
        val_tensor
        and (
            getattr(val_tensor, "is_initializer", False) or type(val_tensor).__name__ == "Constant"
        )
        and getattr(val_tensor, "data", None)
    ):
        val = struct.unpack("<f", val_tensor.data[: struct.calcsize("<f")])[0]
    if len(in_shape) == 2:
        b.emit("int i, j;")
        b.emit(f"for (i = 0; i < {out_shape[0]}; ++i) {{")
        b.push_indent()
        b.emit(f"for (j = 0; j < {out_shape[1]}; ++j) {{")
        b.push_indent()
        b.emit(
            f"if (i < {pads[0]} || i >= {pads[0] + in_shape[0]} || j < {pads[1]} || j >= {pads[1] + in_shape[1]}) {{"
        )
        b.push_indent()
        b.emit(f"{out_name}[i * {out_shape[1]} + j] = {val}f;")
        b.pop_indent()
        b.emit("} else {")
        b.push_indent()
        b.emit(
            f"{out_name}[i * {out_shape[1]} + j] = {in_name}[(i - {pads[0]}) * {in_shape[1]} + (j - {pads[1]})];"
        )
        b.pop_indent()
        b.emit("}")
        b.pop_indent()
        b.emit("}")
        b.pop_indent()
        b.emit("}")
    b.pop_indent()
    b.emit("}")


def generate_slice(b, node, out_tensor, in_tensors, in_names, out_name):
    """Perform generate slice operation."""
    b.emit(f"/* {node.op_type} */")
    b.emit("{")
    b.push_indent()
    in_shape = in_tensors[0].shape
    ndims = len(in_shape)
    idx_vars = []
    for i in range(ndims):
        idx_var = b.new_var(f"d{i}")
        idx_vars.append(idx_var)
        b.emit(f"int {idx_var};")
    b.emit("int out_idx = 0;")
    b.emit("int starts[10] = {0};")
    b.emit("int ends[10] = {0};")
    b.emit("int axes[10] = {0};")
    b.emit("int steps[10] = {0};")
    for i in range(ndims):
        b.emit(f"ends[{i}] = {in_shape[i]};")
        b.emit(f"axes[{i}] = {i};")
        b.emit(f"steps[{i}] = 1;")
    if len(in_names) > 1:
        b.emit(
            f"for(int i=0; i<{len(in_tensors[1].shape)}; i++) starts[i] = (int){in_names[1]}[i];"
        )
    if len(in_names) > 2:
        b.emit(f"for(int i=0; i<{len(in_tensors[2].shape)}; i++) ends[i] = (int){in_names[2]}[i];")
    if len(in_names) > 3:
        b.emit(f"for(int i=0; i<{len(in_tensors[3].shape)}; i++) axes[i] = (int){in_names[3]}[i];")
    if len(in_names) > 4:
        b.emit(f"for(int i=0; i<{len(in_tensors[4].shape)}; i++) steps[i] = (int){in_names[4]}[i];")
    b.emit("int real_starts[10] = {0};")
    b.emit("int real_ends[10] = {0};")
    b.emit("int real_steps[10] = {0};")
    for i in range(ndims):
        b.emit(f"real_ends[{i}] = {in_shape[i]};")
        b.emit(f"real_steps[{i}] = 1;")
    b.emit(f"for(int i=0; i<{(len(in_tensors[1].shape) if len(in_tensors) > 1 else 0)}; i++) {{")
    b.push_indent()
    b.emit("int ax = axes[i];")
    b.emit("if (ax < 0) ax += " + str(ndims) + ";")
    b.emit("real_starts[ax] = starts[i] < 0 ? starts[i] + " + str(ndims) + " : starts[i];")
    b.emit("real_ends[ax] = ends[i] < 0 ? ends[i] + " + str(ndims) + " : ends[i];")
    b.emit("real_steps[ax] = steps[i];")
    b.pop_indent()
    b.emit("}")
    for i in range(ndims):
        b.emit(
            f"for({idx_vars[i]} = real_starts[{i}]; {idx_vars[i]} < real_ends[{i}]; {idx_vars[i]} += real_steps[{i}]) {{"
        )
        b.push_indent()
    strides = []
    stride = 1
    for i in reversed(range(ndims)):
        strides.insert(0, stride)
        stride *= in_shape[i]
    flat_idx = " + ".join([f"({idx_vars[i]} * {strides[i]})" for i in range(ndims)])
    b.emit(f"{out_name}[out_idx++] = {in_names[0]}[{flat_idx}];")
    for i in range(ndims):
        b.pop_indent()
        b.emit("}")
    b.pop_indent()
    b.emit("}")


def generate_gather(b, node, out_tensor, in_tensors, in_names, out_name):
    """Perform generate gather operation."""
    b.emit(f"/* {node.op_type} */")
    b.emit("{")
    b.push_indent()
    axis = int(node.attributes.get("axis", getattr(node.attributes.get("axis"), "value", 0)))
    in_shape = in_tensors[0].shape
    indices_shape = in_tensors[1].shape
    out_shape = out_tensor.shape
    if axis < 0:
        axis += len(in_shape)
    b.emit("int out_idx = 0;")
    from onnx9000.core.profiler import resolve_volume

    strides = []
    stride = 1
    for i in reversed(range(len(in_shape))):
        strides.insert(0, stride)
        stride *= in_shape[i]
    axis_stride = strides[axis]
    axis_dim = in_shape[axis]
    b.emit(f"for (int i = 0; i < {resolve_volume(out_shape)}; ++i) {{")
    b.push_indent()
    b.emit("int in_idx = 0;")
    b.emit("int temp = i;")
    out_strides = []
    ostride = 1
    for i in reversed(range(len(out_shape))):
        out_strides.insert(0, ostride)
        ostride *= out_shape[i]
    idx_vars = []
    for d in range(len(out_shape)):
        v = b.new_var(f"o{d}")
        idx_vars.append(v)
        if d == len(out_shape) - 1:
            b.emit(f"int {v} = temp;")
        else:
            b.emit(f"int {v} = temp / {out_strides[d]};")
            b.emit(f"temp %= {out_strides[d]};")
    b.emit("in_idx = 0;")
    for d in range(axis):
        b.emit(f"in_idx += {idx_vars[d]} * {strides[d]};")
    indices_strides = []
    istride = 1
    for i in reversed(range(len(indices_shape))):
        indices_strides.insert(0, istride)
        istride *= indices_shape[i]
    indices_flat = []
    for d in range(len(indices_shape)):
        v = idx_vars[axis + d]
        if d == len(indices_shape) - 1:
            indices_flat.append(f"({v})")
        else:
            indices_flat.append(f"({v} * {indices_strides[d]})")
    if_expr = " + ".join(indices_flat) if indices_flat else "0"
    idx_val_var = b.new_var("idx_val")
    b.emit(f"int {idx_val_var} = (int){in_names[1]}[{if_expr}];")
    b.emit(f"if ({idx_val_var} < 0) {idx_val_var} += {axis_dim};")
    b.emit(f"if ({idx_val_var} < 0) {idx_val_var} = 0;")
    b.emit(f"if ({idx_val_var} >= {axis_dim}) {idx_val_var} = {axis_dim} - 1;")
    b.emit(f"in_idx += {idx_val_var} * {axis_stride};")
    for d in range(axis + 1, len(in_shape)):
        out_d = axis + len(indices_shape) + (d - axis - 1)
        b.emit(f"in_idx += {idx_vars[out_d]} * {strides[d]};")
    b.emit(f"{out_name}[out_idx++] = {in_names[0]}[in_idx];")
    b.pop_indent()
    b.emit("}")
    b.pop_indent()
    b.emit("}")


def generate_gathernd(b, node, out_tensor, in_tensors, in_names, out_name):
    """Perform generate gathernd operation."""
    b.emit(f"/* {node.op_type} */")
    b.emit("{")
    b.push_indent()
    in_shape = in_tensors[0].shape
    indices_shape = in_tensors[1].shape
    out_shape = out_tensor.shape
    batch_dims = int(
        node.attributes.get("batch_dims", getattr(node.attributes.get("batch_dims"), "value", 0))
    )
    b.emit("int out_idx = 0;")
    from onnx9000.core.profiler import resolve_volume

    in_strides = []
    stride = 1
    for i in reversed(range(len(in_shape))):
        in_strides.insert(0, stride)
        stride *= in_shape[i]
    out_strides = []
    ostride = 1
    for i in reversed(range(len(out_shape))):
        out_strides.insert(0, ostride)
        ostride *= out_shape[i]
    b.emit(f"for (int i = 0; i < {resolve_volume(out_shape)}; ++i) {{")
    b.push_indent()
    b.emit("int temp = i;")
    idx_vars = []
    for d in range(len(out_shape)):
        v = b.new_var(f"o{d}")
        idx_vars.append(v)
        if d == len(out_shape) - 1:
            b.emit(f"int {v} = temp;")
        else:
            b.emit(f"int {v} = temp / {out_strides[d]};")
            b.emit(f"temp %= {out_strides[d]};")
    b.emit("int in_idx = 0;")
    indices_strides = []
    istride = 1
    for i in reversed(range(len(indices_shape))):
        indices_strides.insert(0, istride)
        istride *= indices_shape[i]
    indices_flat = []
    for d in range(len(indices_shape) - 1):
        v = idx_vars[d]
        if d == len(indices_shape) - 2:
            indices_flat.append(f"({v} * {indices_strides[d]})")
        else:
            indices_flat.append(f"({v} * {indices_strides[d]})")
    base_idx_expr = " + ".join(indices_flat) if indices_flat else "0"
    k = indices_shape[-1]
    for b_d in range(batch_dims):
        b.emit(f"in_idx += {idx_vars[b_d]} * {in_strides[b_d]};")
    for j in range(k):
        v_idx = b.new_var("v_idx")
        b.emit(f"int {v_idx} = (int){in_names[1]}[{base_idx_expr} + {j}];")
        b.emit(f"if ({v_idx} < 0) {v_idx} += {in_shape[batch_dims + j]};")
        b.emit(f"if ({v_idx} < 0) {v_idx} = 0;")
        b.emit(
            f"if ({v_idx} >= {in_shape[batch_dims + j]}) {v_idx} = {in_shape[batch_dims + j]} - 1;"
        )
        b.emit(f"in_idx += {v_idx} * {in_strides[batch_dims + j]};")
    for rem in range(batch_dims + k, len(in_shape)):
        o_idx = len(indices_shape) - 1 + (rem - batch_dims - k)
        b.emit(f"in_idx += {idx_vars[o_idx]} * {in_strides[rem]};")
    b.emit(f"{out_name}[out_idx++] = {in_names[0]}[in_idx];")
    b.pop_indent()
    b.emit("}")
    b.pop_indent()
    b.emit("}")


def generate_scatter_elements(b, node, out_tensor, in_tensors, in_names, out_name):
    """Perform generate scatter elements operation."""
    b.emit(f"/* {node.op_type} */")
    b.emit("{")
    b.push_indent()
    in_shape = in_tensors[0].shape
    indices_shape = in_tensors[1].shape
    from onnx9000.core.profiler import resolve_volume

    vol = str(resolve_volume(in_shape))
    b.emit(f"if ({out_name} != {in_names[0]}) {{")
    b.push_indent()
    b.emit(f"memcpy({out_name}, {in_names[0]}, {vol} * sizeof(float));")
    b.pop_indent()
    b.emit("}")
    axis = int(node.attributes.get("axis", getattr(node.attributes.get("axis"), "value", 0)))
    if axis < 0:
        axis += len(in_shape)
    strides = []
    stride = 1
    for i in reversed(range(len(in_shape))):
        strides.insert(0, stride)
        stride *= in_shape[i]
    indices_vol = str(resolve_volume(indices_shape))
    indices_strides = []
    istride = 1
    for i in reversed(range(len(indices_shape))):
        indices_strides.insert(0, istride)
        istride *= indices_shape[i]
    b.emit(f"for (int i = 0; i < {indices_vol}; ++i) {{")
    b.push_indent()
    b.emit("int temp = i;")
    idx_vars = []
    for d in range(len(indices_shape)):
        v = b.new_var(f"o{d}")
        idx_vars.append(v)
        if d == len(indices_shape) - 1:
            b.emit(f"int {v} = temp;")
        else:
            b.emit(f"int {v} = temp / {indices_strides[d]};")
            b.emit(f"temp %= {indices_strides[d]};")
    idx_val_var = b.new_var("idx_val")
    b.emit(f"int {idx_val_var} = (int){in_names[1]}[i];")
    b.emit(f"if ({idx_val_var} < 0) {idx_val_var} += {in_shape[axis]};")
    b.emit(f"if ({idx_val_var} < 0) {idx_val_var} = 0;")
    b.emit(f"if ({idx_val_var} >= {in_shape[axis]}) {idx_val_var} = {in_shape[axis]} - 1;")
    b.emit("int out_idx = 0;")
    for d in range(len(in_shape)):
        if d == axis:
            b.emit(f"out_idx += {idx_val_var} * {strides[d]};")
        else:
            b.emit(f"out_idx += {idx_vars[d]} * {strides[d]};")
    reduction = str(
        node.attributes.get("reduction", getattr(node.attributes.get("reduction"), "value", "none"))
    )
    if reduction == "add":
        b.emit(f"{out_name}[out_idx] += {in_names[2]}[i];")
    elif reduction == "mul":
        b.emit(f"{out_name}[out_idx] *= {in_names[2]}[i];")
    elif reduction == "max":
        b.emit(f"{out_name}[out_idx] = ONNX9000_MAX({out_name}[out_idx], {in_names[2]}[i]);")
    elif reduction == "min":
        b.emit(f"{out_name}[out_idx] = ONNX9000_MIN({out_name}[out_idx], {in_names[2]}[i]);")
    else:
        b.emit(f"{out_name}[out_idx] = {in_names[2]}[i];")
    b.pop_indent()
    b.emit("}")
    b.pop_indent()
    b.emit("}")


def generate_scatternd(b, node, out_tensor, in_tensors, in_names, out_name):
    """Perform generate scatternd operation."""
    b.emit(f"/* {node.op_type} */")
    b.emit("{")
    b.push_indent()
    in_shape = in_tensors[0].shape
    indices_shape = in_tensors[1].shape
    updates_shape = in_tensors[2].shape
    from onnx9000.core.profiler import resolve_volume

    vol = str(resolve_volume(in_shape))
    b.emit(f"if ({out_name} != {in_names[0]}) {{")
    b.push_indent()
    b.emit(f"memcpy({out_name}, {in_names[0]}, {vol} * sizeof(float));")
    b.pop_indent()
    b.emit("}")
    str(
        node.attributes.get("reduction", getattr(node.attributes.get("reduction"), "value", "none"))
    )
    indices_shape[-1]
    str(resolve_volume(updates_shape))
    b.emit("}")


def generate_expand(b, node, out_tensor, in_tensors, in_names, out_name):
    """Perform generate expand operation."""
    b.emit(f"/* {node.op_type} */")
    b.emit("{")
    b.push_indent()
    from onnx9000.core.profiler import resolve_volume

    in_shape = in_tensors[0].shape
    out_shape = out_tensor.shape
    vol = str(resolve_volume(out_shape))
    b.emit(f"for (int i = 0; i < {vol}; ++i) {{")
    b.push_indent()
    from onnx9000.c_compiler.operations import resolve_broadcast_indices

    i_var = "i"
    idx1 = resolve_broadcast_indices(out_shape, in_shape, i_var)
    b.emit(f"{out_name}[i] = {in_names[0]}[{idx1}];")
    b.pop_indent()
    b.emit("}")
    b.pop_indent()
    b.emit("}")


def generate_tile(b, node, out_tensor, in_tensors, in_names, out_name):
    """Perform generate tile operation."""
    b.emit(f"/* {node.op_type} */")
    b.emit("{")
    b.push_indent()
    from onnx9000.core.profiler import resolve_volume

    in_shape = in_tensors[0].shape
    out_shape = out_tensor.shape
    vol = str(resolve_volume(out_shape))
    in_strides = []
    stride = 1
    for i in reversed(range(len(in_shape))):
        in_strides.insert(0, stride)
        stride *= in_shape[i]
    out_strides = []
    ostride = 1
    for i in reversed(range(len(out_shape))):
        out_strides.insert(0, ostride)
        ostride *= out_shape[i]
    b.emit(f"for (int i = 0; i < {vol}; ++i) {{")
    b.push_indent()
    b.emit("int in_idx = 0;")
    b.emit("int temp = i;")
    for d in range(len(out_shape)):
        v = b.new_var(f"o{d}")
        if d == len(out_shape) - 1:
            b.emit(f"int {v} = temp;")
        else:
            b.emit(f"int {v} = temp / {out_strides[d]};")
            b.emit(f"temp %= {out_strides[d]};")
        b.emit(f"in_idx += ({v} % {in_shape[d]}) * {in_strides[d]};")
    b.emit(f"{out_name}[i] = {in_names[0]}[in_idx];")
    b.pop_indent()
    b.emit("}")
    b.pop_indent()
    b.emit("}")


def generate_constant_of_shape(b, node, out_tensor, in_tensors, in_names, out_name):
    """Perform generate constant of shape operation."""
    b.emit(f"/* {node.op_type} */")
    b.emit("{")
    b.push_indent()
    import struct

    from onnx9000.core.dtypes import DType
    from onnx9000.core.profiler import resolve_volume

    vol = str(resolve_volume(out_tensor.shape))
    val = node.attributes.get("value")
    val_str = "0"
    if val:
        val_tensor = getattr(val, "value", val)
        if getattr(val_tensor, "data", None):
            dtype = getattr(val_tensor, "dtype", DType.FLOAT32)
            if dtype in (DType.FLOAT32, DType.FLOAT64):
                unpacked = struct.unpack(
                    "<" + ("d" if dtype == DType.FLOAT64 else "f"),
                    val_tensor.data[: 8 if dtype == DType.FLOAT64 else 4],
                )[0]
                val_str = (
                    f"{unpacked:g}f"
                    if "." in f"{unpacked:g}" or "e" in f"{unpacked:g}"
                    else f"{unpacked:g}.0f"
                )
            elif dtype in (DType.INT32, DType.INT64, DType.INT8, DType.UINT8):
                fmt = (
                    "b"
                    if dtype == DType.INT8
                    else "B"
                    if dtype == DType.UINT8
                    else "i"
                    if dtype == DType.INT32
                    else "q"
                )
                unpacked = struct.unpack("<" + fmt, val_tensor.data[: struct.calcsize(fmt)])[0]
                val_str = str(unpacked)
    b.emit(f"for (int i = 0; i < {vol}; ++i) {{")
    b.push_indent()
    b.emit(f"{out_name}[i] = {val_str};")
    b.pop_indent()
    b.emit("}")
    b.pop_indent()
    b.emit("}")


def generate_cumsum(b, node, out_tensor, in_tensors, in_names, out_name):
    """Perform generate cumsum operation."""
    b.emit(f"/* {node.op_type} */")
    b.emit("{")
    b.push_indent()
    in_shape = in_tensors[0].shape
    from onnx9000.core.profiler import resolve_volume

    b.emit("int axis = 0;")
    if len(in_names) > 1:
        b.emit(f"axis = (int){in_names[1]}[0];")
    b.emit(f"if (axis < 0) axis += {len(in_shape)};")
    b.emit(f"memcpy({out_name}, {in_names[0]}, {resolve_volume(in_shape)} * sizeof(float));")
    exclusive = int(
        node.attributes.get("exclusive", getattr(node.attributes.get("exclusive"), "value", 0))
    )
    reverse = int(
        node.attributes.get("reverse", getattr(node.attributes.get("reverse"), "value", 0))
    )
    in_strides = []
    stride = 1
    for i in reversed(range(len(in_shape))):
        in_strides.insert(0, stride)
        stride *= in_shape[i]
    b.emit("for (int d = 0; d < " + str(len(in_shape)) + "; ++d) {")
    b.push_indent()
    b.emit("if (d == axis) {")
    b.push_indent()
    b.emit("int outer = 1;")
    b.emit("int inner = 1;")
    b.emit("int dim = " + str(in_shape[0]) + ";")
    for i in range(len(in_shape)):
        b.emit(f"if (axis == {i}) dim = {in_shape[i]};")
        b.emit(f"if (axis > {i}) outer *= {in_shape[i]};")
        b.emit(f"if (axis < {i}) inner *= {in_shape[i]};")
    b.emit("for (int o = 0; o < outer; ++o) {")
    b.push_indent()
    b.emit("for (int i = 0; i < inner; ++i) {")
    b.push_indent()
    if reverse:
        b.emit("float acc = 0;")
        b.emit("for (int j = dim - 1; j >= 0; --j) {")
        b.push_indent()
        b.emit("int idx = o * dim * inner + j * inner + i;")
        b.emit(f"float val = {in_names[0]}[idx];")
        if exclusive:
            b.emit(f"{out_name}[idx] = acc;")
            b.emit("acc += val;")
        else:
            b.emit("acc += val;")
            b.emit(f"{out_name}[idx] = acc;")
        b.pop_indent()
        b.emit("}")
    else:
        b.emit("float acc = 0;")
        b.emit("for (int j = 0; j < dim; ++j) {")
        b.push_indent()
        b.emit("int idx = o * dim * inner + j * inner + i;")
        b.emit(f"float val = {in_names[0]}[idx];")
        if exclusive:
            b.emit(f"{out_name}[idx] = acc;")
            b.emit("acc += val;")
        else:
            b.emit("acc += val;")
            b.emit(f"{out_name}[idx] = acc;")
        b.pop_indent()
        b.emit("}")
    b.pop_indent()
    b.emit("}")
    b.pop_indent()
    b.emit("}")
    b.pop_indent()
    b.emit("}")
    b.pop_indent()
    b.emit("}")
    b.pop_indent()
    b.emit("}")


def generate_reverse_sequence(b, node, out_tensor, in_tensors, in_names, out_name):
    """Perform generate reverse sequence operation."""
    b.emit(f"/* {node.op_type} */")
    b.emit("{")
    b.push_indent()
    in_shape = in_tensors[0].shape
    batch_axis = int(
        node.attributes.get("batch_axis", getattr(node.attributes.get("batch_axis"), "value", 1))
    )
    time_axis = int(
        node.attributes.get("time_axis", getattr(node.attributes.get("time_axis"), "value", 0))
    )
    from onnx9000.core.profiler import resolve_volume

    b.emit(f"memcpy({out_name}, {in_names[0]}, {resolve_volume(in_shape)} * sizeof(float));")
    in_strides = []
    stride = 1
    for i in reversed(range(len(in_shape))):
        in_strides.insert(0, stride)
        stride *= in_shape[i]
    b.emit("int out_idx = 0;")
    b.emit(f"for (int i = 0; i < {resolve_volume(in_shape)}; ++i) {{")
    b.push_indent()
    b.emit("int temp = i;")
    idx_vars = []
    for d in range(len(in_shape)):
        v = b.new_var(f"o{d}")
        idx_vars.append(v)
        if d == len(in_shape) - 1:
            b.emit(f"int {v} = temp;")
        else:
            b.emit(f"int {v} = temp / {in_strides[d]};")
            b.emit(f"temp %= {in_strides[d]};")
    b.emit(f"int batch_idx = {idx_vars[batch_axis]};")
    b.emit(f"int seq_len = (int){in_names[1]}[batch_idx];")
    b.emit(f"int time_idx = {idx_vars[time_axis]};")
    b.emit("if (time_idx < seq_len) {")
    b.push_indent()
    b.emit("int rev_time_idx = seq_len - 1 - time_idx;")
    b.emit("int rev_idx = 0;")
    for d in range(len(in_shape)):
        if d == time_axis:
            b.emit(f"rev_idx += rev_time_idx * {in_strides[d]};")
        else:
            b.emit(f"rev_idx += {idx_vars[d]} * {in_strides[d]};")
    b.emit(f"{out_name}[i] = {in_names[0]}[rev_idx];")
    b.pop_indent()
    b.emit("}")
    b.pop_indent()
    b.emit("}")
    b.pop_indent()
    b.emit("}")


def generate_onehot(b, node, out_tensor, in_tensors, in_names, out_name):
    """Perform generate onehot operation."""
    b.emit(f"/* {node.op_type} */")
    b.emit("{")
    b.push_indent()
    in_shape = in_tensors[0].shape
    out_shape = out_tensor.shape
    axis = int(node.attributes.get("axis", getattr(node.attributes.get("axis"), "value", -1)))
    if axis < 0:
        axis += len(in_shape) + 1
    from onnx9000.core.profiler import resolve_volume

    b.emit(f"int out_vol = {resolve_volume(out_shape)};")
    b.emit("int depth = (int)" + in_names[1] + "[0];")
    b.emit(f"float off_value = {in_names[2]}[0];")
    b.emit(f"float on_value = {in_names[2]}[1];")
    b.emit("for (int i = 0; i < out_vol; ++i) {")
    b.push_indent()
    b.emit(f"{out_name}[i] = off_value;")
    b.pop_indent()
    b.emit("}")
    in_strides = []
    stride = 1
    for i in reversed(range(len(in_shape))):
        in_strides.insert(0, stride)
        stride *= in_shape[i]
    out_strides = []
    ostride = 1
    for i in reversed(range(len(out_shape))):
        out_strides.insert(0, ostride)
        ostride *= out_shape[i]
    b.emit(f"for (int i = 0; i < {resolve_volume(in_shape)}; ++i) {{")
    b.push_indent()
    b.emit("int temp = i;")
    idx_vars = []
    for d in range(len(in_shape)):
        v = b.new_var(f"o{d}")
        idx_vars.append(v)
        if d == len(in_shape) - 1:
            b.emit(f"int {v} = temp;")
        else:
            b.emit(f"int {v} = temp / {in_strides[d]};")
            b.emit(f"temp %= {in_strides[d]};")
    b.emit(f"int v_val = (int){in_names[0]}[i];")
    b.emit("if (v_val < 0) v_val += depth;")
    b.emit("if (v_val >= 0 && v_val < depth) {")
    b.push_indent()
    b.emit("int out_idx = 0;")
    for d in range(len(out_shape)):
        if d == axis:
            b.emit(f"out_idx += v_val * {out_strides[d]};")
        elif d < axis:
            b.emit(f"out_idx += {idx_vars[d]} * {out_strides[d]};")
        else:
            b.emit(f"out_idx += {idx_vars[d - 1]} * {out_strides[d]};")
    b.emit(f"{out_name}[out_idx] = on_value;")
    b.pop_indent()
    b.emit("}")
    b.pop_indent()
    b.emit("}")
    b.pop_indent()
    b.emit("}")


def generate_depth_to_space(b, node, out_tensor, in_tensors, in_names, out_name):
    """Perform generate depth to space operation."""
    b.emit(f"/* {node.op_type} */")
    b.emit("{")
    b.push_indent()
    in_shape = in_tensors[0].shape
    out_shape = out_tensor.shape
    blocksize = int(
        node.attributes.get("blocksize", getattr(node.attributes.get("blocksize"), "value", 0))
    )
    mode = str(node.attributes.get("mode", getattr(node.attributes.get("mode"), "value", "DCR")))
    from onnx9000.core.profiler import resolve_volume

    b.emit(f"int out_vol = {resolve_volume(out_shape)};")
    b.emit("for (int i = 0; i < out_vol; ++i) {")
    b.push_indent()
    b.emit("int temp = i;")
    out_strides = []
    ostride = 1
    for i in reversed(range(len(out_shape))):
        out_strides.insert(0, ostride)
        ostride *= out_shape[i]
    idx_vars = []
    for d in range(len(out_shape)):
        v = b.new_var(f"o{d}")
        idx_vars.append(v)
        if d == len(out_shape) - 1:
            b.emit(f"int {v} = temp;")
        else:
            b.emit(f"int {v} = temp / {out_strides[d]};")
            b.emit(f"temp %= {out_strides[d]};")
    b.emit(f"int b_idx = {idx_vars[0]};")
    b.emit(f"int c = {idx_vars[1]};")
    b.emit(f"int h = {idx_vars[2]};")
    b.emit(f"int w = {idx_vars[3]};")
    b.emit("int in_c;")
    if mode == "CRD":
        b.emit(
            f"in_c = c * {blocksize * blocksize} + (h % {blocksize}) * {blocksize} + (w % {blocksize});"
        )
    else:
        b.emit(
            f"in_c = c + (h % {blocksize}) * {out_shape[1]} * {blocksize} + (w % {blocksize}) * {out_shape[1]};"
        )
    b.emit(f"int in_h = h / {blocksize};")
    b.emit(f"int in_w = w / {blocksize};")
    in_strides = []
    istride = 1
    for i in reversed(range(len(in_shape))):
        in_strides.insert(0, istride)
        istride *= in_shape[i]
    b.emit(
        f"int in_idx = b_idx * {in_strides[0]} + in_c * {in_strides[1]} + in_h * {in_strides[2]} + in_w;"
    )
    b.emit(f"{out_name}[i] = {in_names[0]}[in_idx];")
    b.pop_indent()
    b.emit("}")
    b.pop_indent()
    b.emit("}")


def generate_space_to_depth(b, node, out_tensor, in_tensors, in_names, out_name):
    """Perform generate space to depth operation."""
    b.emit(f"/* {node.op_type} */")
    b.emit("{")
    b.push_indent()
    in_shape = in_tensors[0].shape
    out_shape = out_tensor.shape
    blocksize = int(
        node.attributes.get("blocksize", getattr(node.attributes.get("blocksize"), "value", 0))
    )
    from onnx9000.core.profiler import resolve_volume

    b.emit(f"int out_vol = {resolve_volume(out_shape)};")
    b.emit("for (int i = 0; i < out_vol; ++i) {")
    b.push_indent()
    b.emit("int temp = i;")
    out_strides = []
    ostride = 1
    for i in reversed(range(len(out_shape))):
        out_strides.insert(0, ostride)
        ostride *= out_shape[i]
    idx_vars = []
    for d in range(len(out_shape)):
        v = b.new_var(f"o{d}")
        idx_vars.append(v)
        if d == len(out_shape) - 1:
            b.emit(f"int {v} = temp;")
        else:
            b.emit(f"int {v} = temp / {out_strides[d]};")
            b.emit(f"temp %= {out_strides[d]};")
    b.emit(f"int b_idx = {idx_vars[0]};")
    b.emit(f"int c = {idx_vars[1]};")
    b.emit(f"int h = {idx_vars[2]};")
    b.emit(f"int w = {idx_vars[3]};")
    b.emit(f"int in_c = c % {in_shape[1]};")
    b.emit(f"int in_h = h * {blocksize} + (c / {in_shape[1]}) / {blocksize};")
    b.emit(f"int in_w = w * {blocksize} + (c / {in_shape[1]}) % {blocksize};")
    in_strides = []
    istride = 1
    for i in reversed(range(len(in_shape))):
        in_strides.insert(0, istride)
        istride *= in_shape[i]
    b.emit(
        f"int in_idx = b_idx * {in_strides[0]} + in_c * {in_strides[1]} + in_h * {in_strides[2]} + in_w;"
    )
    b.emit(f"{out_name}[i] = {in_names[0]}[in_idx];")
    b.pop_indent()
    b.emit("}")
    b.pop_indent()
    b.emit("}")
