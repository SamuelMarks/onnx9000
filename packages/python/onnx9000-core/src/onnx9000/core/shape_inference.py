"""Static shape inference module."""

from typing import Any, Optional, Union

from onnx9000.core.dtypes import DType
from onnx9000.core.exceptions import ShapeInferenceError
from onnx9000.core.ir import DynamicDim, Graph, Node, Tensor, ValueInfo
from onnx9000.core.symbolic import broadcast_shapes, evaluate_symbolic_expression, simplify_dim
from onnx9000.core.utils import topological_sort


def _promote_types(t1: DType, t2: DType) -> DType:
    # A simple type promotion rule implementation
    if t1 == t2:
        return t1
    if t1 == DType.FLOAT64 or t2 == DType.FLOAT64:
        return DType.FLOAT64
    if t1 == DType.FLOAT32 or t2 == DType.FLOAT32:
        return DType.FLOAT32
    if t1 == DType.FLOAT16 or t2 == DType.FLOAT16:
        return DType.FLOAT16
    if t1 == DType.INT64 or t2 == DType.INT64:
        return DType.INT64
    if t1 == DType.INT32 or t2 == DType.INT32:
        return DType.INT32
    return t1


def get_attr(node: Node, name: str, default: Any = None) -> Any:
    for attr in node.attributes.values():
        if attr.name == name:
            return attr.value
    return default


def infer_shapes_and_types(graph: Graph) -> None:
    """
    Performs static shape and type inference on the given graph.
    Updates the graph.tensors and graph.value_info intrinsically.
    """
    try:
        sorted_nodes = topological_sort(graph)
    except Exception as e:
        raise ShapeInferenceError(f"Cannot infer shapes on a cyclic graph: {e}") from e

    env: dict[str, ValueInfo] = {}

    for inp in graph.inputs:
        if isinstance(inp, str) and inp in graph.tensors:
            t = graph.tensors[inp]
            env[inp] = ValueInfo(t.name, t.shape, t.dtype)
        elif hasattr(inp, "name"):
            env[inp.name] = inp
    for _tensor_name, tensor in graph.tensors.items():
        if tensor.is_initializer:
            env[tensor.name] = ValueInfo(tensor.name, tensor.shape, tensor.dtype)

    for node in sorted_nodes:
        out_shapes = []
        out_dtypes = []

        if node.op_type in [
            "Add",
            "Sub",
            "Mul",
            "Div",
            "And",
            "Or",
            "Equal",
            "Less",
            "Greater",
            "Where",
        ]:
            if len(node.inputs) < 2:
                continue
            in1 = env.get(node.inputs[0])
            in2 = env.get(node.inputs[1])
            if not in1 or not in2:
                continue
            try:
                out_shape = broadcast_shapes(in1.shape, in2.shape)
            except ShapeInferenceError as e:
                raise ShapeInferenceError(f"Node {node.name} ({node.op_type}): {e}") from e

            out_dtype = _promote_types(in1.dtype, in2.dtype)
            if node.op_type in ["And", "Or", "Equal", "Less", "Greater"]:
                out_dtype = DType.BOOL

            if node.op_type == "Where" and len(node.inputs) == 3:
                in3 = env.get(node.inputs[2])
                if in3:
                    out_shape = broadcast_shapes(out_shape, in3.shape)
                    out_dtype = _promote_types(in2.dtype, in3.dtype)

            out_shapes = [out_shape] * len(node.outputs)
            out_dtypes = [out_dtype] * len(node.outputs)

        elif node.op_type in ["MatMul", "Gemm"]:
            if len(node.inputs) < 2:
                continue
            in1 = env.get(node.inputs[0])
            in2 = env.get(node.inputs[1])
            if not in1 or not in2:
                continue
            shape1 = list(in1.shape)
            shape2 = list(in2.shape)

            transA = get_attr(node, "transA", 0) if node.op_type == "Gemm" else 0
            transB = get_attr(node, "transB", 0) if node.op_type == "Gemm" else 0

            if transA and len(shape1) >= 2:
                shape1[-1], shape1[-2] = shape1[-2], shape1[-1]
            if transB and len(shape2) >= 2:
                shape2[-1], shape2[-2] = shape2[-2], shape2[-1]

            if len(shape1) >= 2 and len(shape2) >= 2:
                batch_shape = []
                if len(shape1) > 2 or len(shape2) > 2:
                    batch1 = tuple(shape1[:-2])
                    batch2 = tuple(shape2[:-2])
                    try:
                        batch_shape = list(broadcast_shapes(batch1, batch2))
                    except ShapeInferenceError:
                        batch_shape = list(batch1) if len(batch1) > len(batch2) else list(batch2)

                out_shape = tuple(batch_shape + [shape1[-2], shape2[-1]])
            else:
                out_shape = ()

            out_dtype = _promote_types(in1.dtype, in2.dtype)
            out_shapes = [out_shape] * len(node.outputs)
            out_dtypes = [out_dtype] * len(node.outputs)

        elif node.op_type in ["Relu", "Sigmoid", "Tanh", "Exp", "Log", "Cast", "Shape", "Size"]:
            if len(node.inputs) < 1:
                continue
            in1 = env.get(node.inputs[0])
            if not in1:
                continue

            if node.op_type == "Cast":
                to_type = get_attr(node, "to", DType.FLOAT32.value)
                out_dtype = DType(to_type)
                out_shape = in1.shape
            elif node.op_type == "Shape":
                out_dtype = DType.INT64
                out_shape = (len(in1.shape),)
            elif node.op_type == "Size":
                out_dtype = DType.INT64
                out_shape = ()
            else:
                out_dtype = in1.dtype
                out_shape = in1.shape

            out_shapes = [out_shape] * len(node.outputs)
            out_dtypes = [out_dtype] * len(node.outputs)

        elif node.op_type == "Reshape":
            if len(node.inputs) < 2:
                continue
            in1 = env.get(node.inputs[0])
            if not in1:
                continue

            shape_tensor_name = node.inputs[1]
            out_shape = None
            if shape_tensor_name in graph.tensors:
                shape_tensor = graph.tensors[shape_tensor_name]
                if hasattr(shape_tensor, "data") and shape_tensor.data:
                    pass

            if not out_shape and hasattr(graph.tensors.get(shape_tensor_name, None), "values"):
                vals = getattr(graph.tensors[shape_tensor_name], "values", None)
                if vals is not None:
                    target_shape = list(vals)
                    try:
                        in_vol = 1
                        for d in in1.shape:
                            in_vol *= int(simplify_dim(d))

                        out_vol = 1
                        neg_idx = -1
                        for i, d in enumerate(target_shape):
                            if d == -1:
                                neg_idx = i
                            else:
                                out_vol *= int(simplify_dim(d))

                        if neg_idx != -1 and out_vol != 0:
                            target_shape[neg_idx] = in_vol // out_vol
                        out_shape = tuple(target_shape)
                    except Exception:
                        out_shape = tuple(vals)

            if not out_shape:
                s_shape = env.get(shape_tensor_name, ValueInfo("", (), DType.INT64)).shape
                dim_count = s_shape[0] if s_shape else 1
                out_shape = tuple(
                    DynamicDim(f"dim_{i}")
                    for i in range(dim_count if isinstance(dim_count, int) else 1)
                )

            out_shapes = [out_shape] * len(node.outputs)
            out_dtypes = [in1.dtype] * len(node.outputs)

        elif node.op_type in ["Conv", "MaxPool", "AveragePool", "GlobalAveragePool"]:
            if len(node.inputs) < 1:
                continue
            in1 = env.get(node.inputs[0])
            if not in1:
                continue

            in_shape = list(in1.shape)
            out_dtype = in1.dtype

            if node.op_type == "GlobalAveragePool":
                out_shape = tuple(in_shape[:2] + [1] * (len(in_shape) - 2))
                out_shapes = [out_shape] * len(node.outputs)
                out_dtypes = [out_dtype] * len(node.outputs)
            else:
                kernel_shape = get_attr(node, "kernel_shape", [])
                strides = get_attr(node, "strides", [1] * len(kernel_shape))
                pads = get_attr(node, "pads", [0] * (2 * len(kernel_shape)))
                dilations = get_attr(node, "dilations", [1] * len(kernel_shape))

                if node.op_type == "Conv" and len(node.inputs) > 1:
                    w_info = env.get(node.inputs[1])
                    if w_info and len(w_info.shape) > 2:
                        kernel_shape = list(w_info.shape[2:])
                        out_channels = w_info.shape[0]
                    else:
                        out_channels = DynamicDim("C_out")
                else:
                    out_channels = in_shape[1] if len(in_shape) > 1 else DynamicDim("C_out")

                spatial_dims = []
                for i in range(len(kernel_shape)):
                    try:
                        in_dim = in_shape[2 + i]
                        if isinstance(in_dim, DynamicDim) or isinstance(in_dim, str):
                            spatial_dims.append(DynamicDim(f"spatial_{i}"))
                        else:
                            k = kernel_shape[i]
                            s = strides[i]
                            p = pads[i] + pads[i + len(kernel_shape)]
                            d = dilations[i]
                            out_dim = (in_dim + p - ((k - 1) * d + 1)) // s + 1
                            spatial_dims.append(out_dim)
                    except Exception:
                        spatial_dims.append(DynamicDim(f"spatial_{i}"))

                out_shape = tuple(
                    [in_shape[0] if len(in_shape) > 0 else 1, out_channels] + spatial_dims
                )
                out_shapes = [out_shape] * len(node.outputs)
                out_dtypes = [out_dtype] * len(node.outputs)

        elif node.op_type == "ConvTranspose":
            if len(node.inputs) < 1:
                continue
            in1 = env.get(node.inputs[0])
            if not in1:
                continue
            in_shape = list(in1.shape)
            out_dtype = in1.dtype

            kernel_shape = get_attr(node, "kernel_shape", [])
            strides = get_attr(node, "strides", [1] * len(kernel_shape))
            pads = get_attr(node, "pads", [0] * (2 * len(kernel_shape)))
            dilations = get_attr(node, "dilations", [1] * len(kernel_shape))
            output_padding = get_attr(node, "output_padding", [0] * len(kernel_shape))

            if len(node.inputs) > 1:
                w_info = env.get(node.inputs[1])
                if w_info and len(w_info.shape) > 2:
                    kernel_shape = list(w_info.shape[2:])
                    out_channels = w_info.shape[1]
                else:
                    out_channels = DynamicDim("C_out")
            else:
                out_channels = DynamicDim("C_out")

            spatial_dims = []
            for i in range(len(kernel_shape)):
                try:
                    in_dim = in_shape[2 + i]
                    if isinstance(in_dim, DynamicDim) or isinstance(in_dim, str):
                        spatial_dims.append(DynamicDim(f"spatial_{i}"))
                    else:
                        k = kernel_shape[i]
                        s = strides[i]
                        p = pads[i] + pads[i + len(kernel_shape)]
                        d = dilations[i]
                        opad = output_padding[i]
                        out_dim = (in_dim - 1) * s - p + (k - 1) * d + 1 + opad
                        spatial_dims.append(out_dim)
                except Exception:
                    spatial_dims.append(DynamicDim(f"spatial_{i}"))

            out_shape = tuple(
                [in_shape[0] if len(in_shape) > 0 else 1, out_channels] + spatial_dims
            )
            out_shapes = [out_shape] * len(node.outputs)
            out_dtypes = [out_dtype] * len(node.outputs)

        elif node.op_type == "Gather":
            if len(node.inputs) < 2:
                continue
            in1 = env.get(node.inputs[0])
            in2 = env.get(node.inputs[1])
            if not in1 or not in2:
                continue
            axis = get_attr(node, "axis", 0)
            in_shape = list(in1.shape)
            indices_shape = list(in2.shape)
            if axis < 0:
                axis += len(in_shape)

            out_shape = tuple(in_shape[:axis] + indices_shape + in_shape[axis + 1 :])
            out_shapes = [out_shape] * len(node.outputs)
            out_dtypes = [in1.dtype] * len(node.outputs)

        elif node.op_type == "Slice":
            if len(node.inputs) < 1:
                continue
            in1 = env.get(node.inputs[0])
            if not in1:
                continue
            in_shape = list(in1.shape)
            out_shape = in_shape.copy()

            # Since slicing involves tensors (starts, ends, axes, steps),
            # if they are constants/values, we can evaluate them statically.
            # Otherwise we output symbolic/dynamic shapes

            axes = []
            starts = []
            ends = []
            steps = []

            def get_tensor_vals(idx, n=node, default_len=None):
                if len(n.inputs) > idx:
                    t_name = n.inputs[idx]
                    if t_name in graph.tensors and hasattr(graph.tensors[t_name], "values"):
                        vals = graph.tensors[t_name].values
                        if vals is not None:
                            return list(vals)
                return [] if default_len is None else [1] * default_len

            starts = get_tensor_vals(1)
            ends = get_tensor_vals(2)
            axes = get_tensor_vals(3)
            steps = get_tensor_vals(4)

            if not axes and starts:
                axes = list(range(len(starts)))
            if not steps and starts:
                steps = [1] * len(starts)

            for i, axis in enumerate(axes):
                if axis < 0:
                    axis += len(in_shape)
                if i < len(starts) and i < len(ends) and i < len(steps):
                    start = starts[i]
                    end = ends[i]
                    step = steps[i]
                    dim = out_shape[axis]
                    if isinstance(dim, int):
                        if start < 0:
                            start += dim
                        if end < 0:
                            end += dim
                        start = max(0, min(dim, start))
                        end = max(0, min(dim, end))
                        out_shape[axis] = (
                            max(0, (end - start + step - 1) // step)
                            if step > 0
                            else max(0, (start - end - step - 1) // -step)
                        )
                    else:
                        out_shape[axis] = DynamicDim(f"sliced_{axis}")

            out_shapes = [tuple(out_shape)] * len(node.outputs)
            out_dtypes = [in1.dtype] * len(node.outputs)

        elif node.op_type == "Concat":
            if len(node.inputs) < 1:
                continue
            in1 = env.get(node.inputs[0])
            if not in1:
                continue
            axis = get_attr(node, "axis", 0)
            in_shape = list(in1.shape)
            if axis < 0:
                axis += len(in_shape)

            sum_dim = 0
            is_dynamic = False
            for inp_name in node.inputs:
                inp_info = env.get(inp_name)
                if inp_info:
                    d = inp_info.shape[axis] if axis < len(inp_info.shape) else 0
                    if isinstance(d, int):
                        sum_dim += d
                    else:
                        is_dynamic = True

            out_shape = in_shape.copy()
            if is_dynamic:
                out_shape[axis] = DynamicDim(f"concat_{axis}")
            else:
                out_shape[axis] = sum_dim

            out_shapes = [tuple(out_shape)] * len(node.outputs)
            out_dtypes = [in1.dtype] * len(node.outputs)

        elif node.op_type == "Split":
            if len(node.inputs) < 1:
                continue
            in1 = env.get(node.inputs[0])
            if not in1:
                continue
            axis = get_attr(node, "axis", 0)
            in_shape = list(in1.shape)
            if axis < 0:
                axis += len(in_shape)

            splits = []
            if len(node.inputs) > 1:
                split_t = graph.tensors.get(node.inputs[1])
                if split_t and hasattr(split_t, "values") and split_t.values is not None:
                    splits = list(split_t.values)
            if not splits:
                splits = get_attr(node, "split", [])

            out_shapes = []
            out_dtypes = []
            num_outputs = len(node.outputs)
            dim_val = in_shape[axis] if axis < len(in_shape) else 0

            for i in range(num_outputs):
                shape_i = in_shape.copy()
                if splits and i < len(splits):
                    shape_i[axis] = splits[i]
                else:
                    if isinstance(dim_val, int) and dim_val % num_outputs == 0:
                        shape_i[axis] = dim_val // num_outputs
                    else:
                        shape_i[axis] = DynamicDim(f"split_{axis}")
                out_shapes.append(tuple(shape_i))
                out_dtypes.append(in1.dtype)

        elif node.op_type in ["Tile", "Expand"]:
            if len(node.inputs) < 2:
                continue
            in1 = env.get(node.inputs[0])
            if not in1:
                continue

            shape_t = graph.tensors.get(node.inputs[1])
            repeats = []
            if shape_t and hasattr(shape_t, "values") and shape_t.values is not None:
                repeats = list(shape_t.values)

            if node.op_type == "Tile":
                out_shape = list(in1.shape)
                for i, r in enumerate(repeats):
                    if i < len(out_shape):
                        if isinstance(out_shape[i], int) and isinstance(r, int):
                            out_shape[i] *= r
                        else:
                            out_shape[i] = DynamicDim(f"tiled_{i}")
                out_shape = tuple(out_shape)
            else:  # Expand
                out_shape = []
                # Expand uses broadcasting rules between in1.shape and repeats (the target shape)
                try:
                    out_shape = broadcast_shapes(in1.shape, tuple(repeats))
                except ShapeInferenceError:
                    out_shape = tuple(DynamicDim(f"expanded_{i}") for i in range(len(repeats)))

            out_shapes = [out_shape] * len(node.outputs)
            out_dtypes = [in1.dtype] * len(node.outputs)

        elif node.op_type == "Pad":
            if len(node.inputs) < 1:
                continue
            in1 = env.get(node.inputs[0])
            if not in1:
                continue

            pads = []
            if len(node.inputs) > 1:
                pads_t = graph.tensors.get(node.inputs[1])
                if pads_t and hasattr(pads_t, "values") and pads_t.values is not None:
                    pads = list(pads_t.values)

            in_shape = list(in1.shape)
            out_shape = in_shape.copy()
            rank = len(in_shape)
            if len(pads) == 2 * rank:
                for i in range(rank):
                    p1 = pads[i]
                    p2 = pads[i + rank]
                    if isinstance(out_shape[i], int):
                        out_shape[i] += p1 + p2
                    else:
                        out_shape[i] = DynamicDim(f"padded_{i}")

            out_shapes = [tuple(out_shape)] * len(node.outputs)
            out_dtypes = [in1.dtype] * len(node.outputs)

        elif node.op_type in ["TopK", "ArgMax", "ArgMin"]:
            if len(node.inputs) < 1:
                continue
            in1 = env.get(node.inputs[0])
            if not in1:
                continue
            axis = get_attr(node, "axis", -1)
            keepdims = get_attr(node, "keepdims", 1)
            in_shape = list(in1.shape)
            if axis < 0:
                axis += len(in_shape)

            out_shape = in_shape.copy()
            if node.op_type == "TopK":
                k = 1
                if len(node.inputs) > 1:
                    k_t = graph.tensors.get(node.inputs[1])
                    if (
                        k_t
                        and hasattr(k_t, "values")
                        and k_t.values is not None
                        and len(k_t.values) > 0
                    ):
                        k = int(k_t.values[0])
                if axis < len(out_shape):
                    out_shape[axis] = k
                out_shapes = [tuple(out_shape)] * len(node.outputs)
                out_dtypes = [in1.dtype, DType.INT64] if len(node.outputs) > 1 else [in1.dtype]
            else:
                if keepdims:
                    if axis < len(out_shape):
                        out_shape[axis] = 1
                else:
                    if axis < len(out_shape):
                        out_shape.pop(axis)
                out_shapes = [tuple(out_shape)] * len(node.outputs)
                out_dtypes = [DType.INT64] * len(node.outputs)

        elif node.op_type == "If":
            # An If node contains a 'then_branch' and an 'else_branch' Graph
            then_graph = get_attr(node, "then_branch")
            else_graph = get_attr(node, "else_branch")

            if then_graph and isinstance(then_graph, Graph):
                infer_shapes_and_types(then_graph)
            if else_graph and isinstance(else_graph, Graph):
                infer_shapes_and_types(else_graph)

            out_shapes = []
            out_dtypes = []
            for i, out in enumerate(node.outputs):
                if then_graph and i < len(then_graph.outputs):
                    then_out_name = then_graph.outputs[i]
                    then_shape = (
                        then_graph.tensors[then_out_name].shape
                        if then_out_name in then_graph.tensors
                        else ()
                    )
                    then_dtype = (
                        then_graph.tensors[then_out_name].dtype
                        if then_out_name in then_graph.tensors
                        else DType.FLOAT32
                    )
                    out_shapes.append(then_shape)
                    out_dtypes.append(then_dtype)
                else:
                    out_shapes.append(())
                    out_dtypes.append(DType.FLOAT32)

        elif node.op_type == "Loop":
            # A Loop node contains a 'body' Graph
            body_graph = get_attr(node, "body")

            if body_graph and isinstance(body_graph, Graph):
                # The loop body receives (iter_count, cond_in, v_in...)
                # And returns (cond_out, v_out..., scan_out...)
                # Propagate input shapes into the subgraph
                for i, b_inp in enumerate(body_graph.inputs):
                    if i == 0:
                        continue  # iter_count
                    if i == 1:
                        continue  # cond_in
                    node_inp_idx = i  # node.inputs: (M, cond, v_in...)
                    if node_inp_idx < len(node.inputs):
                        ext_in = env.get(node.inputs[node_inp_idx])
                        if ext_in and b_inp in body_graph.tensors:
                            body_graph.tensors[b_inp].shape = ext_in.shape
                            body_graph.tensors[b_inp].dtype = ext_in.dtype

                infer_shapes_and_types(body_graph)

            out_shapes = []
            out_dtypes = []

            # node.outputs: (v_out..., scan_out...)
            num_v = len(node.inputs) - 2  # v_in count

            for i, out in enumerate(node.outputs):
                if body_graph and i + 1 < len(body_graph.outputs):
                    body_out = body_graph.outputs[i + 1]  # skip cond_out
                    if body_out in body_graph.tensors:
                        b_shape = body_graph.tensors[body_out].shape
                        b_dtype = body_graph.tensors[body_out].dtype

                        if i >= num_v:
                            # It's a scan output, prepend iteration dimension
                            m_shape = env.get(node.inputs[0]) if len(node.inputs) > 0 else None
                            m_val = (
                                m_shape.shape[0]
                                if m_shape and len(m_shape.shape) > 0
                                else DynamicDim("loop_iters")
                            )

                            m_tensor = (
                                graph.tensors.get(node.inputs[0]) if len(node.inputs) > 0 else None
                            )
                            if (
                                m_tensor
                                and hasattr(m_tensor, "values")
                                and m_tensor.values is not None
                                and len(m_tensor.values) > 0
                            ):
                                m_val = int(m_tensor.values[0])

                            out_shapes.append(tuple([m_val] + list(b_shape)))
                        else:
                            out_shapes.append(b_shape)
                        out_dtypes.append(b_dtype)
                    else:
                        out_shapes.append(())
                        out_dtypes.append(DType.FLOAT32)
                else:
                    out_shapes.append(())
                    out_dtypes.append(DType.FLOAT32)

        elif node.op_type == "NonZero":
            if len(node.inputs) < 1:
                continue
            in1 = env.get(node.inputs[0])
            if not in1:
                continue
            rank = len(in1.shape)
            out_shape = (rank, DynamicDim("nonzero_count"))
            out_shapes = [out_shape] * len(node.outputs)
            out_dtypes = [DType.INT64] * len(node.outputs)
        for i, out_name in enumerate(node.outputs):
            if i < len(out_shapes):
                s = out_shapes[i]
                d = out_dtypes[i]
            else:
                in0 = env.get(node.inputs[0]) if node.inputs else None
                s = ()
                d = in0.dtype if in0 else DType.FLOAT32

            env[out_name] = ValueInfo(out_name, s, d)
            if out_name not in graph.tensors:
                graph.add_tensor(Tensor(out_name, s, d))
            else:
                graph.tensors[out_name].shape = s
                graph.tensors[out_name].dtype = d
