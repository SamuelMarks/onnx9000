"""Provides constant_folding.py module functionality."""

import logging

import numpy as np
from onnx9000.core.ir import Attribute, Graph, Node, Tensor
from onnx9000.optimizer.simplifier.passes.base import GraphPass

logger = logging.getLogger(__name__)
MAX_CONSTANT_SIZE_BYTES = 10 * 1024 * 1024


def _evaluate_pool(x, kernel_shape, strides, pads, pool_mode="max", ceil_mode=0):

    from numpy.lib.stride_tricks import as_strided

    spatial_dims = len(kernel_shape)

    pad_width = [(0, 0), (0, 0)]
    if pads:
        for i in range(spatial_dims):
            pad_width.append((pads[i], pads[i + spatial_dims]))
    else:
        for i in range(spatial_dims):
            pad_width.append((0, 0))

    if pool_mode == "max":
        x_padded = np.pad(x, pad_width, mode="constant", constant_values=-np.inf)
    else:
        x_padded = np.pad(x, pad_width, mode="constant", constant_values=0)

    output_shape = [x.shape[0], x.shape[1]]
    strides_w = [x_padded.strides[0], x_padded.strides[1]]

    for i in range(spatial_dims):
        dim_in = x_padded.shape[2 + i]
        k = kernel_shape[i]
        s = strides[i] if strides else 1

        if ceil_mode:
            out_dim = int(np.ceil((dim_in - k) / s)) + 1
        else:
            out_dim = int(np.floor((dim_in - k) / s)) + 1

        output_shape.append(out_dim)
        strides_w.append(s * x_padded.strides[2 + i])

    shape_w = tuple(output_shape) + tuple(kernel_shape)

    for i in range(spatial_dims):
        strides_w.append(x_padded.strides[2 + i])

    x_w = as_strided(x_padded, shape=shape_w, strides=tuple(strides_w))
    axes_to_reduce = tuple(range(2 + spatial_dims, 2 + 2 * spatial_dims))

    if pool_mode == "max":
        return x_w.max(axis=axes_to_reduce)
    elif pool_mode == "avg":
        return x_w.mean(axis=axes_to_reduce)


def _evaluate_conv(x, w, b, strides, pads, dilations, group):

    from numpy.lib.stride_tricks import as_strided

    spatial_dims = len(x.shape) - 2

    if strides is None:
        strides = [1] * spatial_dims
    if pads is None:
        pads = [0] * (2 * spatial_dims)
    if dilations is None:
        dilations = [1] * spatial_dims

    pad_width = [(0, 0), (0, 0)]
    for i in range(spatial_dims):
        pad_width.append((pads[i], pads[i + spatial_dims]))

    x_padded = np.pad(x, pad_width, mode="constant", constant_values=0)

    kernel_shape = list(w.shape[2:])
    eff_kernel_size = [k + (k - 1) * (d - 1) for k, d in zip(kernel_shape, dilations)]

    output_shape = [x.shape[0], x.shape[1]]
    strides_w = [x_padded.strides[0], x_padded.strides[1]]

    for i in range(spatial_dims):
        dim_in = x_padded.shape[2 + i]
        k = eff_kernel_size[i]
        s = strides[i]
        out_dim = (dim_in - k) // s + 1
        output_shape.append(out_dim)
        strides_w.append(s * x_padded.strides[2 + i])

    for i in range(spatial_dims):
        strides_w.append(dilations[i] * x_padded.strides[2 + i])

    shape_w = tuple(output_shape) + tuple(kernel_shape)
    x_w = as_strided(x_padded, shape=shape_w, strides=tuple(strides_w))

    if group == 1:
        axes_A = [1] + list(range(2 + spatial_dims, 2 + 2 * spatial_dims))
        axes_W = [1] + list(range(2, 2 + spatial_dims))
        out = np.tensordot(x_w, w, axes=(axes_A, axes_W))
        out = np.moveaxis(out, -1, 1)
    else:
        N = x.shape[0]
        M = w.shape[0]
        C_per_g = x.shape[1] // group
        M_per_g = M // group
        out_shape = [N, M] + output_shape[2:]
        out = np.zeros(out_shape, dtype=x.dtype)

        axes_A = [1] + list(range(2 + spatial_dims, 2 + 2 * spatial_dims))
        axes_W = [1] + list(range(2, 2 + spatial_dims))

        for g in range(group):
            x_w_g = x_w[:, g * C_per_g : (g + 1) * C_per_g, ...]
            w_g = w[g * M_per_g : (g + 1) * M_per_g, ...]
            out_g = np.tensordot(x_w_g, w_g, axes=(axes_A, axes_W))
            out_g = np.moveaxis(out_g, -1, 1)
            out[:, g * M_per_g : (g + 1) * M_per_g, ...] = out_g

    if b is not None:
        b_shape = [1, w.shape[0]] + [1] * spatial_dims
        out += b.reshape(b_shape)
    return out


def _numpy_to_tensor_proto(arr, name=""):

    from onnx9000.core import onnx_pb2

    if isinstance(arr, (int, float, bool)):
        arr = np.array(arr)
    t = onnx_pb2.TensorProto()
    t.name = name
    t.dims.extend(arr.shape)

    dtype_mapping = {
        np.float32: onnx_pb2.TensorProto.FLOAT,
        np.float64: onnx_pb2.TensorProto.DOUBLE,
        np.float16: onnx_pb2.TensorProto.FLOAT16,
        np.int32: onnx_pb2.TensorProto.INT32,
        np.int64: onnx_pb2.TensorProto.INT64,
        np.int16: onnx_pb2.TensorProto.INT16,
        np.int8: onnx_pb2.TensorProto.INT8,
        np.uint8: onnx_pb2.TensorProto.UINT8,
        np.uint16: onnx_pb2.TensorProto.UINT16,
        np.uint32: onnx_pb2.TensorProto.UINT32,
        np.uint64: onnx_pb2.TensorProto.UINT64,
        np.bool_: onnx_pb2.TensorProto.BOOL,
    }

    t.data_type = dtype_mapping.get(arr.dtype.type, onnx_pb2.TensorProto.UNDEFINED)
    if t.data_type == onnx_pb2.TensorProto.UNDEFINED and arr.dtype == np.dtype("O"):
        # Might be strings, not handled yet
        pass
    else:
        # Ensure contiguous before tobytes()
        t.raw_data = np.ascontiguousarray(arr).tobytes()
    return t


def _tensor_to_numpy(tensor):

    from onnx9000.core.dtypes import DType

    if isinstance(tensor.data, np.ndarray):
        return tensor.data

    if tensor.data is None:
        return None

    dtype_mapping = {
        DType.FLOAT32: np.float32,
        DType.FLOAT64: np.float64,
        DType.FLOAT16: np.float16,
        DType.INT32: np.int32,
        DType.INT64: np.int64,
        DType.INT16: np.int16,
        DType.INT8: np.int8,
        DType.UINT8: np.uint8,
        DType.UINT16: np.uint16,
        DType.UINT32: np.uint32,
        DType.UINT64: np.uint64,
        DType.BOOL: bool,
    }

    if tensor.dtype == DType.BFLOAT16:
        # Pure Python fallback for BFloat16 -> Float32
        data = np.frombuffer(tensor.data, dtype=np.uint16)
        data = np.left_shift(data.astype(np.uint32), 16)
        data = data.view(np.float32)
        shape_list = [d.value if hasattr(d, "value") else d for d in tensor.shape]
        return data.reshape(shape_list)

    np_dtype = dtype_mapping.get(tensor.dtype)
    if not np_dtype:
        return None

    shape_list = [d.value if hasattr(d, "value") else d for d in tensor.shape]
    try:
        return np.frombuffer(tensor.data, dtype=np_dtype).reshape(shape_list)
    except Exception:
        return None


class ConstantFoldingPass(GraphPass):
    """Perform constant folding across the entire graph by pre-evaluating static subgraphs."""

    def __init__(self, max_size_mb: float = 10.0):
        """Initialize the optimization pass."""
        """Initialize the optimization pass."""
        super().__init__()
        self.max_size_mb = max_size_mb

    """
    Evaluates deterministic operations at compile-time when all of their
    inputs are statically known constants or initializers.
    """

    def run(self, graph: Graph) -> bool:
        """Implement the run method or operation."""
        changed = False
        while True:
            iteration_changed = self._run_once(graph)

            # Recurse into subgraphs
            for node in graph.nodes:
                for attr_name, attr in node.attributes.items():
                    if hasattr(attr, "attr_type") and attr.attr_type == "GRAPH":
                        sub_changed = self.run(attr.value)
                        if sub_changed:
                            iteration_changed = True

            if not iteration_changed:
                break
            changed = True
        return changed

    def _run_once(self, graph: Graph) -> bool:
        """Implement the _run_once method or operation."""
        changed = False
        known_values = {}
        for init_name in graph.initializers:
            if (
                init_name in graph.tensors
                and getattr(graph.tensors[init_name], "data", None) is not None
            ):
                val = _tensor_to_numpy(graph.tensors[init_name])
                if val is not None:
                    known_values[init_name] = val

        for node in graph.nodes:
            if node.op_type == "Constant":
                val = node.attributes.get("value")
                if isinstance(val, Tensor) and val.data is not None:
                    np_val = _tensor_to_numpy(val)
                    if np_val is not None:
                        known_values[node.outputs[0]] = np_val
                elif hasattr(val, "attr_type") and val.attr_type == "TENSOR":
                    from onnx9000.core.parser.core import parse_tensor_proto

                    t = parse_tensor_proto(val.value)
                    if t.data is not None:
                        np_val = _tensor_to_numpy(t)
                        if np_val is not None:
                            known_values[node.outputs[0]] = np_val
                elif hasattr(val, "ndim"):
                    known_values[node.outputs[0]] = val
                elif "value_float" in node.attributes:
                    attr_val = node.attributes["value_float"]
                    known_values[node.outputs[0]] = np.array(
                        attr_val.value if hasattr(attr_val, "value") else attr_val, dtype=np.float32
                    )
                elif "value_int" in node.attributes:
                    attr_val = node.attributes["value_int"]
                    known_values[node.outputs[0]] = np.array(
                        attr_val.value if hasattr(attr_val, "value") else attr_val, dtype=np.int64
                    )
        new_nodes = []
        for node in graph.nodes:
            if node.op_type == "Constant":
                new_nodes.append(node)
                continue
            all_known = True
            input_vals = []
            for inp in node.inputs:
                if inp in known_values:
                    input_vals.append(known_values[inp])
                elif inp == "":
                    input_vals.append(None)
                else:
                    all_known = False
                    break

            if not all_known and node.op_type == "Shape":
                inp_tensor = None
                for inp in graph.inputs:
                    if getattr(inp, "name", inp) == node.inputs[0]:
                        inp_tensor = inp
                        break
                if not inp_tensor and node.inputs[0] in graph.tensors:
                    inp_tensor = graph.tensors[node.inputs[0]]

                if inp_tensor and hasattr(inp_tensor, "shape") and len(inp_tensor.shape) > 0:
                    from onnx9000.core.ir import DynamicDim

                    if not any(
                        isinstance(d, DynamicDim) or isinstance(d, str) for d in inp_tensor.shape
                    ):
                        shape_arr = np.array(
                            [
                                int(d.value if isinstance(d, DynamicDim) else d)
                                for d in inp_tensor.shape
                            ],
                            dtype=np.int64,
                        )
                        input_vals = [np.zeros(shape_arr.tolist(), dtype=np.float32)]
                        all_known = True
                    else:
                        logger.debug(
                            f"Branch blocked by dynamic Shape at {node.name} (input shape contains dynamic axes)."
                        )

            is_custom = getattr(node, "domain", "") not in ("", "ai.onnx")
            if is_custom:
                logger.warning(
                    f"Simplifier encountered unsupported CustomOp domain '{getattr(node, 'domain', '')}' at {node.name}. Breaking cascade."
                )

            folded = False
            is_custom = getattr(node, "domain", "") not in ("", "ai.onnx")
            if (
                all_known
                and not is_custom
                and node.op_type
                not in (
                    "RandomUniform",
                    "RandomNormal",
                    "RandomUniformLike",
                    "RandomNormalLike",
                    "Multinomial",
                )
            ):
                try:
                    result = self._evaluate_node(node.op_type, input_vals, node.attributes, node)
                    print(
                        "EVALUATED:",
                        node.op_type,
                        result is not None,
                        "size:",
                        result.nbytes if hasattr(result, "nbytes") else None,
                    )
                    print(
                        "EVALUATED:",
                        node.op_type,
                        result is not None,
                        "size:",
                        result.nbytes if hasattr(result, "nbytes") else None,
                    )
                    print(
                        "EVALUATED:",
                        node.op_type,
                        result is not None,
                        "size:",
                        result.nbytes if hasattr(result, "nbytes") else None,
                    )
                    print(
                        "EVALUATED:",
                        node.op_type,
                        result is not None,
                        "size:",
                        result.nbytes if hasattr(result, "nbytes") else None,
                    )
                    print(
                        "EVALUATED:",
                        node.op_type,
                        result is not None,
                        "size:",
                        result.nbytes if hasattr(result, "nbytes") else None,
                    )
                    print(
                        "EVALUATED:",
                        node.op_type,
                        result is not None,
                        "size:",
                        result.nbytes if hasattr(result, "nbytes") else None,
                    )
                    print(
                        "EVALUATED:",
                        node.op_type,
                        result is not None,
                        "size:",
                        result.nbytes if hasattr(result, "nbytes") else None,
                    )
                    print(
                        "EVALUATED:",
                        node.op_type,
                        result is not None,
                        "size:",
                        result.nbytes if hasattr(result, "nbytes") else None,
                    )
                    if result is not None:

                        def _has_nan_inf(val):
                            if isinstance(val, np.ndarray):
                                if np.issubdtype(val.dtype, np.number):
                                    return np.any(np.isnan(val)) or np.any(np.isinf(val))
                                return False
                            elif isinstance(val, float):
                                return np.isnan(val) or np.isinf(val)
                            elif isinstance(val, list):
                                return any(_has_nan_inf(v) for v in val)
                            return False

                        if _has_nan_inf(result):
                            logger.warning(
                                f"Skipping constant folding at {node.name} due to NaN/Inf generation."
                            )
                            continue
                        if isinstance(result, list):
                            size = sum(r.nbytes for r in result)
                        else:
                            size = result.nbytes
                        if size <= self.max_size_mb * 1024 * 1024:
                            if isinstance(result, list):
                                if len(result) > len(node.outputs):
                                    raise IndexError("Split result length exceeds node outputs")
                                for idx, res in enumerate(result):
                                    const_node = Node(
                                        op_type="Constant",
                                        inputs=[],
                                        outputs=[node.outputs[idx]],
                                        attributes={"value": res},
                                        name=f"folded_{node.name}_{idx}",
                                    )
                                    new_nodes.append(const_node)
                                    known_values[node.outputs[idx]] = res
                            else:
                                const_node = Node(
                                    op_type="Constant",
                                    inputs=[],
                                    outputs=list(node.outputs),
                                    attributes={"value": result},
                                    name=f"folded_{node.name}",
                                )
                                new_nodes.append(const_node)
                                known_values[node.outputs[0]] = result
                            logger.info(f"Folded node {node.name} ({node.op_type})")
                            folded = True
                            changed = True
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    logger.debug(f"Failed to fold {node.name}: {e}")
            if not folded:
                (new_node, local_changed) = self._partial_fold(node, known_values)
                if local_changed:
                    changed = True
                    if new_node is not None:
                        new_nodes.append(new_node)
                else:
                    new_nodes.append(node)
        graph.nodes = new_nodes
        return changed

    def _evaluate_node(self, op_type: str, inputs: list, attrs: dict, node=None):
        """Implement the _evaluate_node method or operation."""
        if op_type == "Add":
            return np.add(inputs[0], inputs[1])
        if op_type == "Sub":
            return np.subtract(inputs[0], inputs[1])
        if op_type == "Mul":
            return np.multiply(inputs[0], inputs[1])
        if op_type == "Div":
            if np.any(inputs[1] == 0):
                logger.warning("Skipping Div constant folding due to division by zero.")
                raise ValueError(
                    "Div by zero"
                )  # Caught by auto-fallback in simplify if implemented, else caught here
            return np.divide(inputs[0], inputs[1])
        if op_type == "Pow":
            return np.power(inputs[0], inputs[1])
        if op_type == "Abs":
            return np.abs(inputs[0])
        if op_type == "Exp":
            return np.exp(inputs[0])
        if op_type == "Log":
            return np.log(inputs[0])
        if op_type == "Sqrt":
            return np.sqrt(inputs[0])
        if op_type == "Sin":
            return np.sin(inputs[0])
        if op_type == "Cos":
            return np.cos(inputs[0])
        if op_type == "Tan":
            return np.tan(inputs[0])
        if op_type == "Asin":
            return np.arcsin(inputs[0])
        if op_type == "Acos":
            return np.arccos(inputs[0])
        if op_type == "Atan":
            return np.arctan(inputs[0])
        if op_type == "Sinh":
            return np.sinh(inputs[0])
        if op_type == "Cosh":
            return np.cosh(inputs[0])
        if op_type == "Tanh":
            return np.tanh(inputs[0])
        if op_type == "Neg":
            return np.negative(inputs[0])
        if op_type == "Sign":
            return np.sign(inputs[0])
        if op_type == "Ceil":
            return np.ceil(inputs[0])
        if op_type == "Floor":
            return np.floor(inputs[0])
        if op_type == "Round":
            return np.round(inputs[0])
        if op_type == "Mod":
            return np.mod(inputs[0], inputs[1])
        if op_type == "Max":
            return np.maximum.reduce(inputs) if len(inputs) > 1 else inputs[0]
        if op_type == "Min":
            return np.minimum.reduce(inputs) if len(inputs) > 1 else inputs[0]
        if op_type == "Clip":
            min_val = inputs[1] if len(inputs) > 1 and inputs[1] is not None else None
            max_val = inputs[2] if len(inputs) > 2 and inputs[2] is not None else None
            return np.clip(inputs[0], min_val, max_val)
        if op_type == "And":
            return np.logical_and(inputs[0], inputs[1])
        if op_type == "Or":
            return np.logical_or(inputs[0], inputs[1])
        if op_type == "Not":
            return np.logical_not(inputs[0])
        if op_type == "Xor":
            return np.logical_xor(inputs[0], inputs[1])
        if op_type == "Equal":
            return np.equal(inputs[0], inputs[1])
        if op_type == "Greater":
            return np.greater(inputs[0], inputs[1])
        if op_type == "Less":
            return np.less(inputs[0], inputs[1])
        if op_type == "GreaterOrEqual":
            return np.greater_equal(inputs[0], inputs[1])
        if op_type == "LessOrEqual":
            return np.less_equal(inputs[0], inputs[1])
        if op_type == "BitShift":
            direction = attrs.get("direction", "")
            if direction == "RIGHT":
                return np.right_shift(inputs[0], inputs[1])
            elif direction == "LEFT":
                return np.left_shift(inputs[0], inputs[1])
            return None
        if op_type == "BitwiseAnd":
            return np.bitwise_and(inputs[0], inputs[1])
        if op_type == "BitwiseOr":
            return np.bitwise_or(inputs[0], inputs[1])
        if op_type == "BitwiseNot":
            return np.bitwise_not(inputs[0])
        if op_type == "BitwiseXor":
            return np.bitwise_xor(inputs[0], inputs[1])
        if op_type == "IsInf":
            detect_positive = attrs.get("detect_positive", 1)
            detect_negative = attrs.get("detect_negative", 1)
            if detect_positive and detect_negative:
                return np.isinf(inputs[0])
            elif detect_positive:
                return np.isposinf(inputs[0])
            elif detect_negative:
                return np.isneginf(inputs[0])
            return np.zeros_like(inputs[0], dtype=bool)
        if op_type == "IsNaN":
            return np.isnan(inputs[0])
        if op_type == "Erf":
            from scipy.special import erf

            return erf(inputs[0])
        if op_type == "Relu":
            return np.maximum(inputs[0], 0)
        if op_type == "Sigmoid":
            return 1 / (1 + np.exp(-inputs[0]))
        if op_type == "ReduceSum":
            axes = attrs.get("axes")
            keepdims = attrs.get("keepdims", 1)
            if axes is not None:
                axes = tuple(axes)
            elif len(inputs) > 1 and inputs[1] is not None:
                axes = tuple(inputs[1].tolist())
            return np.sum(inputs[0], axis=axes, keepdims=bool(keepdims))
        if op_type == "ReduceMean":
            axes = attrs.get("axes")
            keepdims = attrs.get("keepdims", 1)
            if axes is not None:
                axes = tuple(axes)
            elif len(inputs) > 1 and inputs[1] is not None:
                axes = tuple(inputs[1].tolist())
            return np.mean(inputs[0], axis=axes, keepdims=bool(keepdims))
        if op_type == "ReduceMax":
            axes = attrs.get("axes")
            keepdims = attrs.get("keepdims", 1)
            if axes is not None:
                axes = tuple(axes)
            elif len(inputs) > 1 and inputs[1] is not None:
                axes = tuple(inputs[1].tolist())
            return np.max(inputs[0], axis=axes, keepdims=bool(keepdims))
        if op_type == "ReduceMin":
            axes = attrs.get("axes")
            keepdims = attrs.get("keepdims", 1)
            if axes is not None:
                axes = tuple(axes)
            elif len(inputs) > 1 and inputs[1] is not None:
                axes = tuple(inputs[1].tolist())
            return np.min(inputs[0], axis=axes, keepdims=bool(keepdims))
        if op_type == "ReduceProd":
            axes = attrs.get("axes")
            keepdims = attrs.get("keepdims", 1)
            if axes is not None:
                axes = tuple(axes)
            elif len(inputs) > 1 and inputs[1] is not None:
                axes = tuple(inputs[1].tolist())
            return np.prod(inputs[0], axis=axes, keepdims=bool(keepdims))
        if op_type == "ReduceL1":
            axes = attrs.get("axes")
            keepdims = attrs.get("keepdims", 1)
            if axes is not None:
                axes = tuple(axes)
            elif len(inputs) > 1 and inputs[1] is not None:
                axes = tuple(inputs[1].tolist())
            return np.sum(np.abs(inputs[0]), axis=axes, keepdims=bool(keepdims))
        if op_type == "ReduceL2":
            axes = attrs.get("axes")
            keepdims = attrs.get("keepdims", 1)
            if axes is not None:
                axes = tuple(axes)
            elif len(inputs) > 1 and inputs[1] is not None:
                axes = tuple(inputs[1].tolist())
            return np.sqrt(np.sum(np.square(inputs[0]), axis=axes, keepdims=bool(keepdims)))
        if op_type == "ReduceLogSum":
            axes = attrs.get("axes")
            keepdims = attrs.get("keepdims", 1)
            if axes is not None:
                axes = tuple(axes)
            elif len(inputs) > 1 and inputs[1] is not None:
                axes = tuple(inputs[1].tolist())
            return np.log(np.sum(inputs[0], axis=axes, keepdims=bool(keepdims)))
        if op_type == "ReduceLogSumExp":
            axes = attrs.get("axes")
            keepdims = attrs.get("keepdims", 1)
            if axes is not None:
                axes = tuple(axes)
            elif len(inputs) > 1 and inputs[1] is not None:
                axes = tuple(inputs[1].tolist())
            m = np.max(inputs[0], axis=axes, keepdims=True)
            res = m + np.log(np.sum(np.exp(inputs[0] - m), axis=axes, keepdims=True))
            if not keepdims:
                res = np.squeeze(res, axis=axes)
            return res
        if op_type == "CastLike":
            return inputs[0].astype(inputs[1].dtype)
        if op_type == "Cast":
            to_type = attrs.get("to")
            dtype_map = {
                1: np.float32,
                2: np.uint8,
                3: np.int8,
                4: np.uint16,
                5: np.int16,
                6: np.int32,
                7: np.int64,
                10: np.float16,
                11: np.float64,
                9: bool,
            }
            if to_type in dtype_map:
                return inputs[0].astype(dtype_map[to_type])
            return inputs[0]
        if op_type == "Reshape":
            return inputs[0].reshape(inputs[1])
        if op_type == "Transpose":
            return np.transpose(inputs[0], axes=attrs.get("perm"))
        if op_type == "Squeeze":
            axes = attrs.get("axes")
            if len(inputs) > 1 and inputs[1] is not None:
                axes = tuple(inputs[1].tolist())
            return np.squeeze(inputs[0], axis=tuple(axes) if axes is not None else None)
        if op_type == "Unsqueeze":
            axes = attrs.get("axes")
            if len(inputs) > 1 and inputs[1] is not None:
                axes = tuple(inputs[1].tolist())
            res = inputs[0]
            for ax in sorted(axes):
                res = np.expand_dims(res, axis=ax)
            return res
        if op_type == "Flatten":
            axis = attrs.get("axis", 1)
            return inputs[0].reshape((np.prod(inputs[0].shape[:axis], dtype=int), -1))
        if op_type == "Concat":
            axis_attr = attrs.get("axis", 0)
            axis = int(getattr(axis_attr, "value", axis_attr))
            clean_inputs = []
            for i in inputs:
                if isinstance(i, (int, float, bool)):
                    clean_inputs.append(np.array([i]))
                else:
                    clean_inputs.append(np.atleast_1d(i))
            return np.concatenate(clean_inputs, axis=axis)
        if op_type == "Slice":
            (data, starts, ends) = (inputs[0], inputs[1], inputs[2])
            axes = (
                inputs[3] if len(inputs) > 3 and inputs[3] is not None else list(range(len(starts)))
            )
            steps = inputs[4] if len(inputs) > 4 and inputs[4] is not None else [1] * len(starts)
            slices = [slice(None)] * data.ndim
            for i, ax in enumerate(axes):
                slices[ax] = slice(starts[i], ends[i], steps[i])
            return data[tuple(slices)]
        if op_type == "BatchNormalization":
            x, scale, b, mean, var = inputs
            epsilon = attrs.get("epsilon", 1e-05)
            ndim = x.ndim
            target_shape = [1] * ndim
            if ndim > 1:
                target_shape[1] = -1
            else:
                target_shape[0] = -1
            scale = scale.reshape(target_shape)
            b = b.reshape(target_shape)
            mean = mean.reshape(target_shape)
            var = var.reshape(target_shape)
            return scale * (x - mean) / np.sqrt(var + epsilon) + b
        if op_type == "Split":
            axis = attrs.get("axis", 0)
            if len(inputs) > 1 and inputs[1] is not None:
                split = inputs[1].tolist()
                indices = np.cumsum(split)[:-1]
                return np.split(inputs[0], indices, axis=axis)
            else:
                num_outputs = len(node.outputs)
                return np.split(inputs[0], num_outputs, axis=axis)
        if op_type == "Expand":
            return np.broadcast_to(inputs[0], tuple(inputs[1]))
        if op_type == "Tile":
            return np.tile(inputs[0], tuple(inputs[1]))
        if op_type == "Pad":
            pads = inputs[1]
            mode = attrs.get("mode", "constant")
            value = inputs[2] if len(inputs) > 2 and inputs[2] is not None else 0
            if isinstance(value, np.ndarray):
                value = value.item() if value.size == 1 else value[0]
            ndims = inputs[0].ndim
            pad_width = [(pads[i], pads[i + ndims]) for i in range(ndims)]
            if mode == "constant":
                return np.pad(inputs[0], pad_width, mode=mode, constant_values=value)
            elif mode == "reflect":
                return np.pad(inputs[0], pad_width, mode="reflect")
            elif mode == "edge":
                return np.pad(inputs[0], pad_width, mode="edge")
            return None
        if op_type == "ConstantOfShape":
            shape = tuple(inputs[0].tolist()) if len(inputs) > 0 else ()
            val = attrs.get("value", np.array([0], dtype=np.float32))
            if isinstance(val, Tensor):
                val = val.data
            elif isinstance(val, list):
                val = np.array(val)
            val = val.item() if val.size == 1 else val[0]
            return np.full(
                shape, val, dtype=type(val) if not isinstance(val, float) else np.float32
            )
        if op_type == "Where":
            return np.where(inputs[0], inputs[1], inputs[2])
        if op_type == "CumSum":
            axis = inputs[1].item() if inputs[1].size == 1 else inputs[1][0]
            exclusive = attrs.get("exclusive", 0)
            reverse = attrs.get("reverse", 0)
            data = inputs[0]
            if reverse:
                data = np.flip(data, axis=axis)
            res = np.cumsum(data, axis=axis)
            if exclusive:
                shape = list(data.shape)
                shape[axis] = 1
                zeros = np.zeros(shape, dtype=data.dtype)
                res = np.concatenate(
                    (zeros, res.take(indices=range(res.shape[axis] - 1), axis=axis)), axis=axis
                )
            if reverse:
                res = np.flip(res, axis=axis)
            return res
        if op_type == "Trilu":
            k = inputs[1].item() if len(inputs) > 1 and inputs[1] is not None else 0
            upper = attrs.get("upper", 1)
            if upper:
                return np.triu(inputs[0], k=k)
            else:
                return np.tril(inputs[0], k=k)
        if op_type in ("MatMul", "Gemm"):
            if op_type == "MatMul":
                return np.matmul(inputs[0], inputs[1])
            else:
                transA = attrs.get("transA", 0)
                transB = attrs.get("transB", 0)
                alpha = attrs.get("alpha", 1.0)
                beta = attrs.get("beta", 1.0)
                A = inputs[0].T if transA else inputs[0]
                B = inputs[1].T if transB else inputs[1]
                res = alpha * np.matmul(A, B)
                if len(inputs) > 2 and inputs[2] is not None:
                    res += beta * inputs[2]
                return res
        if op_type == "Conv":
            return None
        if op_type == "Gather":
            return np.take(inputs[0], inputs[1], axis=attrs.get("axis", 0))
        if op_type == "GatherND":
            data = inputs[0]
            indices = inputs[1]
            batch_dims_attr = attrs.get("batch_dims", 0)
            if hasattr(batch_dims_attr, "value"):
                batch_dims = int(batch_dims_attr.value)
            else:
                batch_dims = int(batch_dims_attr)

            # Naive pure python/numpy implementation of GatherND
            # This works well for constant folding sizes

            data_shape = data.shape
            indices_shape = indices.shape

            # The innermost dim of indices is the number of indices we gather per element
            k = indices_shape[-1]

            # Output shape:
            # batch_dims + (indices_shape[:-1] - batch_dims) + data_shape[batch_dims + k:]
            out_shape = (
                data_shape[:batch_dims]
                + indices_shape[batch_dims:-1]
                + data_shape[batch_dims + k :]
            )

            # Flattening to make gather simple
            res = np.empty(out_shape, dtype=data.dtype)

            # Very slow but robust pure numpy iterator for constant folding
            it = np.nditer(np.zeros(indices_shape[:-1]), flags=["multi_index"])
            for _ in it:
                idx = it.multi_index

                # Batch dims
                b_idx = idx[:batch_dims]

                # Gather coords
                gather_idx = tuple(indices[idx])

                # Result idx
                out_idx = b_idx + idx[batch_dims:]

                res[out_idx] = data[b_idx + gather_idx]

            return res

        if op_type == "ScatterND":
            data = inputs[0].copy()
            indices = inputs[1]
            updates = inputs[2]
            reduction_attr = attrs.get("reduction", "none")
            if hasattr(reduction_attr, "value"):
                reduction = str(reduction_attr.value)
            else:
                reduction = str(reduction_attr)

            indices_shape = indices.shape
            k = indices_shape[-1]

            it = np.nditer(np.zeros(indices_shape[:-1]), flags=["multi_index"])
            for _ in it:
                idx = it.multi_index
                gather_idx = tuple(indices[idx])
                if reduction == "add":
                    data[gather_idx] += updates[idx]
                elif reduction == "mul":
                    data[gather_idx] *= updates[idx]
                elif reduction == "max":
                    data[gather_idx] = np.maximum(data[gather_idx], updates[idx])
                elif reduction == "min":
                    data[gather_idx] = np.minimum(data[gather_idx], updates[idx])
                else:
                    data[gather_idx] = updates[idx]

            return data

        if op_type == "SequenceConstruct":
            # ONNX Sequences are lists of tensors
            return list(inputs)

        if op_type == "SequenceAt":
            seq = inputs[0]
            idx = inputs[1]
            if isinstance(idx, np.ndarray):
                idx = idx.item() if idx.size == 1 else idx[0]
            idx = int(idx)

            if isinstance(seq, list) and -len(seq) <= idx < len(seq):
                return seq[idx]
            return None

        if op_type == "SplitToSequence":
            if len(inputs) > 1 and inputs[1] is not None:
                split = inputs[1]
                if isinstance(split, np.ndarray) and split.size == 1:
                    split = split.item()
                if isinstance(split, int):
                    axis_attr = attrs.get("axis", 0)
                    if hasattr(axis_attr, "value"):
                        axis = int(axis_attr.value)
                    else:
                        axis = int(axis_attr)

                    # Prevent massive arrays
                    dim_size = inputs[0].shape[axis]
                    if dim_size / split > 10000:
                        logger.warning(
                            f"Skipping SplitToSequence: sequence too large ({dim_size / split} items)."
                        )
                        return None

                    res = []
                    for i in range(0, dim_size, split):
                        sl = [slice(None)] * inputs[0].ndim
                        sl[axis] = slice(i, min(i + split, dim_size))
                        res.append(inputs[0][tuple(sl)])
                    return res
            return None

        if op_type == "GatherElements":
            axis = attrs.get("axis", 0)
            return np.take_along_axis(inputs[0], inputs[1], axis=axis)
        if op_type in ("Scatter", "ScatterElements"):
            axis = attrs.get("axis", 0)
            res = inputs[0].copy()
            reduction = attrs.get("reduction", "none")
            if reduction == "none":
                np.put_along_axis(res, inputs[1], inputs[2], axis)
            return res
        if op_type == "Shape":
            return np.array(inputs[0].shape, dtype=np.int64)
        if op_type == "Size":
            return np.array(inputs[0].size, dtype=np.int64)
        if op_type == "NonMaxSuppression":
            boxes = inputs[0]
            scores = inputs[1]
            max_output_boxes_per_class = (
                int(inputs[2].item() if inputs[2].size == 1 else inputs[2][0])
                if len(inputs) > 2 and inputs[2] is not None
                else 0
            )
            iou_threshold = (
                float(inputs[3].item() if inputs[3].size == 1 else inputs[3][0])
                if len(inputs) > 3 and inputs[3] is not None
                else 0.0
            )
            score_threshold = (
                float(inputs[4].item() if inputs[4].size == 1 else inputs[4][0])
                if len(inputs) > 4 and inputs[4] is not None
                else -np.inf
            )

            center_point_box = attrs.get(
                "center_point_box", Attribute("center_point_box", "INT", 0)
            ).value

            num_batches = boxes.shape[0]
            num_classes = scores.shape[1]
            boxes.shape[1]

            selected_indices = []

            for batch_index in range(num_batches):
                for class_index in range(num_classes):
                    class_scores = scores[batch_index, class_index]

                    # Filter by score
                    valid_mask = class_scores >= score_threshold
                    valid_scores = class_scores[valid_mask]
                    valid_boxes = boxes[batch_index, valid_mask]
                    valid_indices = np.where(valid_mask)[0]

                    # Sort by score descending
                    sorted_order = np.argsort(valid_scores)[::-1]
                    valid_boxes = valid_boxes[sorted_order]
                    valid_indices = valid_indices[sorted_order]

                    selected_for_class = []
                    for i in range(len(valid_indices)):
                        if len(selected_for_class) >= max_output_boxes_per_class:
                            break
                        box1 = valid_boxes[i]
                        keep = True
                        for j in selected_for_class:
                            box2 = valid_boxes[j]

                            # calculate IoU
                            if center_point_box == 0:  # [y1, x1, y2, x2]
                                y1_int = max(box1[0], box2[0])
                                x1_int = max(box1[1], box2[1])
                                y2_int = min(box1[2], box2[2])
                                x2_int = min(box1[3], box2[3])
                            else:  # [x_center, y_center, width, height]
                                box1_y1, box1_x1 = box1[1] - box1[3] / 2, box1[0] - box1[2] / 2
                                box1_y2, box1_x2 = box1[1] + box1[3] / 2, box1[0] + box1[2] / 2
                                box2_y1, box2_x1 = box2[1] - box2[3] / 2, box2[0] - box2[2] / 2
                                box2_y2, box2_x2 = box2[1] + box2[3] / 2, box2[0] + box2[2] / 2

                                y1_int = max(box1_y1, box2_y1)
                                x1_int = max(box1_x1, box2_x1)
                                y2_int = min(box1_y2, box2_y2)
                                x2_int = min(box1_x2, box2_x2)

                            intersect_area = max(0.0, y2_int - y1_int) * max(0.0, x2_int - x1_int)

                            if center_point_box == 0:
                                area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                                area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                            else:
                                area1 = box1[2] * box1[3]
                                area2 = box2[2] * box2[3]

                            union_area = area1 + area2 - intersect_area
                            iou = intersect_area / union_area if union_area > 0 else 0.0

                            if iou > iou_threshold:
                                keep = False
                                break

                        if keep:
                            selected_for_class.append(i)
                            selected_indices.append([batch_index, class_index, valid_indices[i]])

            if not selected_indices:
                return np.empty((0, 3), dtype=np.int64)
            return np.array(selected_indices, dtype=np.int64)

        if op_type == "NonZero":
            return np.array(np.nonzero(inputs[0]), dtype=np.int64)
        return None

    def _partial_fold(self, node: Node, known_values: dict) -> tuple:
        """Implement the _partial_fold method or operation."""
        if node.op_type == "Add":
            if node.inputs[1] in known_values and np.all(known_values[node.inputs[1]] == 0):
                (node.op_type, node.inputs) = ("Identity", [node.inputs[0]])
                return (node, True)
            if node.inputs[0] in known_values and np.all(known_values[node.inputs[0]] == 0):
                (node.op_type, node.inputs) = ("Identity", [node.inputs[1]])
                return (node, True)
        if node.op_type == "Mul":
            if node.inputs[1] in known_values and np.all(known_values[node.inputs[1]] == 1):
                (node.op_type, node.inputs) = ("Identity", [node.inputs[0]])
                return (node, True)
            if node.inputs[0] in known_values and np.all(known_values[node.inputs[0]] == 1):
                (node.op_type, node.inputs) = ("Identity", [node.inputs[1]])
                return (node, True)
            if node.inputs[1] in known_values and np.all(known_values[node.inputs[1]] == 0):
                return (
                    Node(
                        "Constant",
                        [],
                        node.outputs,
                        {"value": np.zeros_like(known_values[node.inputs[1]])},
                        f"partial_{node.name}",
                    ),
                    True,
                )
        if node.op_type == "Sub":
            if node.inputs[1] in known_values and np.all(known_values[node.inputs[1]] == 0):
                (node.op_type, node.inputs) = ("Identity", [node.inputs[0]])
                return (node, True)
            if node.inputs[0] == node.inputs[1]:
                return (
                    Node(
                        "Constant",
                        [],
                        node.outputs,
                        {"value": np.zeros(1, dtype=np.float32)},
                        f"partial_{node.name}",
                    ),
                    True,
                )
        if node.op_type == "Where" and node.inputs[0] in known_values:
            cond = known_values[node.inputs[0]]
            if isinstance(cond, np.ndarray) and (cond.size == 1 or np.all(cond == cond.flat[0])):
                if np.all(cond):
                    (node.op_type, node.inputs) = ("Identity", [node.inputs[1]])
                else:
                    (node.op_type, node.inputs) = ("Identity", [node.inputs[2]])
                return (node, True)
        if node.op_type == "Div":
            if node.inputs[1] in known_values and np.all(known_values[node.inputs[1]] == 1):
                (node.op_type, node.inputs) = ("Identity", [node.inputs[0]])
                return (node, True)
            if node.inputs[0] == node.inputs[1]:
                return (
                    Node(
                        "Constant",
                        [],
                        node.outputs,
                        {"value": np.ones(1, dtype=np.float32)},
                        f"partial_{node.name}",
                    ),
                    True,
                )
        if node.op_type == "Pow":
            if node.inputs[1] in known_values and np.all(known_values[node.inputs[1]] == 1):
                (node.op_type, node.inputs) = ("Identity", [node.inputs[0]])
                return (node, True)
            if node.inputs[1] in known_values and np.all(known_values[node.inputs[1]] == 0):
                return (
                    Node(
                        "Constant",
                        [],
                        node.outputs,
                        {"value": np.ones(1, dtype=np.float32)},
                        f"partial_{node.name}",
                    ),
                    True,
                )
        return (node, False)


def constant_folding(graph: Graph, max_size_mb: float = 10.0) -> None:
    """Implement the constant_folding method or operation."""
    ConstantFoldingPass(max_size_mb).run(graph)
