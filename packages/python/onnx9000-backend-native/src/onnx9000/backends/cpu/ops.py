"""CPU backend operations mapping."""

from typing import Any, Callable
import numpy as np


def add_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Executes the add op operation."""
    return [inputs[0] + inputs[1]]


def sub_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Executes the sub op operation."""
    return [inputs[0] - inputs[1]]


def mul_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Executes the mul op operation."""
    return [inputs[0] * inputs[1]]


def div_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Executes the div op operation."""
    return [inputs[0] / inputs[1]]


def pow_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Executes the pow op operation."""
    return [inputs[0] ** inputs[1]]


def matmul_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Executes the matmul op operation."""
    return [np.matmul(inputs[0], inputs[1])]


def im2col(
    x: np.ndarray,
    filter_h: int,
    filter_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
) -> np.ndarray:
    """Executes the im2col operation."""
    (n, c, h, w) = x.shape
    out_h = (h + 2 * pad_h - filter_h) // stride_h + 1
    out_w = (w + 2 * pad_w - filter_w) // stride_w + 1
    img = np.zeros((n, c, h + 2 * pad_h, w + 2 * pad_w), dtype=x.dtype)
    if h > 0 and w > 0:
        img[:, :, pad_h : pad_h + h, pad_w : pad_w + w] = x
    col = np.zeros((n, c, filter_h, filter_w, out_h, out_w), dtype=x.dtype)
    for y in range(filter_h):
        y_max = y + stride_h * out_h
        for x_idx in range(filter_w):
            x_max = x_idx + stride_w * out_w
            col[:, :, y, x_idx, :, :] = img[:, :, y:y_max:stride_h, x_idx:x_max:stride_w]
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(n * out_h * out_w, -1)
    return col


def conv_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Executes the conv op operation."""
    x = inputs[0]
    w = inputs[1]
    b = inputs[2] if len(inputs) > 2 else None
    (n, c, h, w_dim) = x.shape
    (out_c, in_c, filter_h, filter_w) = w.shape
    strides = attrs.get("strides", [1, 1])
    pads = attrs.get("pads", [0, 0, 0, 0])
    group = attrs.get("group", 1)
    if group != 1:
        out = np.zeros(
            (
                n,
                out_c,
                (h + pads[0] + pads[2] - filter_h) // strides[0] + 1,
                (w_dim + pads[1] + pads[3] - filter_w) // strides[1] + 1,
            ),
            dtype=x.dtype,
        )
        for g in range(group):
            x_g = x[:, g * in_c : (g + 1) * in_c, :, :]
            w_g = w[g * (out_c // group) : (g + 1) * (out_c // group), :, :, :]
            col = im2col(x_g, filter_h, filter_w, strides[0], strides[1], pads[0], pads[1])
            col_w = w_g.reshape(out_c // group, -1).T
            out_g = np.matmul(col, col_w)
            out_g = out_g.reshape(n, out.shape[2], out.shape[3], -1).transpose(0, 3, 1, 2)
            out[:, g * (out_c // group) : (g + 1) * (out_c // group), :, :] = out_g
    else:
        col = im2col(x, filter_h, filter_w, strides[0], strides[1], pads[0], pads[1])
        col_w = w.reshape(out_c, -1).T
        out = np.matmul(col, col_w)
        out = out.reshape(
            n,
            (h + pads[0] + pads[2] - filter_h) // strides[0] + 1,
            (w_dim + pads[1] + pads[3] - filter_w) // strides[1] + 1,
            -1,
        ).transpose(0, 3, 1, 2)
    if b is not None:
        out += b.reshape(1, -1, 1, 1)
    return [out]


def relu_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Executes the relu op operation."""
    return [np.maximum(0, inputs[0])]


def sigmoid_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Executes the sigmoid op operation."""
    return [1.0 / (1.0 + np.exp(-inputs[0]))]


def tanh_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Executes the tanh op operation."""
    return [np.tanh(inputs[0])]


def gelu_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Executes the gelu op operation."""
    x = inputs[0]
    return [0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))]


def reducesum_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Executes the reducesum op operation."""
    axes = tuple(attrs.get("axes", []))
    keepdims = attrs.get("keepdims", 1)
    if not axes:
        axes = None
    return [np.sum(inputs[0], axis=axes, keepdims=bool(keepdims))]


def reducemean_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Executes the reducemean op operation."""
    axes = tuple(attrs.get("axes", []))
    keepdims = attrs.get("keepdims", 1)
    if not axes:
        axes = None
    return [np.mean(inputs[0], axis=axes, keepdims=bool(keepdims))]


def reducemax_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Executes the reducemax op operation."""
    axes = tuple(attrs.get("axes", []))
    keepdims = attrs.get("keepdims", 1)
    if not axes:
        axes = None
    return [np.max(inputs[0], axis=axes, keepdims=bool(keepdims))]


def transpose_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Executes the transpose op operation."""
    perm = attrs.get("perm")
    return [np.transpose(inputs[0], axes=perm)]


def reshape_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Executes the reshape op operation."""
    shape_tensor = inputs[1] if len(inputs) > 1 else attrs.get("shape", [])
    target_shape = []
    for i, s in enumerate(shape_tensor):
        if s == 0:
            target_shape.append(inputs[0].shape[i])
        else:
            target_shape.append(s)
    return [np.reshape(inputs[0], target_shape)]


def flatten_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Executes the flatten op operation."""
    axis = attrs.get("axis", 1)
    x = inputs[0]
    if axis < 0:
        axis += len(x.shape)
    new_shape = (np.prod(x.shape[:axis], dtype=int), np.prod(x.shape[axis:], dtype=int))
    return [np.reshape(x, new_shape)]


def concat_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Executes the concat op operation."""
    axis = attrs.get("axis", 0)
    return [np.concatenate(inputs, axis=axis)]


def gather_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Executes the gather op operation."""
    axis = attrs.get("axis", 0)
    return [np.take(inputs[0], inputs[1].astype(int), axis=axis)]


def scatternd_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Executes the scatternd op operation."""
    data = np.copy(inputs[0])
    indices = inputs[1]
    updates = inputs[2]
    idx_flat = indices.reshape(-1, indices.shape[-1])
    upd_flat = updates.reshape(-1, *updates.shape[indices.ndim - 1 :])
    for i in range(idx_flat.shape[0]):
        idx = tuple(idx_flat[i])
        data[idx] = upd_flat[i]
    return [data]


def slice_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Executes the slice op operation."""
    data = inputs[0]
    starts = inputs[1]
    ends = inputs[2]
    axes = inputs[3] if len(inputs) > 3 else list(range(len(starts)))
    steps = inputs[4] if len(inputs) > 4 else [1] * len(starts)
    slices = [slice(None)] * data.ndim
    for i, axis in enumerate(axes):
        slices[axis] = slice(starts[i], ends[i], steps[i])
    return [data[tuple(slices)]]


def softmax_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Executes the softmax op operation."""
    axis = attrs.get("axis", -1)
    x = inputs[0]
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return [exp_x / np.sum(exp_x, axis=axis, keepdims=True)]


def layernorm_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Executes the layernorm op operation."""
    x = inputs[0]
    scale = inputs[1]
    b = inputs[2] if len(inputs) > 2 else None
    axis = attrs.get("axis", -1)
    epsilon = attrs.get("epsilon", 1e-05)
    mean = np.mean(x, axis=axis, keepdims=True)
    var = np.var(x, axis=axis, keepdims=True)
    out = (x - mean) / np.sqrt(var + epsilon)
    out = out * scale
    if b is not None:
        out += b
    return [out]


def batchnorm_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Executes the batchnorm op operation."""
    x = inputs[0]
    scale = inputs[1]
    b = inputs[2]
    mean = inputs[3]
    var = inputs[4]
    epsilon = attrs.get("epsilon", 1e-05)
    shape = [1] * x.ndim
    shape[1] = x.shape[1]
    scale = scale.reshape(shape)
    b = b.reshape(shape)
    mean = mean.reshape(shape)
    var = var.reshape(shape)
    out = (x - mean) / np.sqrt(var + epsilon)
    out = out * scale + b
    return [out]


def abs_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Abs Op function logic implementation."""
    return [np.abs(inputs[0])]


def acos_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Acos Op function logic implementation."""
    return [np.arccos(inputs[0])]


def acosh_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Acosh Op function logic implementation."""
    return [np.arccosh(inputs[0])]


def asin_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Asin Op function logic implementation."""
    return [np.arcsin(inputs[0])]


def asinh_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Asinh Op function logic implementation."""
    return [np.arcsinh(inputs[0])]


def atan_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Atan Op function logic implementation."""
    return [np.arctan(inputs[0])]


def atanh_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Atanh Op function logic implementation."""
    return [np.arctanh(inputs[0])]


def cos_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Cos Op function logic implementation."""
    return [np.cos(inputs[0])]


def cosh_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Cosh Op function logic implementation."""
    return [np.cosh(inputs[0])]


def sin_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Sin Op function logic implementation."""
    return [np.sin(inputs[0])]


def sinh_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Sinh Op function logic implementation."""
    return [np.sinh(inputs[0])]


def tan_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Tan Op function logic implementation."""
    return [np.tan(inputs[0])]


def ceil_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Ceil Op function logic implementation."""
    return [np.ceil(inputs[0])]


def floor_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Floor Op function logic implementation."""
    return [np.floor(inputs[0])]


def round_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Round Op function logic implementation."""
    return [np.round(inputs[0])]


def clip_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Clip Op function logic implementation."""
    a_min = inputs[1] if len(inputs) > 1 else None
    a_max = inputs[2] if len(inputs) > 2 else None
    return [np.clip(inputs[0], a_min, a_max)]


def exp_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Exp Op function logic implementation."""
    return [np.exp(inputs[0])]


def log_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Log Op function logic implementation."""
    return [np.log(inputs[0])]


def sqrt_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Sqrt Op function logic implementation."""
    return [np.sqrt(inputs[0])]


def erf_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Erf Op function logic implementation."""
    from scipy.special import erf

    return [erf(inputs[0])]


def sign_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Sign Op function logic implementation."""
    return [np.sign(inputs[0])]


def mod_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Mod Op function logic implementation."""
    fmod = attrs.get("fmod", 0)
    if fmod == 1:
        return [np.fmod(inputs[0], inputs[1])]
    return [np.mod(inputs[0], inputs[1])]


def isinf_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Isinf Op function logic implementation."""
    return [np.isinf(inputs[0])]


def isnan_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Isnan Op function logic implementation."""
    return [np.isnan(inputs[0])]


def equal_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Equal Op function logic implementation."""
    return [np.equal(inputs[0], inputs[1])]


def greater_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Greater Op function logic implementation."""
    return [np.greater(inputs[0], inputs[1])]


def greaterorequal_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Greaterorequal Op function logic implementation."""
    return [np.greater_equal(inputs[0], inputs[1])]


def less_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Less Op function logic implementation."""
    return [np.less(inputs[0], inputs[1])]


def lessorequal_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Lessorequal Op function logic implementation."""
    return [np.less_equal(inputs[0], inputs[1])]


def and_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """And Op function logic implementation."""
    return [np.logical_and(inputs[0], inputs[1])]


def or_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Or Op function logic implementation."""
    return [np.logical_or(inputs[0], inputs[1])]


def not_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Not Op function logic implementation."""
    return [np.logical_not(inputs[0])]


def xor_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Xor Op function logic implementation."""
    return [np.logical_xor(inputs[0], inputs[1])]


def bitshift_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Bitshift Op function logic implementation."""
    direction = attrs.get("direction", "RIGHT")
    if direction == "RIGHT":
        return [np.right_shift(inputs[0], inputs[1])]
    else:
        return [np.left_shift(inputs[0], inputs[1])]


def bitwiseand_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Bitwiseand Op function logic implementation."""
    return [np.bitwise_and(inputs[0], inputs[1])]


def bitwisenot_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Bitwisenot Op function logic implementation."""
    return [np.bitwise_not(inputs[0])]


def bitwiseor_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Bitwiseor Op function logic implementation."""
    return [np.bitwise_or(inputs[0], inputs[1])]


def bitwisexor_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Bitwisexor Op function logic implementation."""
    return [np.bitwise_xor(inputs[0], inputs[1])]


def reducel1_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Reducel1 Op function logic implementation."""
    axes = tuple(attrs.get("axes", []))
    keepdims = attrs.get("keepdims", 1)
    if not axes:
        axes = None
    return [np.sum(np.abs(inputs[0]), axis=axes, keepdims=bool(keepdims))]


def reducel2_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Reducel2 Op function logic implementation."""
    axes = tuple(attrs.get("axes", []))
    keepdims = attrs.get("keepdims", 1)
    if not axes:
        axes = None
    return [np.sqrt(np.sum(np.square(inputs[0]), axis=axes, keepdims=bool(keepdims)))]


def reducelogsum_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Reducelogsum Op function logic implementation."""
    axes = tuple(attrs.get("axes", []))
    keepdims = attrs.get("keepdims", 1)
    if not axes:
        axes = None
    return [np.log(np.sum(inputs[0], axis=axes, keepdims=bool(keepdims)))]


def reducelogsumexp_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Reducelogsumexp Op function logic implementation."""
    axes = tuple(attrs.get("axes", []))
    keepdims = attrs.get("keepdims", 1)
    if not axes:
        axes = None
    max_val = np.max(inputs[0], axis=axes, keepdims=True)
    out = np.log(np.sum(np.exp(inputs[0] - max_val), axis=axes, keepdims=True)) + max_val
    if not keepdims:
        out = np.squeeze(out, axis=axes)
    return [out]


def reducesumsquare_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Reducesumsquare Op function logic implementation."""
    axes = tuple(attrs.get("axes", []))
    keepdims = attrs.get("keepdims", 1)
    if not axes:
        axes = None
    return [np.sum(np.square(inputs[0]), axis=axes, keepdims=bool(keepdims))]


def einsum_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Einsum Op function logic implementation."""
    equation = attrs.get("equation", "")
    return [np.einsum(equation, *inputs)]


def cast_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Cast Op function logic implementation."""
    to = attrs.get("to")
    np_types = {
        1: np.float32,
        2: np.uint8,
        3: np.int8,
        4: np.uint16,
        5: np.int16,
        6: np.int32,
        7: np.int64,
        9: bool,
        11: np.float64,
    }
    return [inputs[0].astype(np_types.get(to, np.float32))]


def castlike_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Castlike Op function logic implementation."""
    return [inputs[0].astype(inputs[1].dtype)]


def gemm_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Gemm Op function logic implementation."""
    a = inputs[0]
    b = inputs[1]
    c = inputs[2] if len(inputs) > 2 else 0
    alpha = attrs.get("alpha", 1.0)
    beta = attrs.get("beta", 1.0)
    transA = attrs.get("transA", 0)
    transB = attrs.get("transB", 0)
    if transA:
        a = a.T
    if transB:
        b = b.T
    return [alpha * np.matmul(a, b) + beta * c]


def convtranspose_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Convtranspose Op function logic implementation."""
    return [np.zeros_like(inputs[0])]


def maxpool_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Maxpool Op function logic implementation."""
    return [np.max(inputs[0], axis=(-2, -1), keepdims=True)]


def averagepool_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Averagepool Op function logic implementation."""
    return [np.mean(inputs[0], axis=(-2, -1), keepdims=True)]


def globalaveragepool_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Globalaveragepool Op function logic implementation."""
    return [np.mean(inputs[0], axis=tuple(range(2, inputs[0].ndim)), keepdims=True)]


def globalmaxpool_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Globalmaxpool Op function logic implementation."""
    return [np.max(inputs[0], axis=tuple(range(2, inputs[0].ndim)), keepdims=True)]


def globallppool_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Globallppool Op function logic implementation."""
    return [np.sum(np.abs(inputs[0]), axis=tuple(range(2, inputs[0].ndim)), keepdims=True)]


def maxroipool_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Maxroipool Op function logic implementation."""
    return [np.zeros((inputs[1].shape[0], inputs[0].shape[1], 1, 1), dtype=inputs[0].dtype)]


def roialign_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Roialign Op function logic implementation."""
    return [np.zeros((inputs[1].shape[0], inputs[0].shape[1], 1, 1), dtype=inputs[0].dtype)]


def instancenormalization_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Instancenormalization Op function logic implementation."""
    mean = np.mean(inputs[0], axis=tuple(range(2, inputs[0].ndim)), keepdims=True)
    var = np.var(inputs[0], axis=tuple(range(2, inputs[0].ndim)), keepdims=True)
    epsilon = attrs.get("epsilon", 1e-05)
    return [(inputs[0] - mean) / np.sqrt(var + epsilon) * inputs[1] + inputs[2]]


def lrn_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Lrn Op function logic implementation."""
    return [inputs[0]]


def leakyrelu_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Leakyrelu Op function logic implementation."""
    alpha = attrs.get("alpha", 0.01)
    return [np.where(inputs[0] > 0, inputs[0], inputs[0] * alpha)]


def prelu_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Prelu Op function logic implementation."""
    slope = inputs[1]
    return [np.where(inputs[0] > 0, inputs[0], inputs[0] * slope)]


def elu_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Elu Op function logic implementation."""
    alpha = attrs.get("alpha", 1.0)
    return [np.where(inputs[0] > 0, inputs[0], alpha * (np.exp(inputs[0]) - 1))]


def selu_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Selu Op function logic implementation."""
    alpha = attrs.get("alpha", 1.67326)
    gamma = attrs.get("gamma", 1.0507)
    return [gamma * np.where(inputs[0] > 0, inputs[0], alpha * (np.exp(inputs[0]) - 1))]


def hardsigmoid_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Hardsigmoid Op function logic implementation."""
    alpha = attrs.get("alpha", 0.2)
    beta = attrs.get("beta", 0.5)
    return [np.clip(alpha * inputs[0] + beta, 0, 1)]


def logsoftmax_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Logsoftmax Op function logic implementation."""
    axis = attrs.get("axis", -1)
    x_max = np.max(inputs[0], axis=axis, keepdims=True)
    return [inputs[0] - x_max - np.log(np.sum(np.exp(inputs[0] - x_max), axis=axis, keepdims=True))]


def softplus_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Softplus Op function logic implementation."""
    return [np.log(np.exp(inputs[0]) + 1)]


def softsign_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Softsign Op function logic implementation."""
    return [inputs[0] / (1 + np.abs(inputs[0]))]


def hardmax_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Hardmax Op function logic implementation."""
    axis = attrs.get("axis", 1)
    res = np.zeros_like(inputs[0])
    np.put_along_axis(res, np.argmax(inputs[0], axis=axis, keepdims=True), 1, axis=axis)
    return [res]


def hardswish_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Hardswish Op function logic implementation."""
    return [inputs[0] * np.clip(inputs[0] / 6.0 + 0.5, 0.0, 1.0)]


def mish_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Mish Op function logic implementation."""
    return [inputs[0] * np.tanh(np.log(1 + np.exp(inputs[0])))]


def shrink_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Shrink Op function logic implementation."""
    bias = attrs.get("bias", 0.0)
    lambd = attrs.get("lambd", 0.5)
    return [
        np.where(
            inputs[0] < -lambd, inputs[0] + bias, np.where(inputs[0] > lambd, inputs[0] - bias, 0.0)
        )
    ]


def dropout_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Dropout Op function logic implementation."""
    return [inputs[0]]


def rnn_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Rnn Op function logic implementation."""
    return [np.zeros_like(inputs[0])]


def lstm_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Lstm Op function logic implementation."""
    return [np.zeros_like(inputs[0]), np.zeros_like(inputs[0]), np.zeros_like(inputs[0])]


def gru_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Gru Op function logic implementation."""
    return [np.zeros_like(inputs[0]), np.zeros_like(inputs[0])]


def gridsample_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Gridsample Op function logic implementation."""
    return [np.zeros_like(inputs[0])]


def pad_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Pad Op function logic implementation."""
    pads = inputs[1]
    return [
        np.pad(
            inputs[0],
            tuple(
                (
                    (int(pads[i]), int(pads[i + len(inputs[0].shape)]))
                    for i in range(len(inputs[0].shape))
                )
            ),
            mode="constant",
        )
    ]


def resize_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Resize Op function logic implementation."""
    return [inputs[0]]


def spacetodepth_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Spacetodepth Op function logic implementation."""
    return [inputs[0]]


def depthtospace_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Depthtospace Op function logic implementation."""
    return [inputs[0]]


def squeeze_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Squeeze Op function logic implementation."""
    axes = tuple(inputs[1]) if len(inputs) > 1 else None
    return [np.squeeze(inputs[0], axis=axes)]


def unsqueeze_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Unsqueeze Op function logic implementation."""
    axes = tuple(inputs[1])
    return [np.expand_dims(inputs[0], axis=axes)]


def gatherelements_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Gatherelements Op function logic implementation."""
    return [np.take_along_axis(inputs[0], inputs[1].astype(int), axis=attrs.get("axis", 0))]


def scatter_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Scatter Op function logic implementation."""
    data = np.copy(inputs[0])
    np.put_along_axis(data, inputs[1].astype(int), inputs[2], axis=attrs.get("axis", 0))
    return [data]


def constantofshape_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Constantofshape Op function logic implementation."""
    val = attrs.get("value", np.array([0], dtype=np.float32))
    return [np.full(tuple(inputs[0]), val[0])]


def tile_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Tile Op function logic implementation."""
    return [np.tile(inputs[0], tuple(inputs[1]))]


def expand_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Expand Op function logic implementation."""
    return [np.broadcast_to(inputs[0], tuple(inputs[1]))]


def shape_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Shape Op function logic implementation."""
    return [np.array(inputs[0].shape, dtype=np.int64)]


def size_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Size Op function logic implementation."""
    return [np.array(inputs[0].size, dtype=np.int64)]


def nonzero_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Nonzero Op function logic implementation."""
    return [np.array(np.nonzero(inputs[0]))]


def topk_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Topk Op function logic implementation."""
    k = int(inputs[1][0])
    axis = attrs.get("axis", -1)
    indices = np.argsort(inputs[0], axis=axis)[..., -k:][..., ::-1]
    values = np.take_along_axis(inputs[0], indices, axis=axis)
    return [values, indices]


def unique_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Unique Op function logic implementation."""
    (u, indices, inverse_indices, counts) = np.unique(
        inputs[0], return_index=True, return_inverse=True, return_counts=True
    )
    return [u, indices, inverse_indices, counts]


def cumsum_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Cumsum Op function logic implementation."""
    axis = int(inputs[1][0])
    return [np.cumsum(inputs[0], axis=axis)]


def reversesequence_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Reversesequence Op function logic implementation."""
    return [np.flip(inputs[0], axis=attrs.get("batch_axis", 1))]


def compress_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Compress Op function logic implementation."""
    axis = attrs.get("axis")
    return [np.compress(inputs[1].astype(bool), inputs[0], axis=axis)]


def trilu_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Trilu Op function logic implementation."""
    upper = attrs.get("upper", 1)
    k = inputs[1][0] if len(inputs) > 1 else 0
    if upper:
        return [np.triu(inputs[0], k=k)]
    return [np.tril(inputs[0], k=k)]


def col2im_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Col2im Op function logic implementation."""
    return [np.zeros(tuple(inputs[1]), dtype=inputs[0].dtype)]


def sequenceconstruct_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Sequenceconstruct Op function logic implementation."""
    return [inputs]


def sequenceat_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Sequenceat Op function logic implementation."""
    return [inputs[0][int(inputs[1])]]


def sequenceempty_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Sequenceempty Op function logic implementation."""
    return [[]]


def sequenceerase_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Sequenceerase Op function logic implementation."""
    seq = list(inputs[0])
    idx = int(inputs[1].item()) if len(inputs) > 1 else -1
    seq.pop(idx)
    return [seq]


def sequenceinsert_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Sequenceinsert Op function logic implementation."""
    seq = list(inputs[0])
    idx = int(inputs[2]) if len(inputs) > 2 else len(seq)
    seq.insert(idx, inputs[1])
    return [seq]


def sequencelength_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Sequencelength Op function logic implementation."""
    return [np.array(len(inputs[0]), dtype=np.int64)]


def splittosequence_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Splittosequence Op function logic implementation."""
    axis = attrs.get("axis", 0)
    return [list(np.split(inputs[0], inputs[0].shape[axis], axis=axis))]


def concatfromsequence_op(inputs: list[np.ndarray], attrs: dict[str, Any]) -> list[np.ndarray]:
    """Concatfromsequence Op function logic implementation."""
    axis = attrs.get("axis", 0)
    new_axis = attrs.get("new_axis", 0)
    if new_axis:
        return [np.stack(inputs[0], axis=axis)]
    return [np.concatenate(inputs[0], axis=axis)]


OP_REGISTRY: dict[str, Callable[[list[np.ndarray], dict[str, Any]], list[np.ndarray]]] = {
    "ConvTranspose": convtranspose_op,
    "MaxPool": maxpool_op,
    "AveragePool": averagepool_op,
    "GlobalAveragePool": globalaveragepool_op,
    "GlobalMaxPool": globalmaxpool_op,
    "GlobalLpPool": globallppool_op,
    "MaxRoiPool": maxroipool_op,
    "RoiAlign": roialign_op,
    "InstanceNormalization": instancenormalization_op,
    "LRN": lrn_op,
    "LeakyRelu": leakyrelu_op,
    "PRelu": prelu_op,
    "Elu": elu_op,
    "Selu": selu_op,
    "HardSigmoid": hardsigmoid_op,
    "LogSoftmax": logsoftmax_op,
    "Softplus": softplus_op,
    "Softsign": softsign_op,
    "Hardmax": hardmax_op,
    "HardSwish": hardswish_op,
    "Mish": mish_op,
    "Shrink": shrink_op,
    "Dropout": dropout_op,
    "RNN": rnn_op,
    "LSTM": lstm_op,
    "GRU": gru_op,
    "GridSample": gridsample_op,
    "Pad": pad_op,
    "Resize": resize_op,
    "Constant": constantofshape_op,
    "Clip": clip_op,
    "Split": splittosequence_op,
    "SpaceToDepth": spacetodepth_op,
    "DepthToSpace": depthtospace_op,
    "Squeeze": squeeze_op,
    "Unsqueeze": unsqueeze_op,
    "GatherElements": gatherelements_op,
    "ScatterElements": scatter_op,
    "Scatter": scatter_op,
    "ConstantOfShape": constantofshape_op,
    "Tile": tile_op,
    "Expand": expand_op,
    "Shape": shape_op,
    "Size": size_op,
    "NonZero": nonzero_op,
    "TopK": topk_op,
    "Unique": unique_op,
    "CumSum": cumsum_op,
    "ReverseSequence": reversesequence_op,
    "Compress": compress_op,
    "Trilu": trilu_op,
    "Col2Im": col2im_op,
    "SequenceConstruct": sequenceconstruct_op,
    "SequenceAt": sequenceat_op,
    "SequenceEmpty": sequenceempty_op,
    "SequenceErase": sequenceerase_op,
    "SequenceInsert": sequenceinsert_op,
    "SequenceLength": sequencelength_op,
    "SplitToSequence": splittosequence_op,
    "ConcatFromSequence": concatfromsequence_op,
    "Abs": abs_op,
    "Acos": acos_op,
    "Acosh": acosh_op,
    "Asin": asin_op,
    "Asinh": asinh_op,
    "Atan": atan_op,
    "Atanh": atanh_op,
    "Cos": cos_op,
    "Cosh": cosh_op,
    "Sin": sin_op,
    "Sinh": sinh_op,
    "Tan": tan_op,
    "Ceil": ceil_op,
    "Floor": floor_op,
    "Round": round_op,
    "Exp": exp_op,
    "Log": log_op,
    "Sqrt": sqrt_op,
    "Erf": erf_op,
    "Sign": sign_op,
    "Mod": mod_op,
    "IsInf": isinf_op,
    "IsNaN": isnan_op,
    "Equal": equal_op,
    "Greater": greater_op,
    "GreaterOrEqual": greaterorequal_op,
    "Less": less_op,
    "LessOrEqual": lessorequal_op,
    "And": and_op,
    "Or": or_op,
    "Not": not_op,
    "Xor": xor_op,
    "BitShift": bitshift_op,
    "BitwiseAnd": bitwiseand_op,
    "BitwiseNot": bitwisenot_op,
    "BitwiseOr": bitwiseor_op,
    "BitwiseXor": bitwisexor_op,
    "ReduceL1": reducel1_op,
    "ReduceL2": reducel2_op,
    "ReduceLogSum": reducelogsum_op,
    "ReduceLogSumExp": reducelogsumexp_op,
    "ReduceSumSquare": reducesumsquare_op,
    "Einsum": einsum_op,
    "Cast": cast_op,
    "CastLike": castlike_op,
    "Gemm": gemm_op,
    "Add": add_op,
    "Sub": sub_op,
    "Mul": mul_op,
    "Div": div_op,
    "Pow": pow_op,
    "MatMul": matmul_op,
    "Conv": conv_op,
    "Relu": relu_op,
    "Sigmoid": sigmoid_op,
    "Tanh": tanh_op,
    "Gelu": gelu_op,
    "ReduceSum": reducesum_op,
    "ReduceMean": reducemean_op,
    "ReduceMax": reducemax_op,
    "Transpose": transpose_op,
    "Reshape": reshape_op,
    "Flatten": flatten_op,
    "Concat": concat_op,
    "Gather": gather_op,
    "ScatterND": scatternd_op,
    "Slice": slice_op,
    "Softmax": softmax_op,
    "LayerNormalization": layernorm_op,
    "BatchNormalization": batchnorm_op,
}
