"""CPU backend operations mapping."""

import numpy as np
from typing import Callable, Dict, Any, List


def add_op(inputs: List[np.ndarray], attrs: Dict[str, Any]) -> List[np.ndarray]:
    """Executes the add op operation."""
    return [inputs[0] + inputs[1]]


def sub_op(inputs: List[np.ndarray], attrs: Dict[str, Any]) -> List[np.ndarray]:
    """Executes the sub op operation."""
    return [inputs[0] - inputs[1]]


def mul_op(inputs: List[np.ndarray], attrs: Dict[str, Any]) -> List[np.ndarray]:
    """Executes the mul op operation."""
    return [inputs[0] * inputs[1]]


def div_op(inputs: List[np.ndarray], attrs: Dict[str, Any]) -> List[np.ndarray]:
    """Executes the div op operation."""
    return [inputs[0] / inputs[1]]


def pow_op(inputs: List[np.ndarray], attrs: Dict[str, Any]) -> List[np.ndarray]:
    """Executes the pow op operation."""
    return [inputs[0] ** inputs[1]]


def matmul_op(inputs: List[np.ndarray], attrs: Dict[str, Any]) -> List[np.ndarray]:
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
    n, c, h, w = x.shape
    out_h = (h + 2 * pad_h - filter_h) // stride_h + 1
    out_w = (w + 2 * pad_w - filter_w) // stride_w + 1

    img = np.pad(x, [(0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)], "constant")
    col = np.zeros((n, c, filter_h, filter_w, out_h, out_w), dtype=x.dtype)

    for y in range(filter_h):
        y_max = y + stride_h * out_h
        for x_idx in range(filter_w):
            x_max = x_idx + stride_w * out_w
            col[:, :, y, x_idx, :, :] = img[
                :, :, y:y_max:stride_h, x_idx:x_max:stride_w
            ]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(n * out_h * out_w, -1)
    return col


def conv_op(inputs: List[np.ndarray], attrs: Dict[str, Any]) -> List[np.ndarray]:
    """Executes the conv op operation."""
    x = inputs[0]
    w = inputs[1]
    b = inputs[2] if len(inputs) > 2 else None

    # Handle standard 2D convolution via im2col and matmul
    n, c, h, w_dim = x.shape
    out_c, in_c, filter_h, filter_w = w.shape

    strides = attrs.get("strides", [1, 1])
    pads = attrs.get("pads", [0, 0, 0, 0])
    group = attrs.get("group", 1)

    if group != 1:
        # Grouped convolution fallback
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
            col = im2col(
                x_g, filter_h, filter_w, strides[0], strides[1], pads[0], pads[1]
            )
            col_w = w_g.reshape(out_c // group, -1).T
            out_g = np.matmul(col, col_w)
            out_g = out_g.reshape(n, out.shape[2], out.shape[3], -1).transpose(
                0, 3, 1, 2
            )
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


def relu_op(inputs: List[np.ndarray], attrs: Dict[str, Any]) -> List[np.ndarray]:
    """Executes the relu op operation."""
    return [np.maximum(0, inputs[0])]


def sigmoid_op(inputs: List[np.ndarray], attrs: Dict[str, Any]) -> List[np.ndarray]:
    """Executes the sigmoid op operation."""
    return [1.0 / (1.0 + np.exp(-inputs[0]))]


def tanh_op(inputs: List[np.ndarray], attrs: Dict[str, Any]) -> List[np.ndarray]:
    """Executes the tanh op operation."""
    return [np.tanh(inputs[0])]


def gelu_op(inputs: List[np.ndarray], attrs: Dict[str, Any]) -> List[np.ndarray]:
    """Executes the gelu op operation."""
    x = inputs[0]
    return [
        0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
    ]


def reducesum_op(inputs: List[np.ndarray], attrs: Dict[str, Any]) -> List[np.ndarray]:
    """Executes the reducesum op operation."""
    axes = tuple(attrs.get("axes", []))
    keepdims = attrs.get("keepdims", 1)
    if not axes:
        axes = None
    return [np.sum(inputs[0], axis=axes, keepdims=bool(keepdims))]


def reducemean_op(inputs: List[np.ndarray], attrs: Dict[str, Any]) -> List[np.ndarray]:
    """Executes the reducemean op operation."""
    axes = tuple(attrs.get("axes", []))
    keepdims = attrs.get("keepdims", 1)
    if not axes:
        axes = None
    return [np.mean(inputs[0], axis=axes, keepdims=bool(keepdims))]


def reducemax_op(inputs: List[np.ndarray], attrs: Dict[str, Any]) -> List[np.ndarray]:
    """Executes the reducemax op operation."""
    axes = tuple(attrs.get("axes", []))
    keepdims = attrs.get("keepdims", 1)
    if not axes:
        axes = None
    return [np.max(inputs[0], axis=axes, keepdims=bool(keepdims))]


def transpose_op(inputs: List[np.ndarray], attrs: Dict[str, Any]) -> List[np.ndarray]:
    """Executes the transpose op operation."""
    perm = attrs.get("perm", None)
    return [np.transpose(inputs[0], axes=perm)]


def reshape_op(inputs: List[np.ndarray], attrs: Dict[str, Any]) -> List[np.ndarray]:
    """Executes the reshape op operation."""
    shape_tensor = inputs[1] if len(inputs) > 1 else attrs.get("shape", [])
    target_shape = []
    for i, s in enumerate(shape_tensor):
        if s == 0:
            target_shape.append(inputs[0].shape[i])
        else:
            target_shape.append(s)
    return [np.reshape(inputs[0], target_shape)]


def flatten_op(inputs: List[np.ndarray], attrs: Dict[str, Any]) -> List[np.ndarray]:
    """Executes the flatten op operation."""
    axis = attrs.get("axis", 1)
    x = inputs[0]
    if axis < 0:
        axis += len(x.shape)
    new_shape = (np.prod(x.shape[:axis], dtype=int), np.prod(x.shape[axis:], dtype=int))
    return [np.reshape(x, new_shape)]


def concat_op(inputs: List[np.ndarray], attrs: Dict[str, Any]) -> List[np.ndarray]:
    """Executes the concat op operation."""
    axis = attrs.get("axis", 0)
    return [np.concatenate(inputs, axis=axis)]


def gather_op(inputs: List[np.ndarray], attrs: Dict[str, Any]) -> List[np.ndarray]:
    """Executes the gather op operation."""
    axis = attrs.get("axis", 0)
    return [np.take(inputs[0], inputs[1].astype(int), axis=axis)]


def scatternd_op(inputs: List[np.ndarray], attrs: Dict[str, Any]) -> List[np.ndarray]:
    """Executes the scatternd op operation."""
    data = np.copy(inputs[0])
    indices = inputs[1]
    updates = inputs[2]
    # Simple implementation, may need optimization
    idx_flat = indices.reshape(-1, indices.shape[-1])
    upd_flat = updates.reshape(-1, *updates.shape[indices.ndim - 1 :])
    for i in range(idx_flat.shape[0]):
        idx = tuple(idx_flat[i])
        data[idx] = upd_flat[i]
    return [data]


def slice_op(inputs: List[np.ndarray], attrs: Dict[str, Any]) -> List[np.ndarray]:
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


def softmax_op(inputs: List[np.ndarray], attrs: Dict[str, Any]) -> List[np.ndarray]:
    """Executes the softmax op operation."""
    axis = attrs.get("axis", -1)
    x = inputs[0]
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return [exp_x / np.sum(exp_x, axis=axis, keepdims=True)]


def layernorm_op(inputs: List[np.ndarray], attrs: Dict[str, Any]) -> List[np.ndarray]:
    """Executes the layernorm op operation."""
    x = inputs[0]
    scale = inputs[1]
    b = inputs[2] if len(inputs) > 2 else None
    axis = attrs.get("axis", -1)
    epsilon = attrs.get("epsilon", 1e-5)
    mean = np.mean(x, axis=axis, keepdims=True)
    var = np.var(x, axis=axis, keepdims=True)
    out = (x - mean) / np.sqrt(var + epsilon)
    out = out * scale
    if b is not None:
        out += b
    return [out]


def batchnorm_op(inputs: List[np.ndarray], attrs: Dict[str, Any]) -> List[np.ndarray]:
    """Executes the batchnorm op operation."""
    x = inputs[0]
    scale = inputs[1]
    b = inputs[2]
    mean = inputs[3]
    var = inputs[4]
    epsilon = attrs.get("epsilon", 1e-5)
    # x shape (N, C, ...)
    # reshape scale, b, mean, var to (1, C, 1, ...)
    shape = [1] * x.ndim
    shape[1] = x.shape[1]
    scale = scale.reshape(shape)
    b = b.reshape(shape)
    mean = mean.reshape(shape)
    var = var.reshape(shape)

    out = (x - mean) / np.sqrt(var + epsilon)
    out = out * scale + b
    return [out]


OP_REGISTRY: Dict[
    str, Callable[[List[np.ndarray], Dict[str, Any]], List[np.ndarray]]
] = {
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
