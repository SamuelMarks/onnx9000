"""Functional interface."""

from typing import Any, Optional, Tuple, Union
from onnx9000.frontends.frontend.tensor import Tensor


def relu(input: Tensor) -> Tensor:
    """Provides semantic functionality and verification."""
    return input.relu()


def sigmoid(input: Tensor) -> Tensor:
    """Provides semantic functionality and verification."""
    return input.sigmoid()


def tanh(input: Tensor) -> Tensor:
    """Provides semantic functionality and verification."""
    return input.tanh()


def gelu(input: Tensor) -> Tensor:
    """Provides semantic functionality and verification."""
    return input.gelu()


def softmax(input: Tensor, dim: int = -1) -> Tensor:
    """Provides semantic functionality and verification."""
    return input.softmax(dim)


def linear(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    """Provides semantic functionality and verification."""
    res = input @ weight.T
    if bias is not None:
        res = res + bias
    return res


def conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Any = 1,
    padding: Any = 0,
    dilation: Any = 1,
    groups: int = 1,
) -> Tensor:
    """Provides semantic functionality and verification."""
    from onnx9000.frontends.frontend.utils import record_op

    def _pair(x):
        """Provides  pair functionality and verification."""
        return (x, x) if isinstance(x, int) else x

    attrs = {
        "kernel_shape": list(weight.shape[2:]),
        "strides": list(_pair(stride)),
        "pads": list(_pair(padding)) * 2,
        "dilations": list(_pair(dilation)),
        "group": groups,
    }
    inputs = [input, weight]
    if bias is not None:
        inputs.append(bias)
    return record_op("Conv", inputs, attrs)


def pad(
    input: Tensor, pad: Tuple[int], mode: str = "constant", value: float = 0.0
) -> Tensor:
    """Provides semantic functionality and verification."""
    from onnx9000.frontends.frontend.utils import record_op
    import numpy as np
    from onnx9000.core.dtypes import DType
    from onnx9000.frontends.frontend.tensor import Parameter

    dims = len(input.shape)
    num_pad_dims = len(pad) // 2
    onnx_pads = [0] * (dims * 2)
    for i in range(num_pad_dims):
        onnx_pads[dims - 1 - i] = pad[i * 2]
        onnx_pads[2 * dims - 1 - i] = pad[i * 2 + 1]
    pads_tensor = Parameter(
        (len(onnx_pads),), DType.INT64, "pads", np.array(onnx_pads, dtype=np.int64)
    )
    inputs = [input, pads_tensor]
    if mode == "constant":
        val_tensor = Parameter(
            (), DType.FLOAT32, "constant_value", np.array(value, dtype=np.float32)
        )
        inputs.append(val_tensor)
    return record_op("Pad", inputs, {"mode": mode})


def interpolate(
    input: Tensor,
    size: Optional[Any] = None,
    scale_factor: Optional[Any] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
) -> Tensor:
    """Provides semantic functionality and verification."""
    from onnx9000.frontends.frontend.utils import record_op
    import numpy as np
    from onnx9000.core.dtypes import DType
    from onnx9000.frontends.frontend.tensor import Parameter

    attrs = {"mode": mode}
    if align_corners is not None and mode in [
        "linear",
        "bilinear",
        "trilinear",
        "bicubic",
    ]:
        return None
    inputs = [input]
    inputs.append(Parameter((0,), DType.FLOAT32, "roi", np.array([], dtype=np.float32)))
    if scale_factor is not None:
        scales = [1.0, 1.0]
        if isinstance(scale_factor, float):
            scales += [scale_factor] * (len(input.shape) - 2)
        elif isinstance(scale_factor, (tuple, list)):
            scales += list(scale_factor)
        scales_tensor = Parameter(
            (len(scales),), DType.FLOAT32, "scales", np.array(scales, dtype=np.float32)
        )
        inputs.append(scales_tensor)
    else:
        return None
    if size is not None:
        return None
    return record_op("Resize", inputs, attrs)


def one_hot(tensor: Tensor, num_classes: int = -1) -> Tensor:
    """Provides semantic functionality and verification."""
    from onnx9000.frontends.frontend.utils import record_op
    import numpy as np
    from onnx9000.core.dtypes import DType
    from onnx9000.frontends.frontend.tensor import Parameter

    depth = Parameter((), DType.INT64, "depth", np.array(num_classes, dtype=np.int64))
    values = Parameter((2,), DType.INT64, "values", np.array([0, 1], dtype=np.int64))
    return record_op("OneHot", [tensor, depth, values], {"axis": -1})
