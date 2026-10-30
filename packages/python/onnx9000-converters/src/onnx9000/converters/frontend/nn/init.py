"""Initialization utilities."""

import math

from onnx9000.converters.frontend.tensor import Tensor
from onnx9000.core.exceptions import ConversionError


def calculate_fan_in_and_fan_out(tensor: Tensor) -> tuple[int, int]:
    """Execute calculate fan in and fan out operation.

    Args:
        tensor: Parameter specifying tensor (Tensor).

    Returns:
        Result of calculate fan in and fan out (tuple[int, int]).

    Raises:
        ValueError: If an invalid state or argument is encountered during execution.
    """
    dimensions = tensor.shape
    if len(dimensions) < 2:
        raise ConversionError(
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
        )
    num_input_fmaps = dimensions[1]
    num_output_fmaps = dimensions[0]
    receptive_field_size = 1
    if len(dimensions) > 2:
        for s in dimensions[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    return (fan_in, fan_out)


def xavier_uniform_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """Execute xavier uniform operation.

    Args:
        tensor: Parameter specifying tensor (Tensor).
        gain: Parameter specifying gain (float).

    Returns:
        Result of xavier uniform (Tensor).
    """
    import numpy as np

    if tensor.data is not None and isinstance(tensor.data, np.ndarray):
        (fan_in, fan_out) = calculate_fan_in_and_fan_out(tensor)
        std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
        a = math.sqrt(3.0) * std
        tensor.data = np.random.uniform(-a, a, tensor.shape).astype(tensor.data.dtype)
    return tensor


def xavier_normal_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """Execute xavier normal operation.

    Args:
        tensor: Parameter specifying tensor (Tensor).
        gain: Parameter specifying gain (float).

    Returns:
        Result of xavier normal (Tensor).
    """
    import numpy as np

    if tensor.data is not None and isinstance(tensor.data, np.ndarray):
        (fan_in, fan_out) = calculate_fan_in_and_fan_out(tensor)
        std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
        tensor.data = np.random.normal(0.0, std, tensor.shape).astype(tensor.data.dtype)
    return tensor


def kaiming_uniform_(
    tensor: Tensor, a: float = 0, mode: str = "fan_in", nonlinearity: str = "leaky_relu"
) -> Tensor:
    """Execute kaiming uniform operation.

    Args:
        tensor: Parameter specifying tensor (Tensor).
        a: Parameter specifying a (float).
        mode: Parameter specifying mode (str).
        nonlinearity: Parameter specifying nonlinearity (str).

    Returns:
        Result of kaiming uniform (Tensor).
    """
    import numpy as np

    if tensor.data is not None and isinstance(tensor.data, np.ndarray):
        (fan_in, fan_out) = calculate_fan_in_and_fan_out(tensor)
        fan = fan_in if mode == "fan_in" else fan_out
        gain = math.sqrt(2.0 / (1 + a**2))
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std
        tensor.data = np.random.uniform(-bound, bound, tensor.shape).astype(tensor.data.dtype)
    return tensor


def kaiming_normal_(
    tensor: Tensor, a: float = 0, mode: str = "fan_in", nonlinearity: str = "leaky_relu"
) -> Tensor:
    """Execute kaiming normal operation.

    Args:
        tensor: Parameter specifying tensor (Tensor).
        a: Parameter specifying a (float).
        mode: Parameter specifying mode (str).
        nonlinearity: Parameter specifying nonlinearity (str).

    Returns:
        Result of kaiming normal (Tensor).
    """
    import numpy as np

    if tensor.data is not None and isinstance(tensor.data, np.ndarray):
        (fan_in, fan_out) = calculate_fan_in_and_fan_out(tensor)
        fan = fan_in if mode == "fan_in" else fan_out
        gain = math.sqrt(2.0 / (1 + a**2))
        std = gain / math.sqrt(fan)
        tensor.data = np.random.normal(0.0, std, tensor.shape).astype(tensor.data.dtype)
    return tensor


def constant_(tensor: Tensor, val: float) -> Tensor:
    """Execute constant operation.

    Args:
        tensor: Parameter specifying tensor (Tensor).
        val: Parameter specifying val (float).

    Returns:
        Result of constant (Tensor).
    """
    import numpy as np

    if tensor.data is not None and isinstance(tensor.data, np.ndarray):
        tensor.data.fill(val)
    return tensor


def zeros_(tensor: Tensor) -> Tensor:
    """Execute zeros operation.

    Args:
        tensor: Parameter specifying tensor (Tensor).

    Returns:
        Result of zeros (Tensor).
    """
    return constant_(tensor, 0.0)


def ones_(tensor: Tensor) -> Tensor:
    """Execute ones operation.

    Args:
        tensor: Parameter specifying tensor (Tensor).

    Returns:
        Result of ones (Tensor).
    """
    return constant_(tensor, 1.0)
