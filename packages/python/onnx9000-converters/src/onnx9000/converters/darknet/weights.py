"""Darknet weights parser."""

import struct
from typing import BinaryIO

import numpy as np
from onnx9000.core.exceptions import ConversionError


def load_weights(
    f: BinaryIO,
) -> dict[str, int | float | str | bool | bytes | list | dict | tuple | None]:
    """Parse a Darknet .weights file.

    Args:
        f (BinaryIO): File object opened in binary mode.

    Returns:
        dict[str, Union[int, float, str, bool]]: Header information and a flat array of weights.

    """
    header_data = f.read(12)
    if len(header_data) < 12:
        raise ConversionError("Invalid weights file: too short for header")

    major, minor, revision = struct.unpack("iii", header_data)

    if (major * 10 + minor) >= 2 and major < 1000 and minor < 1000:
        seen_data = f.read(8)
        if len(seen_data) < 8:
            raise ConversionError("Invalid weights file: too short for seen (64-bit)")
        seen = struct.unpack("q", seen_data)[0]
    else:
        seen_data = f.read(4)
        if len(seen_data) < 4:
            raise ConversionError("Invalid weights file: too short for seen (32-bit)")
        seen = struct.unpack("i", seen_data)[0]

    weights = np.frombuffer(f.read(), dtype=np.float32).copy()

    return {
        "major": major,
        "minor": minor,
        "revision": revision,
        "seen": seen,
        "weights": weights,
    }
