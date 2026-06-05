"""NCNN weights parser."""

import struct
from typing import BinaryIO

import numpy as np


class WeightsReader:
    """Reader for NCNN .bin weights file."""

    def __init__(self, f: BinaryIO):
        """Initialize with a file object."""
        self.f = f

    def read_blob(self, num_elements: int) -> np.ndarray:
        """Read a blob of weights.

        Args:
            num_elements: Number of elements to read.

        Returns:
            np.ndarray: Array of weights.
        """
        if num_elements == 0:
            return np.zeros(0, dtype=np.float32)

        # NCNN tag is 4 bytes
        tag_data = self.f.read(4)
        if not tag_data:
            return np.zeros(num_elements, dtype=np.float32)

        tag = struct.unpack("I", tag_data)[0]

        # 0x01306B47 means float16
        if tag == 0x01306B47:
            data = self.f.read(num_elements * 2)
            # numpy can parse float16
            return np.frombuffer(data, dtype=np.float16).astype(np.float32).copy()
        elif tag == 0x000D4B38:  # INT8
            # Not fully handled, return 0s
            self.f.read(num_elements)
            return np.zeros(num_elements, dtype=np.float32)
        else:
            # tag is usually not a tag if it's float32 in some older models,
            # or it is 0x00000000. Wait, actually, in float32, the tag is 0x00000000.
            # If there's no magic tag, we might have over-read.
            # In NCNN, a model can be saved with or without the tag.
            # To be safe, if tag doesn't match known magics, we assume it's part of float32 data.
            # Then we read the rest of the elements.
            data = tag_data + self.f.read((num_elements - 1) * 4)
            return np.frombuffer(data, dtype=np.float32).copy()
