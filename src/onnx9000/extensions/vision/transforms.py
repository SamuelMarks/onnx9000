"""Module providing core logic and structural definitions."""

from typing import Tuple, List, Optional
import math
import random
from .image import ImageData, nearest_neighbor_resize, center_crop


def bilinear_resize(
    img_data: ImageData, target_width: int, target_height: int
) -> ImageData:
    """Provides semantic functionality and verification."""
    out = ImageData(target_width, target_height)
    x_ratio = (img_data.width - 1) / max(1, target_width - 1) if target_width > 1 else 0
    y_ratio = (
        (img_data.height - 1) / max(1, target_height - 1) if target_height > 1 else 0
    )

    for y in range(target_height):
        for x in range(target_width):
            x_l = math.floor(x_ratio * x)
            y_l = math.floor(y_ratio * y)
            x_h = min(img_data.width - 1, math.ceil(x_ratio * x))
            y_h = min(img_data.height - 1, math.ceil(y_ratio * y))

            x_weight = (x_ratio * x) - x_l
            y_weight = (y_ratio * y) - y_l

            a = (y_l * img_data.width + x_l) * 4
            b = (y_l * img_data.width + x_h) * 4
            c = (y_h * img_data.width + x_l) * 4
            d = (y_h * img_data.width + x_h) * 4

            out_idx = (y * target_width + x) * 4

            for c_idx in range(4):
                val = (
                    img_data.data[a + c_idx] * (1 - x_weight) * (1 - y_weight)
                    + img_data.data[b + c_idx] * x_weight * (1 - y_weight)
                    + img_data.data[c + c_idx] * (1 - x_weight) * y_weight
                    + img_data.data[d + c_idx] * x_weight * y_weight
                )
                out.data[out_idx + c_idx] = int(round(val))
    return out


def resize_aspect_ratio(img_data: ImageData, size: int) -> ImageData:
    """Resizes keeping aspect ratio such that shorter edge matches `size`"""
    w, h = img_data.width, img_data.height
    if w < h:
        new_w = size
        new_h = int(size * h / w)
    else:
        new_h = size
        new_w = int(size * w / h)

    return bilinear_resize(img_data, new_w, new_h)


def random_crop(img_data: ImageData, crop_width: int, crop_height: int) -> ImageData:
    """Provides semantic functionality and verification."""
    if img_data.width <= crop_width or img_data.height <= crop_height:
        return center_crop(img_data, crop_width, crop_height)

    start_x = random.randint(0, img_data.width - crop_width)
    start_y = random.randint(0, img_data.height - crop_height)

    out = ImageData(crop_width, crop_height)
    for y in range(crop_height):
        for x in range(crop_width):
            in_idx = ((start_y + y) * img_data.width + (start_x + x)) * 4
            out_idx = (y * crop_width + x) * 4
            out.data[out_idx : out_idx + 4] = img_data.data[in_idx : in_idx + 4]
    return out


def pad(
    img_data: ImageData, pad_w: int, pad_h: int, color: Tuple[int, int, int] = (0, 0, 0)
) -> ImageData:
    """Provides semantic functionality and verification."""
    new_w = img_data.width + 2 * pad_w
    new_h = img_data.height + 2 * pad_h
    out = ImageData(new_w, new_h)

    # Fill with color
    for i in range(0, len(out.data), 4):
        out.data[i] = color[0]
        out.data[i + 1] = color[1]
        out.data[i + 2] = color[2]
        out.data[i + 3] = 255

    # Copy original
    for y in range(img_data.height):
        for x in range(img_data.width):
            in_idx = (y * img_data.width + x) * 4
            out_idx = ((y + pad_h) * new_w + (x + pad_w)) * 4
            out.data[out_idx : out_idx + 4] = img_data.data[in_idx : in_idx + 4]

    return out


def horizontal_flip(img_data: ImageData) -> ImageData:
    """Provides semantic functionality and verification."""
    out = ImageData(img_data.width, img_data.height)
    w = img_data.width
    for y in range(img_data.height):
        for x in range(w):
            in_idx = (y * w + x) * 4
            out_idx = (y * w + (w - 1 - x)) * 4
            out.data[out_idx : out_idx + 4] = img_data.data[in_idx : in_idx + 4]
    return out
