"""Module providing core logic and structural definitions."""

from typing import Tuple, List, Optional
import math


class ImageData:
    """Provides semantic functionality and verification."""

    def __init__(
        self, width: int, height: int, data: Optional[List[int]] = None
    ) -> None:
        """Provides semantic functionality and verification."""
        self.width = width
        self.height = height
        if data is None:
            self.data = [0] * (width * height * 4)
        else:
            self.data = data


def nearest_neighbor_resize(
    img_data: ImageData, target_width: int, target_height: int
) -> ImageData:
    """Provides semantic functionality and verification."""
    out = ImageData(target_width, target_height)
    x_ratio = img_data.width / target_width
    y_ratio = img_data.height / target_height

    for y in range(target_height):
        for x in range(target_width):
            px = int(math.floor(x * x_ratio))
            py = int(math.floor(y * y_ratio))

            in_idx = (py * img_data.width + px) * 4
            out_idx = (y * target_width + x) * 4

            out.data[out_idx : out_idx + 4] = img_data.data[in_idx : in_idx + 4]

    return out


def center_crop(img_data: ImageData, crop_width: int, crop_height: int) -> ImageData:
    """Provides semantic functionality and verification."""
    out = ImageData(crop_width, crop_height)
    start_x = (img_data.width - crop_width) // 2
    start_y = (img_data.height - crop_height) // 2

    for y in range(crop_height):
        for x in range(crop_width):
            in_idx = ((start_y + y) * img_data.width + (start_x + x)) * 4
            out_idx = (y * crop_width + x) * 4
            out.data[out_idx : out_idx + 4] = img_data.data[in_idx : in_idx + 4]

    return out


def to_tensor(
    images: List[ImageData],
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    rgb: bool = True,
) -> Tuple[List[float], Tuple[int, int, int, int]]:
    """Provides semantic functionality and verification."""
    if not images:
        raise ValueError("No images to convert")

    h = images[0].height
    w = images[0].width
    n = len(images)
    c = 3

    data = [0.0] * (n * c * h * w)

    for b in range(n):
        img = images[b]
        if img.width != w or img.height != h:
            raise ValueError("All images must have the same dimensions")

        b_offset = b * c * h * w

        for y in range(h):
            for x in range(w):
                pixel_idx = (y * w + x) * 4

                r = img.data[pixel_idx] / 255.0
                g = img.data[pixel_idx + 1] / 255.0
                bl = img.data[pixel_idx + 2] / 255.0

                r = (r - mean[0]) / std[0]
                g = (g - mean[1]) / std[1]
                bl = (bl - mean[2]) / std[2]

                out_idx_r = b_offset + 0 * h * w + y * w + x
                out_idx_g = b_offset + 1 * h * w + y * w + x
                out_idx_b = b_offset + 2 * h * w + y * w + x

                if rgb:
                    data[out_idx_r] = r
                    data[out_idx_g] = g
                    data[out_idx_b] = bl
                else:
                    data[out_idx_r] = bl
                    data[out_idx_g] = g
                    data[out_idx_b] = r

    return data, (n, c, h, w)
