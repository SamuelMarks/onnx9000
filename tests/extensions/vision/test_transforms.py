"""Module providing core logic and structural definitions."""

import pytest
import random
from onnx9000.extensions.vision.image import ImageData
from onnx9000.extensions.vision.transforms import (
    bilinear_resize,
    resize_aspect_ratio,
    random_crop,
    pad,
    horizontal_flip,
)


def test_bilinear_resize():
    """Provides semantic functionality and verification."""
    data = [100, 0, 0, 255, 200, 0, 0, 255, 0, 100, 0, 255, 0, 200, 0, 255]
    img = ImageData(2, 2, data)
    out = bilinear_resize(img, 3, 3)
    assert out.width == 3
    assert out.height == 3
    assert out.data[0] == 100


def test_resize_aspect_ratio():
    """Provides semantic functionality and verification."""
    img = ImageData(100, 200)
    out = resize_aspect_ratio(img, 50)
    assert out.width == 50
    assert out.height == 100
    img2 = ImageData(200, 100)
    out2 = resize_aspect_ratio(img2, 50)
    assert out2.height == 50
    assert out2.width == 100


def test_random_crop():
    """Provides semantic functionality and verification."""
    random.seed(42)
    img = ImageData(10, 10)
    out = random_crop(img, 5, 5)
    assert out.width == 5
    assert out.height == 5
    out2 = random_crop(img, 20, 20)
    assert out2.width == 20


def test_pad():
    """Provides semantic functionality and verification."""
    img = ImageData(1, 1, [1, 2, 3, 255])
    out = pad(img, 1, 1, color=(10, 20, 30))
    assert out.width == 3
    assert out.height == 3
    assert out.data[0:4] == [10, 20, 30, 255]
    assert out.data[16:20] == [1, 2, 3, 255]


def test_horizontal_flip():
    """Provides semantic functionality and verification."""
    img = ImageData(2, 1, [1, 2, 3, 255, 4, 5, 6, 255])
    out = horizontal_flip(img)
    assert out.data[0:4] == [4, 5, 6, 255]
    assert out.data[4:8] == [1, 2, 3, 255]
