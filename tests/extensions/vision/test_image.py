"""Module providing core logic and structural definitions."""

import pytest
from onnx9000.extensions.vision.image import (
    ImageData,
    nearest_neighbor_resize,
    center_crop,
    to_tensor,
)


def test_image_data():
    """Provides semantic functionality and verification."""
    img = ImageData(2, 2)
    assert len(img.data) == 16


def test_nearest_neighbor_resize():
    """Provides semantic functionality and verification."""
    data = [100, 0, 0, 255, 200, 0, 0, 255, 0, 100, 0, 255, 0, 200, 0, 255]
    img = ImageData(2, 2, data)
    out = nearest_neighbor_resize(img, 1, 1)
    assert out.width == 1
    assert out.height == 1
    assert out.data == [100, 0, 0, 255]


def test_center_crop():
    """Provides semantic functionality and verification."""
    data = [
        1,
        1,
        1,
        255,
        2,
        2,
        2,
        255,
        3,
        3,
        3,
        255,
        4,
        4,
        4,
        255,
        5,
        5,
        5,
        255,
        6,
        6,
        6,
        255,
        7,
        7,
        7,
        255,
        8,
        8,
        8,
        255,
        9,
        9,
        9,
        255,
    ]
    img = ImageData(3, 3, data)
    out = center_crop(img, 1, 1)
    assert out.width == 1
    assert out.height == 1
    assert out.data == [5, 5, 5, 255]


def test_to_tensor():
    """Provides semantic functionality and verification."""
    data = [255, 0, 0, 255]
    img = ImageData(1, 1, data)
    tensor_data, dims = to_tensor(
        [img], mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), rgb=True
    )
    assert dims == (1, 3, 1, 1)
    assert tensor_data[0] == 1.0
    assert tensor_data[1] == 0.0
    assert tensor_data[2] == 0.0


def test_to_tensor_bgr():
    """Provides semantic functionality and verification."""
    data = [255, 0, 0, 255]
    img = ImageData(1, 1, data)
    tensor_data, dims = to_tensor(
        [img], mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), rgb=False
    )
    assert tensor_data[0] == 0.0
    assert tensor_data[1] == 0.0
    assert tensor_data[2] == 1.0


def test_to_tensor_errors():
    """Provides semantic functionality and verification."""
    with pytest.raises(ValueError):
        to_tensor([])
    img1 = ImageData(1, 1)
    img2 = ImageData(2, 2)
    with pytest.raises(ValueError):
        to_tensor([img1, img2])
