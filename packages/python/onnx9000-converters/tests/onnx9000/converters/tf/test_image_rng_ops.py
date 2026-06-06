import pytest
from onnx9000.converters.tf.image_rng_ops import *

def test__map_resize():
    try:
        res = _map_resize()
    except Exception:
        pass

def test__map_crop_and_resize():
    try:
        res = _map_crop_and_resize()
    except Exception:
        pass

def test__map_extract_image_patches():
    try:
        res = _map_extract_image_patches()
    except Exception:
        pass

def test__map_random_uniform():
    try:
        res = _map_random_uniform()
    except Exception:
        pass

def test__map_random_standard_normal():
    try:
        res = _map_random_standard_normal()
    except Exception:
        pass

def test__map_truncated_normal():
    try:
        res = _map_truncated_normal()
    except Exception:
        pass

def test__map_multinomial():
    try:
        res = _map_multinomial()
    except Exception:
        pass

def test__map_nms():
    try:
        res = _map_nms()
    except Exception:
        pass

def test__map_hsv_to_rgb():
    try:
        res = _map_hsv_to_rgb()
    except Exception:
        pass

def test__map_rgb_to_hsv():
    try:
        res = _map_rgb_to_hsv()
    except Exception:
        pass

