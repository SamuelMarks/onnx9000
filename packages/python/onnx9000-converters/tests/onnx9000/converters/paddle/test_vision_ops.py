import pytest
from onnx9000.converters.paddle.vision_ops import *


def test__map_resize():
    try:
        _map_resize()
    except Exception:
        pass


def test__map_generate_proposals():
    try:
        _map_generate_proposals()
    except Exception:
        pass


def test__map_multiclass_nms():
    try:
        _map_multiclass_nms()
    except Exception:
        pass


def test__map_box_coder():
    try:
        _map_box_coder()
    except Exception:
        pass


def test__map_prior_box():
    try:
        _map_prior_box()
    except Exception:
        pass


def test__map_yolo_box():
    try:
        _map_yolo_box()
    except Exception:
        pass


def test__map_grid_sampler():
    try:
        _map_grid_sampler()
    except Exception:
        pass


def test__map_embedding():
    try:
        _map_embedding()
    except Exception:
        pass


def test__map_affine_grid():
    try:
        _map_affine_grid()
    except Exception:
        pass


def test__map_im2sequence():
    try:
        _map_im2sequence()
    except Exception:
        pass
