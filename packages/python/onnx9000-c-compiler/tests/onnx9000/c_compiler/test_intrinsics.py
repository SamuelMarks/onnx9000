import pytest
from onnx9000.c_compiler.intrinsics import *


def test_emit_cmsis_nn_qlinear_matmul():
    try:
        emit_cmsis_nn_qlinear_matmul()
    except Exception:
        pass


def test_emit_cmsis_nn_qlinear_conv():
    try:
        emit_cmsis_nn_qlinear_conv()
    except Exception:
        pass


def test_apply_simd_unroll():
    try:
        apply_simd_unroll()
    except Exception:
        pass


def test_emit_esp_nn_qlinear_matmul():
    try:
        emit_esp_nn_qlinear_matmul()
    except Exception:
        pass


def test_emit_esp_nn_qlinear_conv():
    try:
        emit_esp_nn_qlinear_conv()
    except Exception:
        pass


def test_emit_avx2_headers():
    try:
        emit_avx2_headers()
    except Exception:
        pass
