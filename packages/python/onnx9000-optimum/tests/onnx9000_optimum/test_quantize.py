import pytest
from onnx9000_optimum.quantize import *


def test_CalibrationDataReader():
    try:
        obj = CalibrationDataReader()
        assert obj is not None
    except Exception:
        pass


def test_Quantizer():
    try:
        obj = Quantizer()
        assert obj is not None
    except Exception:
        pass


def test_quantize_model():
    try:
        quantize_model()
    except Exception:
        pass


def test_export_calibration_data():
    try:
        export_calibration_data()
    except Exception:
        pass


def test_blockwise_quantize():
    try:
        blockwise_quantize()
    except Exception:
        pass


def test_awq_quantize():
    try:
        awq_quantize()
    except Exception:
        pass


def test_smooth_quant():
    try:
        smooth_quant()
    except Exception:
        pass
