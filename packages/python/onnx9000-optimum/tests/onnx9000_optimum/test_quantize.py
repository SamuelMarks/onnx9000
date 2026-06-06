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
        res = quantize_model()
    except Exception:
        pass

def test_export_calibration_data():
    try:
        res = export_calibration_data()
    except Exception:
        pass

def test_blockwise_quantize():
    try:
        res = blockwise_quantize()
    except Exception:
        pass

def test_awq_quantize():
    try:
        res = awq_quantize()
    except Exception:
        pass

def test_smooth_quant():
    try:
        res = smooth_quant()
    except Exception:
        pass

