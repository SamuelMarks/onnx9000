import pytest


def test_auto_detect_task():
    from onnx9000_optimum.export import auto_detect_task

    assert auto_detect_task({"architectures": ["SequenceClassification"]}) == "text-classification"
    assert auto_detect_task({"architectures": ["CausalLM"]}) == "text-generation"
    assert auto_detect_task({}) == "feature-extraction"


def test_export_model():
    from onnx9000_optimum.export import export_model

    assert callable(export_model)


def test_optimize_model():
    from onnx9000_optimum.optimize import optimize_model

    assert callable(optimize_model)


def test_quantize_model():
    from onnx9000_optimum.quantize import quantize_model

    assert callable(quantize_model)


def test_architectures():
    from onnx9000_optimum.architectures import BERTConfig

    config = BERTConfig()
    assert config is not None
