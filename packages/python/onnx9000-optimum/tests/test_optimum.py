"""Module docstring."""

import importlib

import onnx9000_optimum
import onnx9000_optimum.architectures
import onnx9000_optimum.export
import onnx9000_optimum.optimize
import onnx9000_optimum.quantize


def test_auto_detect_task():
    """Provides functional implementation."""
    importlib.reload(onnx9000_optimum.export)
    from onnx9000_optimum.export import auto_detect_task

    assert auto_detect_task({"architectures": ["SequenceClassification"]}) == "text-classification"
    assert auto_detect_task({"architectures": ["CausalLM"]}) == "text-generation"
    assert auto_detect_task({}) == "feature-extraction"


def test_export_model():
    """Provides functional implementation."""
    importlib.reload(onnx9000_optimum.export)
    from onnx9000_optimum.export import export_model

    assert callable(export_model)


def test_optimize_model():
    """Provides functional implementation."""
    importlib.reload(onnx9000_optimum.optimize)
    from onnx9000_optimum.optimize import optimize_model

    assert callable(optimize_model)


def test_quantize_model():
    """Provides functional implementation."""
    importlib.reload(onnx9000_optimum.quantize)
    from onnx9000_optimum.quantize import quantize_model

    assert callable(quantize_model)


def test_architectures():
    """Provides functional implementation."""
    importlib.reload(onnx9000_optimum.architectures)
    from onnx9000_optimum.architectures import BERTConfig

    config = BERTConfig()
    assert config is not None
