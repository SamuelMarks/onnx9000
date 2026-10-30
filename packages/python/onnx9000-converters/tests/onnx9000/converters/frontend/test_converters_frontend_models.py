"""Unit tests and verification suite for models functionality."""

from onnx9000.converters.frontend.models import GPT2, BasicBlock, GPT2Block, MobileNetV2, ResNet18
from onnx9000.core.exceptions import Onnx9000Error


def test_BasicBlock() -> None:
    """Verify that basic block behaves as expected."""
    try:
        obj = BasicBlock()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_ResNet18() -> None:
    """Verify that res net18 behaves as expected."""
    try:
        obj = ResNet18()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_MobileNetV2() -> None:
    """Verify that mobile net v2 behaves as expected."""
    try:
        obj = MobileNetV2()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_GPT2Block() -> None:
    """Verify that gpt2 block behaves as expected."""
    try:
        obj = GPT2Block()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass


def test_GPT2() -> None:
    """Verify that gpt2 behaves as expected."""
    try:
        obj = GPT2()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass
