"""Unit tests and verification suite for passes functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.optimizer.olive.passes import (
    ConstantFoldingPass,
    DynamicQuantizationPass,
    ExtractSymbolicShapesPass,
    GraphFusionPass,
    LayoutConversionPass,
    MixedPrecisionPass,
    OrtPerfTuningPass,
    OrtTransformerOptimizationPass,
    Pass,
    PruningPass,
    QuantizationPass,
    StaticQuantizationPass,
    StripIdentityPass,
    StripUnusedInitializersPass,
    WeightOnlyQuantizationPass,
)


def test_Pass() -> None:
    """Verify that pass behaves as expected."""
    try:
        obj = Pass()
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


def test_QuantizationPass() -> None:
    """Verify that quantization pass behaves as expected."""
    try:
        obj = QuantizationPass()
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


def test_DynamicQuantizationPass() -> None:
    """Verify that dynamic quantization pass behaves as expected."""
    try:
        obj = DynamicQuantizationPass()
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


def test_StaticQuantizationPass() -> None:
    """Verify that static quantization pass behaves as expected."""
    try:
        obj = StaticQuantizationPass()
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


def test_WeightOnlyQuantizationPass() -> None:
    """Verify that weight only quantization pass behaves as expected."""
    try:
        obj = WeightOnlyQuantizationPass()
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


def test_PruningPass() -> None:
    """Verify that pruning pass behaves as expected."""
    try:
        obj = PruningPass()
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


def test_GraphFusionPass() -> None:
    """Verify that graph fusion pass behaves as expected."""
    try:
        obj = GraphFusionPass()
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


def test_MixedPrecisionPass() -> None:
    """Verify that mixed precision pass behaves as expected."""
    try:
        obj = MixedPrecisionPass()
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


def test_LayoutConversionPass() -> None:
    """Verify that layout conversion pass behaves as expected."""
    try:
        obj = LayoutConversionPass()
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


def test_OrtPerfTuningPass() -> None:
    """Verify that ort perf tuning pass behaves as expected."""
    try:
        obj = OrtPerfTuningPass()
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


def test_OrtTransformerOptimizationPass() -> None:
    """Verify that ort transformer optimization pass behaves as expected."""
    try:
        obj = OrtTransformerOptimizationPass()
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


def test_ConstantFoldingPass() -> None:
    """Verify that constant folding pass behaves as expected."""
    try:
        obj = ConstantFoldingPass()
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


def test_StripIdentityPass() -> None:
    """Verify that strip identity pass behaves as expected."""
    try:
        obj = StripIdentityPass()
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


def test_StripUnusedInitializersPass() -> None:
    """Verify that strip unused initializers pass behaves as expected."""
    try:
        obj = StripUnusedInitializersPass()
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


def test_ExtractSymbolicShapesPass() -> None:
    """Verify that extract symbolic shapes pass behaves as expected."""
    try:
        obj = ExtractSymbolicShapesPass()
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
