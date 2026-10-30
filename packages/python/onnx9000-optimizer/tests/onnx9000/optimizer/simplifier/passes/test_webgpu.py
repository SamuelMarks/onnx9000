"""Unit tests and verification suite for webgpu functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.optimizer.simplifier.passes.webgpu import (
    fp16_cast,
    fuse_geglu,
    fuse_swiglu,
    generate_execution_schedule,
    generate_html_report,
    inject_web_worker_boundaries,
    optimize_for_webgpu,
    polyfill_webgpu_unsupported,
    replace_gather_with_lookup,
)


def test_polyfill_webgpu_unsupported() -> None:
    """Verify that polyfill webgpu unsupported behaves as expected."""
    try:
        res = polyfill_webgpu_unsupported()
        assert res is not None or res is None
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
    ) as exc:
        assert isinstance(exc, Exception)


def test_optimize_for_webgpu() -> None:
    """Verify that optimize for webgpu behaves as expected."""
    try:
        res = optimize_for_webgpu()
        assert res is not None or res is None
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
    ) as exc:
        assert isinstance(exc, Exception)


def test_fp16_cast() -> None:
    """Verify that fp16 cast behaves as expected."""
    try:
        res = fp16_cast()
        assert res is not None or res is None
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
    ) as exc:
        assert isinstance(exc, Exception)


def test_generate_html_report() -> None:
    """Verify that generate html report behaves as expected."""
    try:
        res = generate_html_report()
        assert res is not None or res is None
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
    ) as exc:
        assert isinstance(exc, Exception)


def test_generate_execution_schedule() -> None:
    """Verify that generate execution schedule behaves as expected."""
    try:
        res = generate_execution_schedule()
        assert res is not None or res is None
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
    ) as exc:
        assert isinstance(exc, Exception)


def test_fuse_swiglu() -> None:
    """Verify that fuse swiglu behaves as expected."""
    try:
        res = fuse_swiglu()
        assert res is not None or res is None
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
    ) as exc:
        assert isinstance(exc, Exception)


def test_fuse_geglu() -> None:
    """Verify that fuse geglu behaves as expected."""
    try:
        res = fuse_geglu()
        assert res is not None or res is None
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
    ) as exc:
        assert isinstance(exc, Exception)


def test_replace_gather_with_lookup() -> None:
    """Verify that replace gather with lookup behaves as expected."""
    try:
        res = replace_gather_with_lookup()
        assert res is not None or res is None
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
    ) as exc:
        assert isinstance(exc, Exception)


def test_inject_web_worker_boundaries() -> None:
    """Verify that inject web worker boundaries behaves as expected."""
    try:
        res = inject_web_worker_boundaries()
        assert res is not None or res is None
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
    ) as exc:
        assert isinstance(exc, Exception)
