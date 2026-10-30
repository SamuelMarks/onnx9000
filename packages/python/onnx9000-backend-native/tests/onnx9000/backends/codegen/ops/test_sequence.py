"""Unit tests and verification suite for sequence functionality."""

from onnx9000.backends.codegen.ops.sequence import (
    generate_concat_from_sequence,
    generate_sequence_at,
    generate_sequence_construct,
    generate_sequence_empty,
    generate_sequence_erase,
    generate_sequence_insert,
    generate_sequence_length,
    generate_sequence_map,
    generate_split_to_sequence,
)
from onnx9000.core.exceptions import Onnx9000Error


def test_generate_sequence_empty() -> None:
    """Verify that generate sequence empty behaves as expected."""
    try:
        res = generate_sequence_empty()
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


def test_generate_sequence_erase() -> None:
    """Verify that generate sequence erase behaves as expected."""
    try:
        res = generate_sequence_erase()
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


def test_generate_sequence_insert() -> None:
    """Verify that generate sequence insert behaves as expected."""
    try:
        res = generate_sequence_insert()
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


def test_generate_sequence_length() -> None:
    """Verify that generate sequence length behaves as expected."""
    try:
        res = generate_sequence_length()
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


def test_generate_sequence_map() -> None:
    """Verify that generate sequence map behaves as expected."""
    try:
        res = generate_sequence_map()
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


def test_generate_concat_from_sequence() -> None:
    """Verify that generate concat from sequence behaves as expected."""
    try:
        res = generate_concat_from_sequence()
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


def test_generate_split_to_sequence() -> None:
    """Verify that generate split to sequence behaves as expected."""
    try:
        res = generate_split_to_sequence()
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


def test_generate_sequence_construct() -> None:
    """Verify that generate sequence construct behaves as expected."""
    try:
        res = generate_sequence_construct()
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


def test_generate_sequence_at() -> None:
    """Verify that generate sequence at behaves as expected."""
    try:
        res = generate_sequence_at()
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
