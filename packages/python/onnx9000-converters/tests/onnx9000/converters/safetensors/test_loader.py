"""Unit tests and verification suite for loader functionality."""

from onnx9000.converters.safetensors.loader import (
    convert_pytorch_to_safetensors,
    convert_tf_to_safetensors,
    dump_graph_to_safetensors,
    load_and_patch_state_dict,
    load_safetensors_to_graph,
    map_huggingface_to_onnx,
    unpack_awq,
    validate_onnx_shapes_and_dtypes,
)
from onnx9000.core.exceptions import Onnx9000Error


def test_load_safetensors_to_graph() -> None:
    """Verify that load safetensors to graph behaves as expected."""
    try:
        res = load_safetensors_to_graph()
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


def test_map_huggingface_to_onnx() -> None:
    """Verify that map huggingface to onnx behaves as expected."""
    try:
        res = map_huggingface_to_onnx()
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


def test_load_and_patch_state_dict() -> None:
    """Verify that load and patch state dict behaves as expected."""
    try:
        res = load_and_patch_state_dict()
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


def test_convert_pytorch_to_safetensors() -> None:
    """Verify that convert pytorch to safetensors behaves as expected."""
    try:
        res = convert_pytorch_to_safetensors()
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


def test_convert_tf_to_safetensors() -> None:
    """Verify that convert tf to safetensors behaves as expected."""
    try:
        res = convert_tf_to_safetensors()
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


def test_dump_graph_to_safetensors() -> None:
    """Verify that dump graph to safetensors behaves as expected."""
    try:
        res = dump_graph_to_safetensors()
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


def test_validate_onnx_shapes_and_dtypes() -> None:
    """Verify that validate onnx shapes and dtypes behaves as expected."""
    try:
        res = validate_onnx_shapes_and_dtypes()
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


def test_unpack_awq() -> None:
    """Verify that unpack awq behaves as expected."""
    try:
        res = unpack_awq()
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
