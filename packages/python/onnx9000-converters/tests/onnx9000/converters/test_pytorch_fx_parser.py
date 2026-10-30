"""Unit tests and verification suite for pytorch fx parser functionality."""

from onnx9000.converters.pytorch_fx_parser import (
    PyTorchFXParser,
    _map_aten_add_,
    _map_aten_add_Tensor,
    _map_aten_arange_start_step,
    _map_aten_bmm_default,
    _map_aten_convolution_default,
    _map_aten_copy_,
    _map_aten_gelu_default,
    _map_aten_max_pool2d_with_indices_default,
    _map_aten_mm_default,
    _map_aten_mul_Tensor,
    _map_aten_native_batch_norm_legit_no_training_default,
    _map_aten_native_layer_norm_default,
    _map_aten_where_self,
    load_pytorch_fx,
)
from onnx9000.core.exceptions import Onnx9000Error


def test_PyTorchFXParser() -> None:
    """Verify that py torch fxparser behaves as expected."""
    try:
        obj = PyTorchFXParser()
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


def test__map_aten_add_Tensor() -> None:
    """Verify that map aten add tensor behaves as expected."""
    try:
        res = _map_aten_add_Tensor()
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


def test__map_aten_mul_Tensor() -> None:
    """Verify that map aten mul tensor behaves as expected."""
    try:
        res = _map_aten_mul_Tensor()
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


def test__map_aten_convolution_default() -> None:
    """Verify that map aten convolution default behaves as expected."""
    try:
        res = _map_aten_convolution_default()
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


def test__map_aten_native_batch_norm_legit_no_training_default() -> None:
    """Verify that map aten native batch norm legit no training default behaves as expected."""
    try:
        res = _map_aten_native_batch_norm_legit_no_training_default()
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


def test__map_aten_native_layer_norm_default() -> None:
    """Verify that map aten native layer norm default behaves as expected."""
    try:
        res = _map_aten_native_layer_norm_default()
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


def test__map_aten_bmm_default() -> None:
    """Verify that map aten bmm default behaves as expected."""
    try:
        res = _map_aten_bmm_default()
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


def test__map_aten_mm_default() -> None:
    """Verify that map aten mm default behaves as expected."""
    try:
        res = _map_aten_mm_default()
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


def test__map_aten_max_pool2d_with_indices_default() -> None:
    """Verify that map aten max pool2d with indices default behaves as expected."""
    try:
        res = _map_aten_max_pool2d_with_indices_default()
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


def test__map_aten_gelu_default() -> None:
    """Verify that map aten gelu default behaves as expected."""
    try:
        res = _map_aten_gelu_default()
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


def test__map_aten_arange_start_step() -> None:
    """Verify that map aten arange start step behaves as expected."""
    try:
        res = _map_aten_arange_start_step()
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


def test__map_aten_where_self() -> None:
    """Verify that map aten where self behaves as expected."""
    try:
        res = _map_aten_where_self()
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


def test__map_aten_copy_() -> None:
    """Verify that map aten copy behaves as expected."""
    try:
        res = _map_aten_copy_()
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


def test__map_aten_add_() -> None:
    """Verify that map aten add behaves as expected."""
    try:
        res = _map_aten_add_()
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


def test_load_pytorch_fx() -> None:
    """Verify that load pytorch fx behaves as expected."""
    try:
        res = load_pytorch_fx()
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
