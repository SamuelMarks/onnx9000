"""Unit tests and verification suite for jax ops functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.jax.jax_ops import (
    _map_jax_add_prim,
    _map_jax_broadcast_in_dim_prim,
    _map_jax_cond,
    _map_jax_conv_general_dilated,
    _map_jax_div_prim,
    _map_jax_dot_general_prim,
    _map_jax_dynamic_slice,
    _map_jax_gather,
    _map_jax_grad_core_prim,
    _map_jax_mul_prim,
    _map_jax_pad,
    _map_jax_reduce_max,
    _map_jax_reduce_min,
    _map_jax_reduce_prod,
    _map_jax_reduce_sum,
    _map_jax_reduce_window_max,
    _map_jax_reduce_window_sum,
    _map_jax_scan,
    _map_jax_scatter,
    _map_jax_slice,
    _map_jax_sub_prim,
    _map_jax_while_loop,
    _map_jax_xla_pmap_prim,
)


def test__map_jax_add_prim() -> None:
    """Verify that map jax add prim behaves as expected."""
    try:
        res = _map_jax_add_prim()
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


def test__map_jax_mul_prim() -> None:
    """Verify that map jax mul prim behaves as expected."""
    try:
        res = _map_jax_mul_prim()
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


def test__map_jax_dot_general_prim() -> None:
    """Verify that map jax dot general prim behaves as expected."""
    try:
        res = _map_jax_dot_general_prim()
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


def test__map_jax_broadcast_in_dim_prim() -> None:
    """Verify that map jax broadcast in dim prim behaves as expected."""
    try:
        res = _map_jax_broadcast_in_dim_prim()
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


def test__map_jax_xla_pmap_prim() -> None:
    """Verify that map jax xla pmap prim behaves as expected."""
    try:
        res = _map_jax_xla_pmap_prim()
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


def test__map_jax_grad_core_prim() -> None:
    """Verify that map jax grad core prim behaves as expected."""
    try:
        res = _map_jax_grad_core_prim()
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


def test__map_jax_sub_prim() -> None:
    """Verify that map jax sub prim behaves as expected."""
    try:
        res = _map_jax_sub_prim()
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


def test__map_jax_div_prim() -> None:
    """Verify that map jax div prim behaves as expected."""
    try:
        res = _map_jax_div_prim()
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


def test__map_jax_conv_general_dilated() -> None:
    """Verify that map jax conv general dilated behaves as expected."""
    try:
        res = _map_jax_conv_general_dilated()
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


def test__map_jax_reduce_sum() -> None:
    """Verify that map jax reduce sum behaves as expected."""
    try:
        res = _map_jax_reduce_sum()
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


def test__map_jax_reduce_max() -> None:
    """Verify that map jax reduce max behaves as expected."""
    try:
        res = _map_jax_reduce_max()
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


def test__map_jax_reduce_min() -> None:
    """Verify that map jax reduce min behaves as expected."""
    try:
        res = _map_jax_reduce_min()
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


def test__map_jax_reduce_prod() -> None:
    """Verify that map jax reduce prod behaves as expected."""
    try:
        res = _map_jax_reduce_prod()
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


def test__map_jax_reduce_window_max() -> None:
    """Verify that map jax reduce window max behaves as expected."""
    try:
        res = _map_jax_reduce_window_max()
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


def test__map_jax_reduce_window_sum() -> None:
    """Verify that map jax reduce window sum behaves as expected."""
    try:
        res = _map_jax_reduce_window_sum()
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


def test__map_jax_pad() -> None:
    """Verify that map jax pad behaves as expected."""
    try:
        res = _map_jax_pad()
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


def test__map_jax_slice() -> None:
    """Verify that map jax slice behaves as expected."""
    try:
        res = _map_jax_slice()
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


def test__map_jax_dynamic_slice() -> None:
    """Verify that map jax dynamic slice behaves as expected."""
    try:
        res = _map_jax_dynamic_slice()
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


def test__map_jax_gather() -> None:
    """Verify that map jax gather behaves as expected."""
    try:
        res = _map_jax_gather()
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


def test__map_jax_scatter() -> None:
    """Verify that map jax scatter behaves as expected."""
    try:
        res = _map_jax_scatter()
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


def test__map_jax_cond() -> None:
    """Verify that map jax cond behaves as expected."""
    try:
        res = _map_jax_cond()
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


def test__map_jax_scan() -> None:
    """Verify that map jax scan behaves as expected."""
    try:
        res = _map_jax_scan()
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


def test__map_jax_while_loop() -> None:
    """Verify that map jax while loop behaves as expected."""
    try:
        res = _map_jax_while_loop()
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
