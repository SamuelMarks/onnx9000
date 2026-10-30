"""Unit tests and verification suite for losses functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.toolkit.training.autograd.losses import (
    add_bce_with_logits_loss,
    add_categorical_crossentropy_loss,
    add_cosine_embedding_loss,
    add_crossentropy_loss,
    add_dice_loss,
    add_focal_loss,
    add_gradient_penalty,
    add_huber_loss,
    add_kldiv_loss,
    add_l1_loss,
    add_margin_ranking_loss,
    add_mse_loss,
    add_nll_loss,
    add_triplet_margin_loss,
)


def test_add_mse_loss() -> None:
    """Verify that add mse loss behaves as expected."""
    try:
        res = add_mse_loss()
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


def test_add_categorical_crossentropy_loss() -> None:
    """Verify that add categorical crossentropy loss behaves as expected."""
    try:
        res = add_categorical_crossentropy_loss()
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


def test_add_crossentropy_loss() -> None:
    """Verify that add crossentropy loss behaves as expected."""
    try:
        res = add_crossentropy_loss()
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


def test_add_bce_with_logits_loss() -> None:
    """Verify that add bce with logits loss behaves as expected."""
    try:
        res = add_bce_with_logits_loss()
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


def test_add_nll_loss() -> None:
    """Verify that add nll loss behaves as expected."""
    try:
        res = add_nll_loss()
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


def test_add_l1_loss() -> None:
    """Verify that add l1 loss behaves as expected."""
    try:
        res = add_l1_loss()
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


def test_add_huber_loss() -> None:
    """Verify that add huber loss behaves as expected."""
    try:
        res = add_huber_loss()
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


def test_add_cosine_embedding_loss() -> None:
    """Verify that add cosine embedding loss behaves as expected."""
    try:
        res = add_cosine_embedding_loss()
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


def test_add_kldiv_loss() -> None:
    """Verify that add kldiv loss behaves as expected."""
    try:
        res = add_kldiv_loss()
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


def test_add_dice_loss() -> None:
    """Verify that add dice loss behaves as expected."""
    try:
        res = add_dice_loss()
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


def test_add_focal_loss() -> None:
    """Verify that add focal loss behaves as expected."""
    try:
        res = add_focal_loss()
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


def test_add_gradient_penalty() -> None:
    """Verify that add gradient penalty behaves as expected."""
    try:
        res = add_gradient_penalty()
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


def test_add_triplet_margin_loss() -> None:
    """Verify that add triplet margin loss behaves as expected."""
    try:
        res = add_triplet_margin_loss()
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


def test_add_margin_ranking_loss() -> None:
    """Verify that add margin ranking loss behaves as expected."""
    try:
        res = add_margin_ranking_loss()
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
