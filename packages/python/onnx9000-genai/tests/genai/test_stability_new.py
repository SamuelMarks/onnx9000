"""Unit tests and verification suite for stability new functionality."""

from typing import NoReturn

import pytest
from onnx9000.genai.stability import (
    BrowserWorkerIsolation,
    EndOfStreamError,
    GeneratorThreadSafety,
    InputShapeValidator,
    LargeVocabManager,
    MalformedChatTemplateError,
    OOMHandler,
    SafeMode,
)


def test_safe_mode() -> None:
    """Verify that safe mode behaves as expected."""
    mode = SafeMode()
    mode.enable()
    assert mode.active
    mode.disable()
    assert not mode.active


def test_shape_validator() -> None:
    """Verify that shape validator behaves as expected."""
    val = InputShapeValidator(10)
    assert not val.validate([])
    assert val.validate([1, 10])
    assert not val.validate([1, 11])


def test_thread_safety() -> None:
    """Verify that thread safety behaves as expected."""
    ts = GeneratorThreadSafety()
    assert ts.acquire()
    ts.release()


def test_browser_isolation() -> None:
    """Verify that browser isolation behaves as expected."""
    iso = BrowserWorkerIsolation()
    isol_worker = "w1"
    iso.initialize_worker(isol_worker)
    assert iso.worker_id == isol_worker
    iso.terminate_worker()
    assert iso.worker_id is None


def test_exceptions() -> NoReturn:
    """Verify that exceptions behaves as expected.

    Raises:
        EndOfStreamError: If an invalid state or argument is encountered during execution.
        MalformedChatTemplateError: If an invalid state or argument is encountered during execution.
    """
    with pytest.raises(MalformedChatTemplateError):
        raise MalformedChatTemplateError("err")
    with pytest.raises(EndOfStreamError):
        raise EndOfStreamError("err")


def test_oom_handler() -> None:
    """Verify that oom handler behaves as expected."""
    oom = OOMHandler()
    assert not oom.clear_memory()
    oom.trigger_oom()
    assert oom.oom_count == 1
    assert oom.clear_memory()
    assert oom.oom_count == 0


def test_vocab_manager() -> None:
    """Verify that vocab manager behaves as expected."""
    mgr = LargeVocabManager(25000)
    assert mgr.get_chunks() == 3
