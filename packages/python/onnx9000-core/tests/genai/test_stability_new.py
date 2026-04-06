import pytest
from onnx9000.genai.stability import (
    SafeMode,
    InputShapeValidator,
    GeneratorThreadSafety,
    BrowserWorkerIsolation,
    MalformedChatTemplateError,
    EndOfStreamError,
    OOMHandler,
    LargeVocabManager,
)


def test_safe_mode():
    mode = SafeMode()
    mode.enable()
    assert mode.active
    mode.disable()
    assert not mode.active


def test_shape_validator():
    val = InputShapeValidator(10)
    assert not val.validate([])
    assert val.validate([1, 10])
    assert not val.validate([1, 11])


def test_thread_safety():
    ts = GeneratorThreadSafety()
    assert ts.acquire()
    ts.release()


def test_browser_isolation():
    iso = BrowserWorkerIsolation()
    isol_worker = "w1"
    iso.initialize_worker(isol_worker)
    assert iso.worker_id == isol_worker
    iso.terminate_worker()
    assert iso.worker_id is None


def test_exceptions():
    with pytest.raises(MalformedChatTemplateError):
        raise MalformedChatTemplateError("err")
    with pytest.raises(EndOfStreamError):
        raise EndOfStreamError("err")


def test_oom_handler():
    oom = OOMHandler()
    assert not oom.clear_memory()
    oom.trigger_oom()
    assert oom.oom_count == 1
    assert oom.clear_memory()
    assert oom.oom_count == 0


def test_vocab_manager():
    mgr = LargeVocabManager(25000)
    assert mgr.get_chunks() == 3
