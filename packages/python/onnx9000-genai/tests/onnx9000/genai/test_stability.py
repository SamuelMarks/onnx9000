"""Unit tests for runtime stability utilities in `onnx9000.genai.stability`."""

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


def test_stability_utilities() -> None:
    """Verify SafeMode, InputShapeValidator, GeneratorThreadSafety, BrowserWorkerIsolation, error classes, OOMHandler, and LargeVocabManager."""
    sm = SafeMode()
    sm.enable()
    assert sm.active is True
    sm.disable()
    assert sm.active is False

    isv = InputShapeValidator(16)
    assert isv.validate([1, 8]) is True
    assert isv.validate([1, 32]) is False

    gts = GeneratorThreadSafety()
    assert gts.acquire() is True
    gts.release()

    bwi = BrowserWorkerIsolation()
    bwi.initialize_worker("w0")
    assert bwi.worker_id == "w0"
    bwi.terminate_worker()
    assert bwi.worker_id is None

    assert isinstance(MalformedChatTemplateError("bad"), Exception)
    assert isinstance(EndOfStreamError("eos"), Exception)

    oom = OOMHandler()
    oom.trigger_oom()
    assert oom.oom_count == 1
    assert oom.clear_memory() is True

    lvm = LargeVocabManager(25000)
    assert lvm.get_chunks() == 3
