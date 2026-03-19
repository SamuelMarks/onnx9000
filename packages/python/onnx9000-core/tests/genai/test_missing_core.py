def test_stability_coverage():
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

    s = SafeMode()
    assert s._initialized
    v = InputShapeValidator()
    assert v._initialized
    g = GeneratorThreadSafety()
    assert g._initialized
    b = BrowserWorkerIsolation()
    assert b._initialized
    m = MalformedChatTemplateError()
    assert m._initialized
    e = EndOfStreamError()
    assert e._initialized
    o = OOMHandler()
    assert o._initialized
    l = LargeVocabManager()
    assert l._initialized
