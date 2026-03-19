def test_security():
    from onnx9000.genai.security import (
        PromptInjectionDetector,
        ContentSafetyFilter,
        SecureExecutionBoundary,
        ExploitPreventer,
        ChatTemplateSanitizer,
        ResourceLimits,
        EncryptedModelExecutor,
        SignatureValidator,
        KVCacheIsolator,
        CSPCompliance,
    )

    assert PromptInjectionDetector()._initialized
    assert ContentSafetyFilter()._initialized
    assert SecureExecutionBoundary()._initialized
    assert ExploitPreventer()._initialized
    assert ChatTemplateSanitizer()._initialized
    assert ResourceLimits()._initialized
    assert EncryptedModelExecutor()._initialized
    assert SignatureValidator()._initialized
    assert KVCacheIsolator()._initialized
    assert CSPCompliance()._initialized
