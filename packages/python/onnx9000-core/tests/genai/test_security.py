"""Module providing functionality for test_security."""

"""Test security."""


def test_security():
    """Docstring."""
    from onnx9000.genai.security import (
        ChatTemplateSanitizer,
        ContentSafetyFilter,
        CSPCompliance,
        EncryptedModelExecutor,
        ExploitPreventer,
        KVCacheIsolator,
        PromptInjectionDetector,
        ResourceLimits,
        SecureExecutionBoundary,
        SignatureValidator,
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
