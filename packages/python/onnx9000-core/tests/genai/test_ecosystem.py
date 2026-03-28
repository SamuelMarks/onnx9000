"""Module providing functionality for test_ecosystem."""

"""Test ecosystem."""


def test_ecosystem():
    """Docstring."""
    from onnx9000.genai.ecosystem import (
        BenchmarksPub,
        DiscordBotTemplate,
        GGUFConverter,
        LangChainIntegration,
        LlamaIndexIntegration,
        NuxtTypings,
        OfflineRAGVectorDB,
        UnifiedPipelineModel,
        V1Certification,
    )

    assert LangChainIntegration()._initialized
    assert LlamaIndexIntegration()._initialized
    assert UnifiedPipelineModel()._initialized
    assert GGUFConverter()._initialized
    assert NuxtTypings()._initialized
    assert DiscordBotTemplate()._initialized
    assert OfflineRAGVectorDB()._initialized
    assert BenchmarksPub()._initialized
    assert V1Certification()._initialized
