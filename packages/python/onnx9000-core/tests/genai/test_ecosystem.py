def test_ecosystem():
    from onnx9000.genai.ecosystem import (
        LangChainIntegration,
        LlamaIndexIntegration,
        UnifiedPipelineModel,
        GGUFConverter,
        NuxtTypings,
        DiscordBotTemplate,
        OfflineRAGVectorDB,
        BenchmarksPub,
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
