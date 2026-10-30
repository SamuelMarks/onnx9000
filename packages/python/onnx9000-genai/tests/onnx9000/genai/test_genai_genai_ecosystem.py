"""Unit tests for GenAI ecosystem integrations in `onnx9000.genai.ecosystem`."""

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


def test_ecosystem_integrations() -> None:
    """Verify LangChain, LlamaIndex, UnifiedPipeline, GGUFConverter, NuxtTypings, DiscordBot, OfflineRAG, BenchmarksPub, and V1Certification."""
    lc = LangChainIntegration()
    lc.register_chain("qa")
    assert lc.invoke("qa", "hi") == "LangChain qa processed: hi"

    li = LlamaIndexIntegration()
    li.add_document("doc")
    assert "1" in li.query("q")

    pipe = UnifiedPipelineModel()
    pipe.add_step("s1")
    assert pipe.run("in") == "in"

    gguf = GGUFConverter()
    gguf.set_option("q", "q4_0")
    assert gguf.convert("a.onnx", "a.gguf") is True

    nuxt = NuxtTypings()
    nuxt.add_typing("Cfg", "id: number;")
    assert "interface Cfg" in nuxt.generate()

    bot = DiscordBotTemplate("tok")
    bot.register_command("!help", "ok")
    assert bot.execute("!help") == "ok"

    rag = OfflineRAGVectorDB()
    rag.insert("d1", [1.0, 0.0])
    assert "d1" in rag.search([1.0, 0.0])

    pub = BenchmarksPub()
    pub.publish("m1", 99.0)
    assert pub.get_score("m1") == 99.0

    cert = V1Certification()
    cert.certify("m1")
    assert cert.is_certified("m1") is True
