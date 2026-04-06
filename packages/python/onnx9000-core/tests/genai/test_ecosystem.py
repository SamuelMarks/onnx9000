import pytest
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


def test_langchain():
    lc = LangChainIntegration()
    lc.register_chain("test_chain")
    assert lc.invoke("test_chain", "data") == "LangChain test_chain processed: data"
    with pytest.raises(ValueError):
        lc.invoke("missing", "data")


def test_llamaindex():
    li = LlamaIndexIntegration()
    li.add_document("doc1")
    assert "1" in li.query("test")


def test_unified_pipeline():
    pipe = UnifiedPipelineModel()
    pipe.add_step("step1")
    assert pipe.run("data") == "data"


def test_gguf_converter():
    conv = GGUFConverter()
    conv.set_option("quant", "q4")
    assert conv.convert("in.onnx", "out.gguf")


def test_nuxt_typings():
    nuxt = NuxtTypings()
    nuxt.add_typing("MyType", "field: string;")
    res = nuxt.generate()
    assert "interface MyType" in res
    assert "field: string;" in res


def test_discord_bot():
    bot = DiscordBotTemplate("token")
    bot.register_command("ping", "pong")
    assert bot.execute("ping") == "pong"
    assert bot.execute("unknown") == "Unknown command"


def test_offline_rag():
    rag = OfflineRAGVectorDB()
    rag.insert("doc1", [1.0, 0.0])
    assert "doc1" in rag.search([1.0, 0.0])


def test_benchmarks_pub():
    pub = BenchmarksPub()
    pub.publish("test1", 0.95)
    assert pub.get_score("test1") == 0.95
    assert pub.get_score("test2") is None


def test_v1_certification():
    cert = V1Certification()
    cert.certify("model1")
    assert cert.is_certified("model1")
    assert not cert.is_certified("model2")
