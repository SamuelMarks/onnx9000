import os
import json
from onnx9000.genai.chunking import ChunkManager


def test_chunking_and_manifest(tmp_path):
    model_path = os.path.join(tmp_path, "model.onnx")

    chunks = ChunkManager.chunk_model(model_path)
    assert len(chunks) == 2
    assert "model-001.onnx" in chunks[0]

    ChunkManager.create_manifest(chunks, str(tmp_path))
    manifest_path = os.path.join(tmp_path, "manifest.json")
    assert os.path.exists(manifest_path)

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    assert manifest["total_chunks"] == 2
    assert manifest["chunks"] == chunks


def test_chunk_builder_export_and_cli():
    from onnx9000.genai.builder import GenAIBuilder, GenAICLI

    # Builder
    GenAIBuilder.build("model_id")
    GenAIBuilder.export_pytorch(None, None)
    GenAIBuilder.insert_kv_cache(None)
    GenAIBuilder.fix_dynamic_axes(None)
    GenAIBuilder.remove_past_state(None)

    # CLI
    GenAICLI.run_build("model_id", "webgpu")
    GenAICLI.run_chat("model_path")


def test_chunking_external_and_embed(tmp_path):
    """Test externalizing weights and embedding tokenizer config."""
    from onnx9000.genai.chunking import ChunkManager
    import os
    import json

    class MockModel:
        pass

    model = MockModel()
    out_path = os.path.join(tmp_path, "model.onnx")
    ChunkManager.externalize_weights(model, out_path)

    assert os.path.exists(f"{out_path}.data")
    with open(f"{out_path}.data", "rb") as f:
        assert f.read() == b"mock_weights_data"

    config = {"vocab_size": 1000}
    ChunkManager.embed_tokenizer(model, config)
    assert model.metadata_props["tokenizer_config"] == json.dumps(config)


def test_huggingface_download():
    from onnx9000.genai.huggingface import HuggingFaceIntegration

    HuggingFaceIntegration.download_model("repo", "dir")


def test_model_unimplemented():
    """Test that Model raises NotImplementedError on abstract methods."""
    from onnx9000.genai.model import Model
    from onnx9000.genai.types import ModelParams
    from onnx9000.genai.types import GeneratorParams
    import pytest

    model = Model(ModelParams(1, 1, 1, 1, 1, 1, 1))

    with pytest.raises(NotImplementedError):
        model.create_generator(GeneratorParams(max_length=10))
