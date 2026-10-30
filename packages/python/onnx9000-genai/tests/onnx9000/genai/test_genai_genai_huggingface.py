"""Unit tests for HuggingFace config loading in `onnx9000.genai.huggingface`."""

import json
import os
import tempfile

from onnx9000.genai.huggingface import HuggingFaceIntegration


def test_huggingface_integration() -> None:
    """Verify HuggingFaceIntegration config loaders."""
    with tempfile.TemporaryDirectory() as tmpdir:
        gen_cfg = os.path.join(tmpdir, "generation_config.json")
        with open(gen_cfg, "w") as f:
            json.dump({"top_k": 40}, f)
        assert HuggingFaceIntegration.load_generation_config(gen_cfg)["top_k"] == 40
