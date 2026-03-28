"""Module providing functionality for test_huggingface."""

import json
import os

from onnx9000.genai.huggingface import HuggingFaceIntegration


def test_load_generation_config(tmp_path):
    """Test load generation config."""
    config_path = os.path.join(tmp_path, "generation_config.json")
    with open(config_path, "w") as f:
        json.dump({"temperature": 0.8}, f)

    config = HuggingFaceIntegration.load_generation_config(config_path)
    assert config["temperature"] == 0.8


def test_load_metadata_from_config(tmp_path):
    """Test load metadata from config."""
    config_path = os.path.join(tmp_path, "config.json")
    with open(config_path, "w") as f:
        json.dump({"vocab_size": 32000}, f)

    config = HuggingFaceIntegration.load_metadata_from_config(config_path)
    assert config["vocab_size"] == 32000
