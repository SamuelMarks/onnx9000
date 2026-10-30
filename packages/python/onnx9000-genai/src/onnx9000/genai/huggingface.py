"""Hugging Face Hub Integration for GenAI.

This module provides utility classes and methods for downloading models directly
from the Hugging Face Hub, loading generation configurations, and extracting
metadata from model configuration files.
"""


class HuggingFaceIntegration:
    """Provides utility methods for Hugging Face Hub integration.

    This class encapsulates operations for downloading model files from the Hugging Face
    Hub, loading generation configurations, and reading metadata from Hugging Face
    configuration files.
    """

    @staticmethod
    def download_model(repo_id: str, output_dir: str) -> None:
        """Download model files from the Hugging Face Hub into the output directory.

        Downloads all model files associated with the specified repository ID from the
        Hugging Face Hub. If `huggingface_hub` is not installed, it falls back to a mock
        implementation that creates a default 'config.json' file under the output directory.

        Args:
            repo_id: The Hugging Face repository ID (e.g., 'org/model-name').
            output_dir: The local directory path where the downloaded files should be stored.

        Raises:
            OSError: If creating directories or writing files fails.
        """
        import os

        try:
            from huggingface_hub import snapshot_download

            snapshot_download(repo_id=repo_id, local_dir=output_dir)
        except ImportError:
            # Fallback mock implementation if huggingface_hub is not installed
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "config.json"), "w") as f:
                f.write('{"model_type": "mock"}')

    @staticmethod
    def load_generation_config(config_path: str) -> dict:
        """Load a standard Hugging Face generation configuration JSON file.

        Reads and parses the generation configuration file from the specified path.

        Args:
            config_path: The file path to the 'generation_config.json' file.

        Returns:
            A dictionary containing the parsed generation configuration.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            json.JSONDecodeError: If the configuration file contains invalid JSON.
        """
        # Support standard HuggingFace generation_config.json loading
        import json

        with open(config_path) as f:
            return json.load(f)

    @staticmethod
    def load_metadata_from_config(config_path: str) -> dict:
        """Load model metadata directly from a Hugging Face configuration JSON file.

        Reads and parses the model configuration file (e.g., 'config.json') to extract
        model metadata.

        Args:
            config_path: The file path to the model's 'config.json' file.

        Returns:
            A dictionary containing the parsed model metadata and configuration details.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            json.JSONDecodeError: If the configuration file contains invalid JSON.
        """
        # Support consuming metadata directly from HuggingFace config.json
        import json

        with open(config_path) as f:
            return json.load(f)
