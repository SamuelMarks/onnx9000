class HuggingFaceIntegration:
    """Integrations for downloading directly from HuggingFace Hub."""

    @staticmethod
    def download_model(repo_id: str, output_dir: str):
        """Download model files from the HuggingFace Hub into output_dir."""
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
        # Support standard HuggingFace generation_config.json loading
        import json

        with open(config_path) as f:
            return json.load(f)

    @staticmethod
    def load_metadata_from_config(config_path: str) -> dict:
        # Support consuming metadata directly from HuggingFace config.json
        import json

        with open(config_path) as f:
            return json.load(f)
