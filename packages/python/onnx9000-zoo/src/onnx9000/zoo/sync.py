"""Provides sync functionality."""

import sys
import types

try:
    import requests
except ImportError:
    requests = types.ModuleType("requests")
    requests.get = lambda *args, **kwargs: types.SimpleNamespace(status_code=404, json=list)  # type: ignore[attr-defined]
    sys.modules["requests"] = requests

try:
    from huggingface_hub import HfApi
except ImportError:

    class HfApi:  # type: ignore[no-redef]
        """Fallback Hugging Face Hub API client when huggingface_hub is not installed."""

        def list_models(
            self,
            author: str | None = None,
            filter: str | None = None,
            limit: int = 10,
        ) -> list[dict[str, str]]:
            """Return an empty list of models."""
            return []


from onnx9000.zoo.catalog import ZooCatalog


class BonsaiHubSynchronizer:
    """Synchronizer for jax-ml/bonsai commit streams."""

    def __init__(
        self,
        catalog: ZooCatalog,
        repo_url: str = "https://api.github.com/repos/jax-ml/bonsai",
    ) -> None:
        """Initialize BonsaiHubSynchronizer with configured parameters and state.

        Args:
            catalog: Parameter specifying catalog (ZooCatalog).
            repo_url: Parameter specifying repo url (str).
        """
        self.catalog = catalog
        self.repo_url = repo_url

    def poll_commits(
        self,
    ) -> list[dict[str, int | float | str | bool | bytes | list | dict | tuple | None]]:
        """Poll the latest commits from the jax-ml/bonsai repository."""
        response = requests.get(f"{self.repo_url}/commits")
        if response.status_code == 200:
            return [
                {"sha": commit["sha"], "message": commit["commit"]["message"]}
                for commit in response.json()
            ]
        return []

    def sync(self) -> None:
        """Trigger automated ingestion pipelines based on new commits."""
        commits = self.poll_commits()
        for commit in commits:
            model_id = f"bonsai_{commit['sha'][:7]}"
            if not self.catalog.get_model(model_id):
                self.catalog.add_model(
                    model_id=model_id,
                    hub="bonsai",
                    git_sha=commit["sha"],
                    hyperparameters='{"source": "jax-ml/bonsai"}',
                    tensor_hash="pending",
                )


class TimmSynchronizer:
    """Synchronizer for PyTorch image models via timm hub."""

    def __init__(self, catalog: ZooCatalog) -> None:
        """Initialize TimmSynchronizer with configured parameters and state.

        Args:
            catalog: Parameter specifying catalog (ZooCatalog).
        """
        self.catalog = catalog
        self.api = HfApi()

    def sync(self, limit: int = 10) -> None:
        """Sync models from the timm hub.

        Args:
            limit: Maximum number of models to sync.
        """
        models = self.api.list_models(author="timm", limit=limit)
        for model in models:
            if not self.catalog.get_model(model.id):
                self.catalog.add_model(
                    model_id=model.id,
                    hub="timm",
                    git_sha=getattr(model, "sha", "unknown"),
                    hyperparameters='{"source": "timm"}',
                    tensor_hash="pending",
                )


class HFHubPoller:
    """Poller for Hugging Face hub scoped strictly to safetensors/GGUF filtered tags."""

    def __init__(self, catalog: ZooCatalog) -> None:
        """Initialize HFHubPoller with configured parameters and state.

        Args:
            catalog: Parameter specifying catalog (ZooCatalog).
        """
        self.catalog = catalog
        self.api = HfApi()

    def sync(self, limit: int = 10) -> None:
        """Sync safetensors and GGUF models.

        Args:
            limit: Maximum number of models to sync.
        """
        # Filter for safetensors
        models = self.api.list_models(filter="safetensors", limit=limit)
        for model in models:
            if not self.catalog.get_model(model.id):
                self.catalog.add_model(
                    model_id=model.id,
                    hub="huggingface",
                    git_sha=getattr(model, "sha", "unknown"),
                    hyperparameters='{"format": "safetensors"}',
                    tensor_hash="pending",
                )

        # Filter for GGUF
        models_gguf = self.api.list_models(filter="gguf", limit=limit)
        for model in models_gguf:
            if not self.catalog.get_model(model.id):
                self.catalog.add_model(
                    model_id=model.id,
                    hub="huggingface",
                    git_sha=getattr(model, "sha", "unknown"),
                    hyperparameters='{"format": "gguf"}',
                    tensor_hash="pending",
                )


class ManifestGenerator:
    """AOT (Ahead-of-Time) manifest generator."""

    def __init__(self, catalog: ZooCatalog) -> None:
        """Initialize ManifestGenerator with configured parameters and state.

        Args:
            catalog: Parameter specifying catalog (ZooCatalog).
        """
        self.catalog = catalog

    def generate_manifest(
        self, model_id: str
    ) -> dict[str, int | float | str | bool | bytes | list | dict | tuple | None]:
        """Extract hyperparameters directly into a config structure before graph parsing begins.

        Args:
            model_id: The ID of the model in the catalog.

        Returns:
            A ModelConfig structure represented as a dictionary.
        """
        model = self.catalog.get_model(model_id)
        if not model:
            return {"error": "Model not found"}

        return {
            "model_id": model["id"],
            "hub": model["hub"],
            "config": {"source_hyperparameters": model["hyperparameters"]},
        }
