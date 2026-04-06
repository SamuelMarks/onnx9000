from unittest.mock import MagicMock, patch

import pytest
from onnx9000.zoo.catalog import ZooCatalog
from onnx9000.zoo.sync import (
    BonsaiHubSynchronizer,
    HFHubPoller,
    ManifestGenerator,
    TimmSynchronizer,
)


def test_bonsai_hub_synchronizer_poll():
    catalog = ZooCatalog()
    sync = BonsaiHubSynchronizer(catalog)

    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"sha": "1234567890", "commit": {"message": "Test commit"}}
        ]
        mock_get.return_value = mock_response

        commits = sync.poll_commits()
        assert len(commits) == 1
        assert commits[0]["sha"] == "1234567890"
    catalog.close()


def test_bonsai_hub_synchronizer_poll_error():
    catalog = ZooCatalog()
    sync = BonsaiHubSynchronizer(catalog)

    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        commits = sync.poll_commits()
        assert len(commits) == 0
    catalog.close()


def test_bonsai_hub_synchronizer_sync():
    catalog = ZooCatalog()
    sync = BonsaiHubSynchronizer(catalog)

    with patch.object(sync, "poll_commits") as mock_poll:
        mock_poll.return_value = [{"sha": "1234567890", "message": "Test commit"}]

        sync.sync()
        model = catalog.get_model("bonsai_1234567")
        assert model is not None
        assert model["hub"] == "bonsai"
    catalog.close()


def test_timm_synchronizer_sync():
    catalog = ZooCatalog()
    sync = TimmSynchronizer(catalog)

    with patch.object(sync.api, "list_models") as mock_list:
        mock_model = MagicMock()
        mock_model.id = "timm/test_model"
        mock_model.sha = "123456"
        mock_list.return_value = [mock_model]

        sync.sync(limit=1)
        model = catalog.get_model("timm/test_model")
        assert model is not None
        assert model["hub"] == "timm"
    catalog.close()


def test_hf_hub_poller_sync():
    catalog = ZooCatalog()
    sync = HFHubPoller(catalog)

    with patch.object(sync.api, "list_models") as mock_list:
        mock_model1 = MagicMock()
        mock_model1.id = "hf/safe_model"
        mock_model1.sha = "abc"

        mock_model2 = MagicMock()
        mock_model2.id = "hf/gguf_model"
        mock_model2.sha = "def"

        # We need to simulate multiple calls to list_models (one for safetensors, one for gguf)
        mock_list.side_effect = [[mock_model1], [mock_model2]]

        sync.sync(limit=1)
        model1 = catalog.get_model("hf/safe_model")
        assert model1 is not None
        assert model1["hub"] == "huggingface"

        model2 = catalog.get_model("hf/gguf_model")
        assert model2 is not None
        assert model2["hub"] == "huggingface"
    catalog.close()


def test_manifest_generator():
    catalog = ZooCatalog()
    catalog.add_model("test_model", "bonsai", "123", '{"layers": 10}', "hash")

    gen = ManifestGenerator(catalog)
    manifest = gen.generate_manifest("test_model")
    assert "error" not in manifest
    assert manifest["model_id"] == "test_model"
    assert manifest["hub"] == "bonsai"
    assert manifest["config"]["source_hyperparameters"] == '{"layers": 10}'

    # Test not found
    manifest_err = gen.generate_manifest("missing")
    assert "error" in manifest_err
    catalog.close()
