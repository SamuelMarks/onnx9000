from unittest.mock import MagicMock, patch

import pytest
from onnx9000.zoo.catalog import ZooCatalog
from onnx9000.zoo.sync import (
    BonsaiHubSynchronizer,
    HFHubPoller,
    ManifestGenerator,
    TimmSynchronizer,
)


def test_bonsai():
    c = ZooCatalog()
    s = BonsaiHubSynchronizer(c)

    # test empty poll
    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp
        assert s.poll_commits() == []

    # test sync
    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"sha": "1234567890", "commit": {"message": "test"}}]
        mock_get.return_value = mock_resp

        s.sync()
        assert c.get_model("bonsai_1234567") is not None
        # test second sync skips
        s.sync()


def test_timm():
    c = ZooCatalog()
    s = TimmSynchronizer(c)

    mock_model = MagicMock()
    mock_model.id = "timm/test"
    mock_model.sha = "123"

    with patch.object(s.api, "list_models", return_value=[mock_model]):
        s.sync()
        assert c.get_model("timm/test") is not None
        # test existing
        s.sync()


def test_hf():
    c = ZooCatalog()
    s = HFHubPoller(c)

    mock_model_st = MagicMock()
    mock_model_st.id = "hf/test_st"
    mock_model_st.sha = "123"

    mock_model_gguf = MagicMock()
    mock_model_gguf.id = "hf/test_gguf"
    mock_model_gguf.sha = "456"

    # Need a side effect depending on filter
    def mock_list(filter=None, limit=10):
        if filter == "safetensors":
            return [mock_model_st]
        elif filter == "gguf":
            return [mock_model_gguf]
        return []

    with patch.object(s.api, "list_models", side_effect=mock_list):
        s.sync()
        assert c.get_model("hf/test_st") is not None
        assert c.get_model("hf/test_gguf") is not None
        s.sync()


def test_manifest():
    c = ZooCatalog()
    c.add_model("test_m", "test_h", "abc", '{"k":"v"}', "")

    m = ManifestGenerator(c)
    res = m.generate_manifest("test_m")
    assert res["model_id"] == "test_m"
    assert res["hub"] == "test_h"

    err = m.generate_manifest("missing")
    assert "error" in err
