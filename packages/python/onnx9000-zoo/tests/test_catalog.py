import pytest
from onnx9000.zoo.catalog import ZooCatalog


def test_catalog_init():
    """Test that the catalog initializes correctly."""
    catalog = ZooCatalog()
    assert catalog.db_path == ":memory:"
    catalog.close()


def test_add_and_get_model():
    """Test adding and retrieving a model."""
    catalog = ZooCatalog()
    catalog.add_model("test_model", "huggingface", "abcdef123", '{"layers": 12}', "hash123")

    model = catalog.get_model("test_model")
    assert model is not None
    assert model["id"] == "test_model"
    assert model["hub"] == "huggingface"
    assert model["git_sha"] == "abcdef123"
    assert model["hyperparameters"] == '{"layers": 12}'
    assert model["tensor_hash"] == "hash123"
    catalog.close()


def test_get_nonexistent_model():
    """Test retrieving a model that doesn't exist."""
    catalog = ZooCatalog()
    model = catalog.get_model("missing_model")
    assert model is None
    catalog.close()


def test_list_models():
    """Test listing models."""
    catalog = ZooCatalog()
    catalog.add_model("m1", "timm", "111", "{}", "h1")
    catalog.add_model("m2", "huggingface", "222", "{}", "h2")
    catalog.add_model("m3", "timm", "333", "{}", "h3")

    all_models = catalog.list_models()
    assert len(all_models) == 3

    timm_models = catalog.list_models("timm")
    assert len(timm_models) == 2
    assert {"m1", "m3"} == {m["id"] for m in timm_models}
    catalog.close()
