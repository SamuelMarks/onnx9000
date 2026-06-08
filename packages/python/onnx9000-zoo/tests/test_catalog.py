import os
import sqlite3

from onnx9000.zoo.catalog import ZooCatalog


def test_catalog():
    c = ZooCatalog()
    c.add_model("test_model", "test_hub", "abc1234", '{"a": 1}', "hash123")

    m = c.get_model("test_model")
    assert m["id"] == "test_model"
    assert m["hub"] == "test_hub"
    assert m["git_sha"] == "abc1234"
    assert m["hyperparameters"] == '{"a": 1}'
    assert m["tensor_hash"] == "hash123"

    assert c.get_model("missing") is None

    m_list = c.list_models()
    assert len(m_list) == 1

    m_list_hub = c.list_models("test_hub")
    assert len(m_list_hub) == 1

    m_list_hub2 = c.list_models("other_hub")
    assert len(m_list_hub2) == 0

    c.close()
