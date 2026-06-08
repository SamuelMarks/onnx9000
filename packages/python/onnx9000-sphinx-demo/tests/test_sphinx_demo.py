from unittest.mock import MagicMock

import pytest
from onnx9000.sphinx_demo import (
    InteractiveDemoDirective,
    InteractiveDemoNode,
    build_frontend,
    copy_frontend_assets,
    depart_interactive_demo_node,
    setup,
    visit_interactive_demo_node,
)


def test_interactive_demo_node():
    node = InteractiveDemoNode()
    assert node is not None

    mock_self = MagicMock()
    mock_self.body = []

    node.attributes = {"initial-source": "keras", "ids": ["1"], "classes": ["cls"]}
    visit_interactive_demo_node(mock_self, node)
    assert len(mock_self.body) == 1
    assert 'data-initial-source="keras"' in mock_self.body[0]

    depart_interactive_demo_node(mock_self, node)


def test_directive():
    from unittest.mock import MagicMock

    mock_state_machine = MagicMock()
    d = InteractiveDemoDirective(
        "test",
        [],
        {"initial-source": "keras", "initial-target": "onnx"},
        None,
        None,
        None,
        None,
        None,
        mock_state_machine,
    )
    nodes_res = d.run()
    assert len(nodes_res) == 1
    assert nodes_res[0]["initial-source"] == "keras"
    assert nodes_res[0]["initial-target"] == "onnx"


@pytest.fixture
def mock_app():
    app = MagicMock()
    app.srcdir = "/test/src"
    app.outdir = "/test/out"
    app.builder.format = "html"
    return app


def test_build_frontend(mock_app, monkeypatch):
    import os
    import subprocess

    monkeypatch.setattr(os.path, "exists", lambda p: True)
    monkeypatch.setattr(os, "listdir", lambda p: ["test.css", "test.js", "umd.cjs"])
    monkeypatch.setattr(subprocess, "check_call", MagicMock())

    build_frontend(mock_app)
    mock_app.add_css_file.assert_called_with("demo-ui/test.css")
    mock_app.add_js_file.assert_called_with("demo-ui/test.js", type="module")

    monkeypatch.setattr(os.path, "exists", lambda p: False)
    build_frontend(mock_app)
    subprocess.check_call.assert_called()


def test_copy_frontend_assets(mock_app, monkeypatch):
    import os
    import shutil

    monkeypatch.setattr(os.path, "exists", lambda p: True)
    monkeypatch.setattr(os, "listdir", lambda p: ["test_file", "test_dir"])
    monkeypatch.setattr(os.path, "isdir", lambda p: p.endswith("test_dir"))
    monkeypatch.setattr(shutil, "copytree", MagicMock())
    monkeypatch.setattr(shutil, "copy2", MagicMock())
    monkeypatch.setattr(shutil, "rmtree", MagicMock())

    copy_frontend_assets(mock_app, None)
    shutil.rmtree.assert_called()
    shutil.copytree.assert_called()
    shutil.copy2.assert_called()

    # test skip on exc
    copy_frontend_assets(mock_app, Exception("Test"))

    # test skip on format
    mock_app.builder.format = "latex"
    copy_frontend_assets(mock_app, None)


def test_setup(mock_app):
    res = setup(mock_app)
    assert res["version"] == "0.1"
