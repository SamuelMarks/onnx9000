"""Tests for sphinx_demo extension."""

import os
import shutil
from unittest.mock import MagicMock, patch

from docutils import nodes
from onnx9000.sphinx_demo import (
    InteractiveDemoDirective,
    InteractiveDemoNode,
    build_frontend,
    copy_frontend_assets,
    depart_interactive_demo_node,
    setup,
    visit_interactive_demo_node,
)


def test_node_and_visitor() -> None:
    """Test the node instantiation and HTML visitor methods."""
    node = InteractiveDemoNode()
    node["initial-source"] = "keras"
    node["initial-target"] = "onnx"

    class MockTranslator:
        def __init__(self):
            self.body = []

    translator = MockTranslator()

    visit_interactive_demo_node(translator, node)
    assert len(translator.body) == 1
    html = translator.body[0]
    assert 'data-initial-source="keras"' in html
    assert 'data-initial-target="onnx"' in html
    assert "interactive-demo-container" in html

    # Depart does nothing but should not crash
    depart_interactive_demo_node(translator, node)


def test_directive() -> None:
    """Test the directive logic."""

    class MockDirective(InteractiveDemoDirective):
        def __init__(self):
            self.options = {"initial-source": "jax", "initial-target": "c"}

    directive = MockDirective()
    res = directive.run()
    assert len(res) == 1
    node = res[0]
    assert isinstance(node, InteractiveDemoNode)
    assert node["initial-source"] == "jax"
    assert node["initial-target"] == "c"

    # Defaults
    class MockDirectiveDefaults(InteractiveDemoDirective):
        def __init__(self):
            self.options = {}

    directive2 = MockDirectiveDefaults()
    res2 = directive2.run()
    node2 = res2[0]
    assert node2["initial-source"] == "keras"
    assert node2["initial-target"] == "onnx"


@patch("os.listdir")
@patch("subprocess.check_call")
@patch("os.path.exists")
def test_build_frontend(mock_exists, mock_check_call, mock_listdir) -> None:
    """Test frontend build hook."""
    app = MagicMock()
    app.srcdir = "/fake/docs"
    mock_listdir.return_value = []

    # Condition: dist does not exist
    mock_exists.return_value = False

    build_frontend(app)
    mock_check_call.assert_called_once()
    assert "build" in mock_check_call.call_args[0][0]

    # Reset
    mock_check_call.reset_mock()
    mock_exists.return_value = True

    # Condition: dist exists, no FORCE_FRONTEND_BUILD -> no build
    with patch.dict(os.environ, {}, clear=True):
        build_frontend(app)
        mock_check_call.assert_not_called()

    # Condition: dist exists, FORCE_FRONTEND_BUILD=1 -> build
    with patch.dict(os.environ, {"FORCE_FRONTEND_BUILD": "1"}, clear=True):
        build_frontend(app)
        mock_check_call.assert_called_once()


@patch("os.listdir")
@patch("os.path.exists")
def test_build_frontend_adds_files(mock_exists, mock_listdir) -> None:
    app = MagicMock()
    app.srcdir = "/fake/docs"
    mock_exists.return_value = True
    mock_listdir.return_value = ["style.css", "main.js", "main.umd.cjs"]

    with patch("subprocess.check_call"):
        build_frontend(app)

    app.add_css_file.assert_called_once_with("demo-ui/style.css")
    app.add_js_file.assert_called_once_with("demo-ui/main.js", type="module")


@patch("shutil.copy2")
@patch("shutil.copytree")
@patch("shutil.rmtree")
@patch("os.listdir")
@patch("os.path.exists")
def test_copy_frontend_assets(
    mock_exists, mock_listdir, mock_rmtree, mock_copytree, mock_copy2
) -> None:
    app = MagicMock()
    app.srcdir = "/fake/docs"
    app.outdir = "/fake/docs/_build"
    app.builder.format = "html"

    # If exception -> return
    copy_frontend_assets(app, Exception("err"))
    mock_copytree.assert_not_called()

    # If not HTML -> return
    app.builder.format = "text"
    copy_frontend_assets(app, None)
    mock_copytree.assert_not_called()

    # Happy path
    app.builder.format = "html"

    # Return true for all exists
    mock_exists.return_value = True
    mock_listdir.return_value = ["pyodide.asm.js", "pyodide.asm.wasm", "some_dir"]

    def side_effect_isdir(path):
        return path.endswith("some_dir")

    with patch("os.path.isdir", side_effect=side_effect_isdir):
        copy_frontend_assets(app, None)

    # static dir was removed and copied
    mock_rmtree.assert_called_once()
    assert mock_copytree.call_count == 2  # 1 for dist, 1 for some_dir in pyodide
    assert mock_copy2.call_count == 2  # 2 files in pyodide


def test_setup() -> None:
    app = MagicMock()
    res = setup(app)

    assert res["version"] == "0.1"
    app.add_node.assert_called_once()
    app.add_directive.assert_called_once_with("interactive-demo", InteractiveDemoDirective)
    assert app.connect.call_count == 2
