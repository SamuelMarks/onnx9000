"""Tests for docs/tests/test_sphinx_ext_demo_ui.py."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sphinx_ext_demo_ui import (
    InteractiveDemoDirective,
    InteractiveDemoNode,
    build_frontend,
    copy_frontend_assets,
    depart_interactive_demo_node,
    setup,
    visit_interactive_demo_node,
)


class MockApp:
    """MockApp implementation."""

    def __init__(self, srcdir, outdir):
        """Perform   init   operation."""
        self.srcdir = srcdir
        self.outdir = outdir
        self.nodes = []
        self.directives = []
        self.css = []
        self.js = []
        self.events = []

        class Builder:
            """Builder implementation."""

            format = "html"

        self.builder = Builder()

    def add_node(self, node, html):
        """Perform add node operation."""
        self.nodes.append(node)

    def add_directive(self, name, directive):
        """Perform add directive operation."""
        self.directives.append((name, directive))

    def connect(self, event, cb):
        """Perform connect operation."""
        self.events.append((event, cb))

    def add_css_file(self, filename):
        """Perform add css file operation."""
        self.css.append(filename)

    def add_js_file(self, filename, **kwargs):
        """Perform add js file operation."""
        self.js.append(filename)


class MockSelf:
    """MockSelf implementation."""

    def __init__(self):
        """Perform   init   operation."""
        self.body = []


def test_setup():
    """Test setup."""
    app = MockApp(".", ".")
    res = setup(app)
    assert res["version"] == "0.1"
    assert res["parallel_read_safe"]
    assert len(app.nodes) == 1
    assert len(app.directives) == 1
    assert len(app.events) == 2


def test_visit_interactive_demo_node():
    """Test visit interactive demo node."""
    node = InteractiveDemoNode()
    node.attributes = {"initial-source": "keras", "ids": [], "classes": []}
    mock_self = MockSelf()
    visit_interactive_demo_node(mock_self, node)
    assert len(mock_self.body) == 1
    assert 'data-initial-source="keras"' in mock_self.body[0]
    assert 'class="onnx9000-demo-container"' in mock_self.body[0]


def test_depart_interactive_demo_node():
    """Test depart interactive demo node."""
    mock_self = MockSelf()
    node = InteractiveDemoNode()
    depart_interactive_demo_node(mock_self, node)


def test_interactive_demo_directive():
    """Test interactive demo directive."""

    class MockOptions:
        """MockOptions implementation."""

        def __init__(self):
            """Perform   init   operation."""
            self.options = {"initial-source": "tensorflow"}

    class MockStateMachine:
        """MockStateMachine implementation."""

        reporter = None

    directive = InteractiveDemoDirective(
        "test", [], MockOptions().options, None, None, None, None, None, MockStateMachine()
    )
    nodes = directive.run()
    assert len(nodes) == 1
    assert nodes[0]["initial-source"] == "tensorflow"
    assert nodes[0]["initial-target"] == "onnx"


from unittest.mock import patch


@patch("os.path.exists")
@patch("subprocess.check_call")
@patch("os.listdir")
def test_build_frontend(mock_listdir, mock_check_call, mock_exists):
    """Test build frontend."""
    app = MockApp(".", ".")
    mock_exists.return_value = False
    mock_listdir.return_value = ["app.css", "app.js", "app.umd.cjs"]
    build_frontend(app)
    mock_check_call.assert_called_once()
    assert "demo-ui/app.css" in app.css
    assert "demo-ui/app.js" in app.js
    assert "demo-ui/app.umd.cjs" not in app.js
    mock_check_call.reset_mock()
    mock_exists.return_value = True
    build_frontend(app)
    mock_check_call.assert_not_called()


@patch("os.path.exists")
@patch("shutil.copytree")
@patch("shutil.rmtree")
def test_copy_frontend_assets(mock_rmtree, mock_copytree, mock_exists):
    """Test copy frontend assets."""
    app = MockApp(".", ".")
    copy_frontend_assets(app, Exception())
    mock_copytree.assert_not_called()
    app.builder.format = "pdf"
    copy_frontend_assets(app, None)
    mock_copytree.assert_not_called()
    app.builder.format = "html"
    mock_exists.return_value = True
    copy_frontend_assets(app, None)
    mock_rmtree.assert_called_once()
    mock_copytree.assert_called_once()
    mock_rmtree.reset_mock()
    mock_copytree.reset_mock()
    mock_exists.return_value = False
    copy_frontend_assets(app, None)
    mock_rmtree.assert_not_called()
    mock_copytree.assert_called_once()
