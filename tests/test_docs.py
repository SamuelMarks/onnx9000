"""Tests for documentation scripts and code generators."""

import os
import sys
import tempfile
from unittest.mock import MagicMock, mock_open, patch

import pytest

# Add docs to path so we can import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_conf_py():
    """Test conf.py."""
    import docs.conf as conf

    # test setup
    app_mock = MagicMock()
    conf.setup(app_mock)
    app_mock.connect.assert_called_with("builder-inited", conf.run_typedoc)
    app_mock.add_lexer.assert_called()

    # test run_typedoc
    with patch("subprocess.run") as mock_run:
        conf.run_typedoc(app_mock)
        mock_run.assert_called()


def test_preprocess_readme():
    """Test preprocess_readme.py."""
    import docs.preprocess_readme as pr

    with patch(
        "builtins.open",
        mock_open(read_data="<!-- DOCS_API_START -->\n<!-- DOCS_API_END -->"),
    ):
        pr.generate_readme()
    pr.generate_docs()


def test_generate_toc():
    """Test generate_toc.py."""
    with patch("builtins.open", mock_open()):
        with patch("glob.glob", return_value=["docs/js-api/README.md", "docs/js-api/test.md"]):
            import docs.generate_toc


def test_gen_py():
    """Test gen.py."""
    with patch("os.makedirs"):
        with patch("builtins.open", mock_open()):
            import gen


def test_fix_box():
    """Test fix_box.py."""
    with patch("glob.glob", return_value=["docs/js-api/test.md"]):
        with patch("builtins.open", mock_open(read_data="(Box.md#height)")) as m_open:
            import docs.fix_box

            m_open().write.assert_called_with("(Box.md)")
