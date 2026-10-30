"""Tests for documentation scripts and code generators."""

import os
import sys
from unittest.mock import MagicMock, mock_open, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_conf_py() -> None:
    """Verify Sphinx docs/conf.py setup and TypeDoc hook registration."""
    from docs import conf

    app_mock = MagicMock()
    conf.setup(app_mock)
    app_mock.connect.assert_called_with("builder-inited", conf.run_typedoc)
    app_mock.add_lexer.assert_called()

    with patch("subprocess.run") as mock_run:
        conf.run_typedoc(app_mock)
        mock_run.assert_called()


def test_preprocess_readme() -> None:
    """Verify README and API docs preprocessing functions."""
    import docs.preprocess_readme as pr

    with patch(
        "builtins.open",
        mock_open(read_data="<!-- DOCS_API_START -->\n<!-- DOCS_API_END -->"),
    ) as m_open:
        pr.generate_readme()
        m_open.assert_called()
    pr.generate_docs()


def test_generate_toc() -> None:
    """Verify table of contents generation for JS API Markdown docs."""
    with patch("builtins.open", mock_open()) as m_open:
        with patch("glob.glob", return_value=["docs/js-api/README.md", "docs/js-api/test.md"]):
            import importlib

            import docs.generate_toc

            importlib.reload(docs.generate_toc)
            m_open.assert_called()


def test_fix_box() -> None:
    """Verify docs/fix_box.py anchor link normalization."""
    with patch("glob.glob", return_value=["docs/js-api/test.md"]):
        with patch("builtins.open", mock_open(read_data="(Box.md#height)")) as m_open:
            import importlib

            import docs.fix_box

            importlib.reload(docs.fix_box)
            m_open().write.assert_called_with("(Box.md)")
