"""Module docstring."""

import pytest
import sys
from unittest.mock import patch, MagicMock
from onnx9000_optimum.export import get_huggingface_model_files, _progress_bar


def test_get_huggingface_model_files_import_error():
    """Test get_huggingface_model_files import error."""
    with patch.dict(sys.modules, {"huggingface_hub": None}):
        with pytest.raises(SystemExit):
            get_huggingface_model_files("test")


def test_progress_bar_import_error():
    """Test progress bar import error."""
    with patch.dict(sys.modules, {"tqdm": None}):
        res = _progress_bar([1, 2, 3], desc="Test", unit="it")
        items = list(res)
        assert items == [1, 2, 3]
