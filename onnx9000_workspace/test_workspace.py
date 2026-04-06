"""Tests for workspace."""

import os
import tempfile

from onnx9000_workspace import setup_workspace


def test_setup_workspace():
    """Test setup workspace."""
    with tempfile.TemporaryDirectory() as tmpdir:
        setup_workspace(tmpdir)
        assert os.path.exists(os.path.join(tmpdir, "models"))
        assert os.path.exists(os.path.join(tmpdir, "configs", "onnx9000.yaml"))
