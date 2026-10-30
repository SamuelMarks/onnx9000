"""Pytest configuration for CLI tests."""

from __future__ import annotations

import os
import sys
import types
from unittest.mock import MagicMock

import pytest

root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
packages_dir = os.path.join(root, "packages", "python")
for pkg in os.listdir(packages_dir):
    src_dir = os.path.join(packages_dir, pkg, "src")
    if os.path.isdir(src_dir) and src_dir not in sys.path:
        sys.path.insert(0, src_dir)

cli_src = os.path.join(root, "apps", "cli", "src")
if cli_src not in sys.path:
    sys.path.insert(0, cli_src)


@pytest.fixture(autouse=True)
def _cli_mock_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide mock torch modules during CLI tests without polluting other test suites."""
    if "torch" not in sys.modules:
        torch_mod = types.ModuleType("torch")
        torch_nn = types.ModuleType("torch.nn")

        class _MockTorchModule:
            """Mock torch.nn.Module base class for CLI tests."""

        torch_nn.Module = _MockTorchModule  # type: ignore[attr-defined]
        torch_export = types.ModuleType("torch.export")
        torch_export.load = MagicMock()  # type: ignore[attr-defined]
        torch_fx = types.ModuleType("torch.fx")
        torch_fx.symbolic_trace = MagicMock()  # type: ignore[attr-defined]

        torch_mod.nn = torch_nn  # type: ignore[attr-defined]
        torch_mod.export = torch_export  # type: ignore[attr-defined]
        torch_mod.fx = torch_fx  # type: ignore[attr-defined]
        torch_mod.load = MagicMock()  # type: ignore[attr-defined]
        torch_mod.__version__ = "2.0.0"  # type: ignore[attr-defined]

        monkeypatch.setitem(sys.modules, "torch", torch_mod)
        monkeypatch.setitem(sys.modules, "torch.nn", torch_nn)
        monkeypatch.setitem(sys.modules, "torch.export", torch_export)
        monkeypatch.setitem(sys.modules, "torch.fx", torch_fx)
