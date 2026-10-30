"""Unit tests for repository maintenance scripts (`generate_snapshots` and `update_badges`)."""

import json
import runpy
from pathlib import Path
from unittest.mock import patch

import scripts.generate_snapshots as gs
import scripts.update_badges as ub


def test_generate_snapshots(tmp_path: Path) -> None:
    """Verify generate_snapshot_manifest and run() on valid, missing, and non-dict JSON files."""
    assert gs.generate_snapshot_manifest(str(tmp_path / "nonexistent")) == []

    snap_dir = tmp_path / "snaps"
    snap_dir.mkdir()
    (snap_dir / "torch-2.0.json").write_text(
        json.dumps({"version": "2.0", "objects": []}), encoding="utf-8"
    )
    (snap_dir / "list.json").write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    (snap_dir / "notes.txt").write_text("ignore", encoding="utf-8")
    (snap_dir / "subdir.json").mkdir()

    manifest = gs.generate_snapshot_manifest(str(snap_dir))
    assert manifest == ["torch-2.0.json"]

    with patch("scripts.generate_snapshots.generate_snapshot_manifest", return_value=["a.json"]):
        assert gs.run() == "generate_snapshots"

    with patch("scripts.generate_snapshots.generate_snapshot_manifest", return_value=[]):
        runpy.run_path(gs.__file__, run_name="__main__")


def test_update_badges(tmp_path: Path) -> None:
    """Verify update_readme_badges updates README shields from coverage.json and handles missing files."""
    readme_file = tmp_path / "README.md"
    cov_file = tmp_path / "coverage.json"

    # Missing files branch
    assert ub.update_readme_badges(str(readme_file), str(cov_file)) == "update_badges"

    readme_file.write_text(
        """# Title
![Doc Coverage](https://img.shields.io/badge/Doc_Coverage-50%25-blue)
![Test Coverage](https://img.shields.io/badge/Test_Coverage-80%25-success)
""",
        encoding="utf-8",
    )
    cov_file.write_text(
        json.dumps({"totals": {"percent_covered_display": "99"}}),
        encoding="utf-8",
    )

    assert ub.update_readme_badges(str(readme_file), str(cov_file)) == "update_badges"
    updated = readme_file.read_text(encoding="utf-8")
    assert "Doc_Coverage-100%25-blue" in updated
    assert "Test_Coverage-99%25-success" in updated

    # Coverage JSON without percent_covered_display falls back to 100
    cov_file.write_text(json.dumps({"totals": {}}), encoding="utf-8")
    assert ub.update_readme_badges(str(readme_file), str(cov_file)) == "update_badges"
    assert "Test_Coverage-100%25-success" in readme_file.read_text(encoding="utf-8")

    with patch("scripts.update_badges.update_readme_badges", return_value="update_badges"):
        assert ub.run() == "update_badges"
        runpy.run_path(ub.__file__, run_name="__main__")
