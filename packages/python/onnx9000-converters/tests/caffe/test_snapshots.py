"""Tests for Caffe snapshots."""

import json
import os


def test_caffe_snapshot_exists_and_valid():
    """Test that the Caffe snapshot is readable and contains expected API objects."""
    snapshot_path = os.path.join(
        os.path.dirname(__file__), "../../../../../snapshots/caffe-0.1.0.json"
    )
    assert os.path.exists(snapshot_path), "Snapshot file not found"

    with open(snapshot_path) as f:
        data = json.load(f)

    assert "version" in data
    assert "objects" in data

    objects = data["objects"]
    names = [obj["name"] for obj in objects]

    # Check for core Caffe objects we might need to mock or emulate eventually
    assert "Net" in names
    assert "Layer" in names
    assert "Classifier" in names
