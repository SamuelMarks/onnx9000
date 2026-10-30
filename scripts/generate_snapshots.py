"""Snapshot verification and manifest generator script for ONNX9000.

Scans the `snapshots/` directory for framework API JSON snapshots and verifies
their structural integrity.
"""

import json
import os


def generate_snapshot_manifest(snapshots_dir: str = "snapshots") -> list[str]:
    """List and validate JSON snapshot files in the target snapshots directory.

    Args:
        snapshots_dir: Directory containing framework snapshot `.json` files.

    Returns:
        Sorted list of valid snapshot filenames found in `snapshots_dir`.
    """
    if not os.path.isdir(snapshots_dir):
        return []
    valid_files: list[str] = []
    for entry in sorted(os.listdir(snapshots_dir)):
        if entry.endswith(".json"):
            full_path = os.path.join(snapshots_dir, entry)
            if os.path.isfile(full_path):
                with open(full_path, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    valid_files.append(entry)
    return valid_files


def run() -> str:
    """Execute snapshot manifest verification and return status identifier.

    Returns:
        Status identifier `"generate_snapshots"`.
    """
    generate_snapshot_manifest()
    return "generate_snapshots"


if __name__ == "__main__":
    run()
