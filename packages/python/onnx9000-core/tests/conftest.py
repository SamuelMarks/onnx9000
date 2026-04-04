"""Module docstring."""

import os
import sys

# Add all workspace python package src directories to sys.path
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
packages_dir = os.path.join(root, "packages", "python")
for pkg in os.listdir(packages_dir):
    src_dir = os.path.join(packages_dir, pkg, "src")
    if os.path.isdir(src_dir):
        sys.path.insert(0, src_dir)

# Also add apps src if any have python code
apps_dir = os.path.join(root, "apps")
for app in os.listdir(apps_dir):
    src_dir = os.path.join(apps_dir, app, "src")
    if os.path.isdir(src_dir):
        sys.path.insert(0, src_dir)
