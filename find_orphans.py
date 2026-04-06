import json
import os
import re

roots = []
packages = {}

# Python packages
python_dir = "packages/python"
if os.path.exists(python_dir):
    for pkg in os.listdir(python_dir):
        pkg_path = os.path.join(python_dir, pkg)
        if os.path.isdir(pkg_path):
            packages[pkg] = []

# JS packages
js_dir = "packages/js"
if os.path.exists(js_dir):
    for pkg in os.listdir(js_dir):
        pkg_path = os.path.join(js_dir, pkg)
        if os.path.isdir(pkg_path):
            packages["@onnx9000/" + pkg] = []

# Apps
apps_dir = "apps"
if os.path.exists(apps_dir):
    for app in os.listdir(apps_dir):
        app_path = os.path.join(apps_dir, app)
        if os.path.isdir(app_path):
            roots.append("app:" + app)

roots.append("python_sdk:onnx9000")

edges = {node: [] for node in list(packages.keys()) + roots}

# Read py dependencies
for pkg in os.listdir(python_dir):
    pyproject_path = os.path.join(python_dir, pkg, "pyproject.toml")
    if os.path.exists(pyproject_path):
        with open(pyproject_path) as f:
            content = f.read()
            for dep in packages.keys():
                if dep in content and not dep.startswith("@onnx9000/"):
                    if dep != pkg:
                        edges[pkg].append(dep)

# Read JS dependencies
for pkg in os.listdir(js_dir):
    pkg_json_path = os.path.join(js_dir, pkg, "package.json")
    if os.path.exists(pkg_json_path):
        with open(pkg_json_path) as f:
            try:
                data = json.load(f)
                deps = list(data.get("dependencies", {}).keys()) + list(
                    data.get("devDependencies", {}).keys()
                )
                for dep in deps:
                    if dep in packages:
                        edges["@onnx9000/" + pkg].append(dep)
            except Exception:
                pass

# Read App dependencies
for app in os.listdir(apps_dir):
    # JS apps
    pkg_json_path = os.path.join(apps_dir, app, "package.json")
    if os.path.exists(pkg_json_path):
        with open(pkg_json_path) as f:
            try:
                data = json.load(f)
                deps = list(data.get("dependencies", {}).keys()) + list(
                    data.get("devDependencies", {}).keys()
                )
                for dep in deps:
                    if dep in packages:
                        edges["app:" + app].append(dep)
            except Exception:
                pass

# Python App/SDK
main_cli_path = "onnx9000/cli.py"
if os.path.exists(main_cli_path):
    with open(main_cli_path) as f:
        content = f.read()
        for dep in packages.keys():
            if dep in content and not dep.startswith("@onnx9000/"):
                edges["python_sdk:onnx9000"].append(dep)

# Reachability
visited = set()


def dfs(node):
    if node in visited:
        return
    visited.add(node)
    for neighbor in edges.get(node, []):
        dfs(neighbor)


for root in roots:
    dfs(root)

orphans = [pkg for pkg in packages if pkg not in visited]
print("Roots:", roots)
print("Orphans:")
for o in orphans:
    print(o)
