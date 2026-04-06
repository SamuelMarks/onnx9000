import os
import json

py_dir = "packages/python"
py_pkgs = [p for p in os.listdir(py_dir) if os.path.isdir(os.path.join(py_dir, p))]

js_dir = "packages/js"
js_pkgs = [p for p in os.listdir(js_dir) if os.path.isdir(os.path.join(js_dir, p))]

py_import_names = {}
for p in py_pkgs:
    name = p.replace("onnx9000-", "").replace("-", "_")
    py_import_names[p] = [f"onnx9000.{name}", f"onnx9000_{name}"]

js_import_names = {}
for p in js_pkgs:
    try:
        with open(os.path.join(js_dir, p, "package.json")) as f:
            data = json.load(f)
            js_import_names[p] = data["name"]
    except:
        js_import_names[p] = f"@onnx9000/{p}"


def get_deps(pkg_dir, is_js=False):
    deps = []
    if is_js:
        pkg_json = os.path.join(pkg_dir, "package.json")
        if os.path.exists(pkg_json):
            try:
                with open(pkg_json) as f:
                    data = json.load(f)
                    deps.extend(data.get("dependencies", {}).keys())
                    deps.extend(data.get("devDependencies", {}).keys())
            except:
                pass
    else:
        pyproject = os.path.join(pkg_dir, "pyproject.toml")
        if os.path.exists(pyproject):
            try:
                with open(pyproject) as f:
                    content = f.read()
                    for p in py_pkgs:
                        if p in content and p != os.path.basename(pkg_dir):
                            deps.append(p)
            except:
                pass
    return deps


edges = {}
for p in py_pkgs:
    edges[p] = get_deps(os.path.join(py_dir, p), False)
for p in js_pkgs:
    edges[js_import_names[p]] = get_deps(os.path.join(js_dir, p), True)

# Find direct usage in apps/ and onnx9000/
direct_uses = set()


def scan_dir(d):
    for root, _, files in os.walk(d):
        for f in files:
            if not (
                f.endswith(".py")
                or f.endswith(".ts")
                or f.endswith(".tsx")
                or f.endswith(".js")
                or f.endswith(".html")
            ):
                continue
            path = os.path.join(root, f)
            try:
                with open(path, "r", encoding="utf-8") as file:
                    content = file.read()
                    for p, names in py_import_names.items():
                        for name in names:
                            if name in content:
                                direct_uses.add(p)
                    for p, name in js_import_names.items():
                        if name in content:
                            direct_uses.add(js_import_names[p])
            except:
                pass


scan_dir("apps")
scan_dir("onnx9000")
scan_dir("onnx9000_workspace")

visited = set()


def dfs(node):
    if node in visited:
        return
    visited.add(node)
    for neighbor in edges.get(node, []):
        dfs(neighbor)


for u in direct_uses:
    dfs(u)

orphans = []
for p in py_pkgs:
    if p not in visited:
        orphans.append(("Python", p))
for p in js_pkgs:
    name = js_import_names[p]
    if name not in visited:
        orphans.append(("JS/TS", name))

print("Direct uses:", direct_uses)
print("Orphans:")
for t, name in orphans:
    print(f"[{t}] {name}")
