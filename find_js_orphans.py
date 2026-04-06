import os
import json

js_packages = set()
js_dir = "packages/js"
if os.path.exists(js_dir):
    for pkg in os.listdir(js_dir):
        pkg_path = os.path.join(js_dir, pkg)
        if os.path.isdir(pkg_path):
            try:
                with open(os.path.join(pkg_path, "package.json")) as f:
                    data = json.load(f)
                    js_packages.add(data["name"])
            except:
                pass

referenced = set()

# Check apps
apps_dir = "apps"
if os.path.exists(apps_dir):
    for app in os.listdir(apps_dir):
        pkg_json_path = os.path.join(apps_dir, app, "package.json")
        if os.path.exists(pkg_json_path):
            try:
                with open(pkg_json_path) as f:
                    data = json.load(f)
                    deps = list(data.get("dependencies", {}).keys()) + list(
                        data.get("devDependencies", {}).keys()
                    )
                    for dep in deps:
                        referenced.add(dep)
            except:
                pass

# Check other js packages
for pkg in os.listdir(js_dir):
    pkg_json_path = os.path.join(js_dir, pkg, "package.json")
    if os.path.exists(pkg_json_path):
        try:
            with open(pkg_json_path) as f:
                data = json.load(f)
                deps = list(data.get("dependencies", {}).keys()) + list(
                    data.get("devDependencies", {}).keys()
                )
                for dep in deps:
                    referenced.add(dep)
        except:
            pass

orphans = js_packages - referenced
print("Orphans:", orphans)
