import json
import os

js_pkgs = set()
js_dir = "packages/js"
for p in os.listdir(js_dir):
    try:
        with open(os.path.join(js_dir, p, "package.json")) as f:
            data = json.load(f)
            js_pkgs.add(data["name"])
    except Exception:
        pass

used_in_apps = set()
apps_dir = "apps"
for app in os.listdir(apps_dir):
    pkg_json = os.path.join(apps_dir, app, "package.json")
    if os.path.exists(pkg_json):
        try:
            with open(pkg_json) as f:
                data = json.load(f)
                deps = list(data.get("dependencies", {}).keys()) + list(
                    data.get("devDependencies", {}).keys()
                )
                for dep in deps:
                    used_in_apps.add(dep)
        except Exception:
            pass

print("JS Packages not in any app:")
print(js_pkgs - used_in_apps)
