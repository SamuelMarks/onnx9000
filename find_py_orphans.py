import os
import toml

py_packages = set()
py_dir = "packages/python"
if os.path.exists(py_dir):
    for pkg in os.listdir(py_dir):
        pkg_path = os.path.join(py_dir, pkg)
        if os.path.isdir(pkg_path):
            py_packages.add(pkg)

referenced = set()


def read_deps(toml_path):
    try:
        with open(toml_path) as f:
            data = toml.load(f)
            # check project.dependencies
            deps = data.get("project", {}).get("dependencies", [])
            for dep in deps:
                # dependencies might have versions, so we just check if pkg name is in dep string
                for pkg in py_packages:
                    if pkg in dep:
                        referenced.add(pkg)
            # check tool.uv.sources
            sources = data.get("tool", {}).get("uv", {}).get("sources", {})
            for pkg in py_packages:
                if pkg in sources:
                    referenced.add(pkg)
    except:
        pass


# Check apps
apps_dir = "apps"
if os.path.exists(apps_dir):
    for app in os.listdir(apps_dir):
        pyproject_path = os.path.join(apps_dir, app, "pyproject.toml")
        if os.path.exists(pyproject_path):
            read_deps(pyproject_path)

# Check other py packages
for pkg in py_packages:
    pyproject_path = os.path.join(py_dir, pkg, "pyproject.toml")
    if os.path.exists(pyproject_path):
        read_deps(pyproject_path)

# Check root pyproject.toml
read_deps("pyproject.toml")

# Check root uv.lock (just text search)
if os.path.exists("uv.lock"):
    with open("uv.lock") as f:
        content = f.read()
        for pkg in py_packages:
            if f'name = "{pkg}"' in content:
                # Wait, this just means it exists. Not who uses it.
                pass

orphans = py_packages - referenced
print("Python Orphans:", orphans)
