import os

py_pkgs = set()
py_dir = "packages/python"
for p in os.listdir(py_dir):
    if os.path.isdir(os.path.join(py_dir, p)):
        py_pkgs.add(p)

cli_toml = "apps/cli/pyproject.toml"
used = set()
with open(cli_toml) as f:
    content = f.read()
    for p in py_pkgs:
        if p in content:
            used.add(p)

print("Python Packages not in CLI toml:")
print(py_pkgs - used)
