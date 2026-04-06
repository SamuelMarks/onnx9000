import ast
import os


def get_all_py_files(d):
    res = []
    for root, _, files in os.walk(d):
        for f in files:
            if f.endswith(".py"):
                res.append(os.path.join(root, f))
    return res


cli_files = get_all_py_files("apps/cli/src/onnx9000_cli")
cli_files = [f for f in cli_files if "coverage" not in f and "test" not in f]

imported_modules = set()

for fpath in cli_files:
    with open(fpath) as f:
        try:
            tree = ast.parse(f.read(), filename=fpath)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imported_modules.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imported_modules.add(node.module)
        except Exception:
            pass

print("Imported onnx9000 modules in CLI:")
for m in sorted(imported_modules):
    if "onnx9000" in m:
        print(m)
