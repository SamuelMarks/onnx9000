import ast
import os


def scan_imports(d):
    imports = set()
    for root, _, files in os.walk(d):
        for f in files:
            if not f.endswith(".py"):
                continue
            path = os.path.join(root, f)
            try:
                with open(path) as file:
                    tree = ast.parse(file.read())
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for n in node.names:
                                imports.add(n.name)
                        elif isinstance(node, ast.ImportFrom) and node.module:
                            imports.add(node.module)
            except Exception:
                pass
    return imports


cli_imports = scan_imports("apps/cli/src/onnx9000_cli")
sdk_imports = scan_imports("onnx9000_workspace")
# add manual imports known
all_imports = cli_imports | sdk_imports

converters = ["jax", "paddle", "sklearn", "safetensors", "mmdnn", "torch", "tensorflow"]
for c in converters:
    mod = f"onnx9000.converters.{c}"
    used = any(i.startswith(mod) for i in all_imports)
    if not used:
        print(f"Unwired: {mod}")

backends = ["cpu", "cuda", "ffi", "codegen", "testing", "memory"]
for b in backends:
    mod = f"onnx9000.backends.{b}"
    used = any(i.startswith(mod) for i in all_imports)
    if not used:
        print(f"Unwired: {mod}")
