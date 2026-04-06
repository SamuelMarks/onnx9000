import importlib
import inspect
import json
import sys

fw = sys.argv[1]
out_path = sys.argv[2]
objects = []
version = "unknown"

try:
    import importlib.metadata

    version = importlib.metadata.version(fw)
except Exception:
    pass

try:
    mod = importlib.import_module(fw)
    if hasattr(mod, "__version__"):
        version = mod.__version__
except Exception:
    pass

try:
    import griffe

    mod = griffe.load(fw)
    for name, member in mod.members.items():
        if str(name).startswith("_"):
            continue
        if member.is_function:
            sig = "(" + ", ".join(p.name for p in member.parameters) + ")"
            objects.append({"name": str(name), "type": "Function", "signature": sig})
        elif member.is_class:
            objects.append({"name": str(name), "type": "Class", "signature": ""})
        else:
            objects.append({"name": str(name), "type": "Object", "signature": ""})
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"version": version, "objects": objects}, f)
except Exception:
    try:
        mod = importlib.import_module(fw)
        visited_mods = set()

        def inspect_traverse(m, prefix=""):
            if id(m) in visited_mods:
                return
            visited_mods.add(id(m))
            members = getattr(m, "__all__", [n for n in dir(m) if not n.startswith("_")])
            members = [
                n
                for n in members
                if n
                not in {
                    "python",
                    "compiler",
                    "internal",
                    "core",
                    "tests",
                    "testing",
                    "experimental",
                    "contrib",
                    "compat",
                    "backends",
                    "backend",
                    "ops",
                    "legacy",
                    "src",
                    "tools",
                    "utils",
                    "fluid",
                }
            ]
            for name in members:
                if str(name).startswith("_"):
                    continue
                try:
                    obj = getattr(m, name)
                    if inspect.isroutine(obj) or inspect.isbuiltin(obj):
                        try:
                            sig = str(inspect.signature(obj))
                        except ValueError:
                            sig = "(...)"
                        objects.append(
                            {"name": prefix + str(name), "type": "Function", "signature": sig}
                        )
                    elif inspect.isclass(obj):
                        try:
                            sig = str(inspect.signature(obj))
                        except (ValueError, TypeError):
                            sig = "(...)"
                        objects.append(
                            {"name": prefix + str(name), "type": "Class", "signature": sig}
                        )
                    elif inspect.ismodule(obj) and getattr(obj, "__name__", "").startswith(
                        fw + "."
                    ):
                        inspect_traverse(obj, prefix + str(name) + ".")
                    else:
                        objects.append(
                            {"name": prefix + str(name), "type": "Object", "signature": ""}
                        )
                except Exception:
                    objects.append({"name": prefix + str(name), "type": "Object", "signature": ""})

        inspect_traverse(mod)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"version": version, "objects": objects}, f)
    except Exception:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"version": "Not Installed", "objects": []}, f)
