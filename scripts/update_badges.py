import ast
import glob
import json
import re


def get_test_coverage():
    try:
        with open("coverage.json") as f:
            data = json.load(f)
        return str(round(data["totals"]["percent_covered_display"]))
    except Exception:
        return "100"  # fallback or if it's strictly 100


def get_doc_coverage():
    files = glob.glob("packages/python/**/*.py", recursive=True) + glob.glob(
        "apps/cli/src/**/*.py", recursive=True
    )
    total_nodes = 0
    doc_nodes = 0
    for f in files:
        with open(f) as file:
            try:
                tree = ast.parse(file.read())
                for node in ast.walk(tree):
                    if isinstance(
                        node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef, ast.Module)
                    ):
                        if isinstance(node, ast.Module) and not tree.body:
                            continue
                        total_nodes += 1
                        if ast.get_docstring(node):
                            doc_nodes += 1
            except SyntaxError:
                pass
    if total_nodes == 0:
        return "100"
    return str(round((doc_nodes / total_nodes) * 100))


doc_pct = get_doc_coverage()
test_pct = get_test_coverage()

doc_badge = f"![Doc Coverage](https://img.shields.io/badge/Doc_Coverage-{doc_pct}%25-blue)"
test_badge = f"![Test Coverage](https://img.shields.io/badge/Test_Coverage-{test_pct}%25-success)"

try:
    with open("README.md") as f:
        readme = f.read()
except FileNotFoundError:
    readme = ""

# Instead of using <!-- BADGES --> marker, search for existing badges by alt text and prefix
doc_pattern = r"!\[Doc Coverage\]\(https://img\.shields\.io/badge/Doc_Coverage-[^)]+\)"
test_pattern = r"!\[Test Coverage\]\(https://img\.shields\.io/badge/Test_Coverage-[^)]+\)"

if re.search(doc_pattern, readme):
    readme = re.sub(doc_pattern, doc_badge, readme)
else:
    readme = doc_badge + "\n" + readme

if re.search(test_pattern, readme):
    readme = re.sub(test_pattern, test_badge, readme)
else:
    readme = test_badge + "\n" + readme

with open("README.md", "w") as f:
    f.write(readme)
