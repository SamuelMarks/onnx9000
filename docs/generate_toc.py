import os
import glob

docs_dir = os.path.dirname(__file__)
toc_file = os.path.join(docs_dir, "js-api_toc.rst")

with open(toc_file, "w") as f:
    f.write("JS API\n======\n\n.. toctree::\n   :hidden:\n\n")

    # All markdown files inside js-api and its subfolders
    pattern = os.path.join(docs_dir, "js-api", "**", "*.md")
    for md_file in glob.glob(pattern, recursive=True):
        rel_path = os.path.relpath(md_file, docs_dir).replace("\\", "/")
        if rel_path == "js-api/README.md":
            continue  # Already explicitly referenced in index.rst
        f.write(f"   {rel_path}\n")
