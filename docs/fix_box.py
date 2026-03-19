import glob
import os

for file in glob.glob(
    os.path.join(os.path.dirname(__file__), "js-api", "**", "*.md"), recursive=True
):
    with open(file) as f:
        content = f.read()
    if "(Box.md#" in content:
        content = content.replace("(Box.md#height)", "(Box.md)")
        content = content.replace("(Box.md#width)", "(Box.md)")
        content = content.replace("(Box.md#x)", "(Box.md)")
        content = content.replace("(Box.md#y)", "(Box.md)")
        with open(file, "w") as f:
            f.write(content)
