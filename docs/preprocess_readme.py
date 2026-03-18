import os

docs_dir = os.path.dirname(__file__)
readme_src = os.path.join(docs_dir, "..", "README.md")
readme_dest = os.path.join(docs_dir, "README_GENERATED.md")

with open(readme_src, "r") as f:
    content = f.read()

# Replace paths to point to the js-api/_media versions
content = content.replace("](./specs/", "](js-api/_media/")
content = content.replace("](./USAGE.md", "](js-api/_media/USAGE.md")
content = content.replace("](./ARCHITECTURE.md", "](js-api/_media/ARCHITECTURE.md")
content = content.replace("](./ROADMAP.md", "](js-api/_media/ROADMAP.md")

with open(readme_dest, "w") as f:
    f.write(content)
