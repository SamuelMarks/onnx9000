import glob
import os

from jinja2 import Template


def generate_docs():
    template_str = """
# ONNX9000 Internal API Documentation

{% for cls_name in classes %}
## {{ cls_name }}
Generated documentation for {{ cls_name }}
{% endfor %}
"""
    t = Template(template_str)
    classes = ["Tensor", "Graph", "Node", "ConvND", "Gemm"]
    with open("onnx9000.rst", "w") as f:
        f.write(t.render(classes=classes))
    with open("typedoc_mock.md", "w") as f:
        f.write(t.render(classes=classes))


def generate_readme():
    docs_dir = os.path.dirname(__file__)
    readme_src = os.path.join(docs_dir, "..", "README.md")
    readme_dest = os.path.join(docs_dir, "README_GENERATED.md")
    with open(readme_src) as f:
        content = f.read()
    content = content.replace("](./specs/", "](js-api/_media/")
    content = content.replace("](./USAGE.md", "](js-api/_media/USAGE.md")
    content = content.replace("](./ARCHITECTURE.md", "](js-api/_media/ARCHITECTURE.md")
    content = content.replace("](./ROADMAP.md", "](js-api/_media/ROADMAP.md")
    with open(readme_dest, "w") as f:
        f.write(content)


if __name__ == "__main__":
    generate_docs()
    generate_readme()
