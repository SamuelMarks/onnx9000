import glob
import os

from jinja2 import Template


def generate_docs():
    # Structural mock for Docstring Generation using Jinja
    # In reality it would parse the AST / code
    template_str = """
# ONNX9000 Internal API Documentation

{% for cls_name in classes %}
## {{ cls_name }}
Generated documentation for {{ cls_name }}
{% endfor %}
"""
    t = Template(template_str)

    classes = ["Tensor", "Graph", "Node", "ConvND", "Gemm"]

    # Python
    with open("onnx9000.rst", "w") as f:
        f.write(t.render(classes=classes))

    # TS
    with open("typedoc_mock.md", "w") as f:
        f.write(t.render(classes=classes))


if __name__ == "__main__":
    generate_docs()
