"""Module containing fix_tests.py definitions."""

import glob

for filepath in glob.glob("tests/**/*.py", recursive=True):
    with open(filepath) as f:
        content = f.read()
    content = content.replace(
        "from onnx9000.converters.frontend.builder import GraphBuilder, Tracing\\nfrom onnx9000.core.ir import Tensor",
        "from onnx9000.converters.frontend.builder import GraphBuilder, Tracing\nfrom onnx9000.core.ir import Tensor",
    )
    with open(filepath, "w") as f:
        f.write(content)
