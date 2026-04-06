import re

with open("packages/python/onnx9000-converters/src/onnx9000/converters/jax/jax_ops.py") as f:
    content = f.read()

real_ops_str = ""
blocks = content.split("@register_op")
for block in blocks[1:]:
    if 'op_type="Identity"' not in block and "op_type='Identity'" not in block:
        real_ops_str += "@register_op" + block

with open(
    "packages/python/onnx9000-converters/src/onnx9000/converters/jax/jax_ops.py.real", "w"
) as f:
    f.write('"""Module providing core logic and structural definitions for jax ops."""\n\n')
    f.write("from typing import Any\n\n")
    f.write("from onnx9000.core.ir import Node\n")
    f.write("from onnx9000.core.registry import register_op\n\n\n")
    f.write(real_ops_str)
