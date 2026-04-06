import re

with open("MODEL_ZOO_PLAN8.md", "r") as f:
    lines = f.readlines()

models = []
for line in lines:
    m = re.match(r"^  - \[ \] (.*)", line)
    if m:
        names = m.group(1).split(",")
        for n in names:
            n = n.strip().replace('"', "")
            if n:
                models.append(n)

models_str = ",\n    ".join(f'"{m}"' for m in models)

with open("packages/python/onnx9000-zoo/tests/test_zoo_matrix.py", "r") as f:
    content = f.read()

import re

content = re.sub(
    r"VISION_MODELS = \[.*?\]", f"VISION_MODELS = [\n    {models_str}\n]", content, flags=re.DOTALL
)

with open("packages/python/onnx9000-zoo/tests/test_zoo_matrix.py", "w") as f:
    f.write(content)
