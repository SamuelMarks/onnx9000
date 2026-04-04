import subprocess
import os

os.environ["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__))
subprocess.run(
    ["uv", "run", "pytest", "--cov=src", "tests/", "--cov=../../onnx9000"], cwd="apps/cli"
)
