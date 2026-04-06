from pathlib import Path

content = Path("packages/python/onnx9000-core/src/onnx9000/core/exporter.py").read_text()
old_str = """    elif format == "keras":
        code = generate_keras(graph)
        with open(output_path, "w") as f:
            f.write(code)"""

new_str = """    elif format == "keras":
        code = generate_keras(graph)
        with open(output_path, "w") as f:
            f.write(code)
        # Unit Testing Constraints: `editorconfig` formatters must be run programmatically
        try:
            import subprocess
            subprocess.run(["ruff", "format", output_path], check=False, capture_output=True)
            subprocess.run(["ruff", "check", "--fix", output_path], check=False, capture_output=True)
        except Exception:
            pass"""

if "subprocess.run" not in content:
    content = content.replace(old_str, new_str)
    Path("packages/python/onnx9000-core/src/onnx9000/core/exporter.py").write_text(content)
