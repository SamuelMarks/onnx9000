import os

packages = [
    ("ort-training", "ORT Training"),
    ("olive-optimizer", "Olive Optimizer"),
    ("triton-server", "Triton Server"),
    ("onnx-tool", "ONNX Tool"),
]

for pkg_name, title in packages:
    # Python
    py_mod = f"onnx9000_{pkg_name.replace('-', '_')}"
    base_dir = f"packages/python/onnx9000-{pkg_name}"
    os.makedirs(f"{base_dir}/src/{py_mod}", exist_ok=True)
    os.makedirs(f"{base_dir}/tests", exist_ok=True)

    with open(f"{base_dir}/pyproject.toml", "w") as f:
        f.write(f"""[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "onnx9000-{pkg_name}"
version = "1.0.0"
description = "{title} for ONNX9000"
readme = "README.md"
requires-python = ">=3.8"
dependencies = [
    "onnx>=1.14.0",
]

[tool.hatch.build.targets.wheel]
packages = ["src/{py_mod}"]
""")

    with open(f"{base_dir}/README.md", "w") as f:
        f.write(f"# ONNX9000 {title}\n\nSDK for {title}.")

    class_name = title.replace(" ", "")
    with open(f"{base_dir}/src/{py_mod}/__init__.py", "w") as f:
        f.write(
            '"""' + title + ' package."""\n\n'
            "class " + class_name + ":\n"
            "    def process(self, input_data: str) -> str:\n"
            '        return "' + title + ' processed " + input_data\n'
        )

    with open(f"{base_dir}/tests/test_main.py", "w") as f:
        f.write(
            "import unittest\n"
            "from " + py_mod + " import " + class_name + "\n\n"
            "class Test" + class_name + "(unittest.TestCase):\n"
            "    def test_process(self):\n"
            "        obj = " + class_name + "()\n"
            '        self.assertEqual(obj.process("test"), "' + title + ' processed test")\n\n'
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        )

    # JS
    js_dir = f"packages/js/{pkg_name}"
    os.makedirs(f"{js_dir}/src", exist_ok=True)
    os.makedirs(f"{js_dir}/tests", exist_ok=True)

    with open(f"{js_dir}/package.json", "w") as f:
        f.write(f"""{{
  "name": "@onnx9000/{pkg_name}",
  "version": "1.0.0",
  "main": "dist/index.js",
  "types": "dist/index.d.ts",
  "scripts": {{
    "build": "tsc",
    "test": "vitest run"
  }},
  "devDependencies": {{
    "typescript": "^5.0.0",
    "vitest": "^1.0.0"
  }}
}}
""")

    with open(f"{js_dir}/tsconfig.json", "w") as f:
        f.write("""{
  "compilerOptions": {
    "target": "es2022",
    "module": "commonjs",
    "declaration": true,
    "outDir": "./dist",
    "strict": true
  },
  "include": ["src"]
}
""")

    with open(f"{js_dir}/src/index.ts", "w") as f:
        f.write(
            "export class " + class_name + " {\n"
            "  process(input: string): string {\n"
            "    return '" + title + " processed ' + input;\n"
            "  }\n"
            "}\n"
        )

    with open(f"{js_dir}/tests/main.test.ts", "w") as f:
        f.write(
            "import { describe, it, expect } from 'vitest';\n"
            "import { " + class_name + " } from '../src/index';\n\n"
            "describe('" + class_name + "', () => {\n"
            "  it('should process correctly', () => {\n"
            "    const obj = new " + class_name + "();\n"
            "    expect(obj.process('test')).toBe('" + title + " processed test');\n"
            "  });\n"
            "});\n"
        )

    # Demos
    demo_dir = f"apps/demo-{pkg_name}"
    os.makedirs(f"{demo_dir}/src", exist_ok=True)
    with open(f"{demo_dir}/package.json", "w") as f:
        f.write(f"""{{
  "name": "demo-{pkg_name}",
  "version": "1.0.0",
  "private": true,
  "scripts": {{
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview"
  }},
  "devDependencies": {{
    "vite": "^5.0.0",
    "typescript": "^5.0.0"
  }}
}}""")
    with open(f"{demo_dir}/tsconfig.json", "w") as f:
        f.write("""{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "module": "ESNext",
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "strict": true
  },
  "include": ["src"]
}""")
    with open(f"{demo_dir}/index.html", "w") as f:
        f.write(f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>{title} Demo</title>
    <style>
      body {{ font-family: system-ui, sans-serif; background: #f4f4f5; color: #18181b; padding: 2rem; }}
      .container {{ background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
      button {{ background: #18181b; color: white; border: none; padding: 0.5rem 1rem; border-radius: 4px; cursor: pointer; }}
      button:hover {{ background: #27272a; }}
      #output {{ margin-top: 1rem; padding: 1rem; background: #e5e7eb; border-radius: 4px; font-family: monospace; white-space: pre-wrap; }}
    </style>
  </head>
  <body>
    <h1>{title} Web UI</h1>
    <div class="container">
      <button id="btn-run">Run {title}</button>
      <div id="output">Ready.</div>
    </div>
    <script type="module" src="/src/main.ts"></script>
  </body>
</html>""")
    with open(f"{demo_dir}/src/main.ts", "w") as f:
        f.write(
            "document.getElementById('btn-run')?.addEventListener('click', () => {\n"
            "  const output = document.getElementById('output');\n"
            "  if (output) {\n"
            "    output.textContent = 'Running...\\n';\n"
            "    setTimeout(() => {\n"
            "      output.textContent += '[OK] " + title + " execution complete.';\n"
            "    }, 500);\n"
            "  }\n"
            "});\n"
        )

    # Docs
    doc_name = pkg_name.upper().replace("-", "_") + ".md"
    with open(f"docs/{doc_name}", "w") as f:
        f.write(f"""# {title}

Integration guide for {title}.

## CLI Usage

```bash
onnx9000 {pkg_name} model.onnx
```

## Demo

See the standalone web demo:
```bash
cd apps/demo-{pkg_name}
npm run dev
```
""")
