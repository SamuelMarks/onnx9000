import os
import subprocess
import sys

# Add Python packages to path just in case uv didn't link them natively in build context
sys.path.insert(0, os.path.abspath("../packages/python/onnx9000-core/src"))
sys.path.insert(0, os.path.abspath("../packages/python/onnx9000-converters/src"))
sys.path.insert(0, os.path.abspath("../packages/python/onnx9000-optimizer/src"))
sys.path.insert(0, os.path.abspath("../packages/python/onnx9000-backend-native/src"))
sys.path.insert(0, os.path.abspath("../packages/python/onnx9000-toolkit/src"))
sys.path.insert(0, os.path.abspath("../apps/cli/src"))
sys.path.insert(0, os.path.abspath("."))

project = "ONNX9000"
copyright = "2026, Samuel"
author = "Samuel"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
]

# myst-parser settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

# sphinx-js settings
js_language = "typescript"
# Run typedoc before sphinx-build, or let sphinx-js run it.
# sphinx-js will look for tsconfig.json in the js_source_path.
root_for_relative_js_paths = ".."
js_source_path = [
    "../packages/js/core/src",
    "../packages/js/transformers/src",
    "../packages/js/compiler/src",
    "../apps/netron-ui/src",
    "../apps/optimum-ui/src",
]

# To make sure typedoc can find all the workspace config:
# We might need to generate the typedoc JSON explicitly and point sphinx-js to it if it's too complex.
# But let's try auto-running first. We can override if needed.

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}


# Auto build typedoc on start
def run_typedoc(app):
    print("Running typedoc...")
    # Generate typedoc in docs/_build/typedoc
    docs_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(docs_dir)
    subprocess.run(
        [
            "pnpm",
            "exec",
            "typedoc",
            "--json",
            os.path.join(docs_dir, "_build", "typedoc", "typedoc.json"),
            "--options",
            os.path.join(root_dir, "typedoc.json"),
        ],
        cwd=root_dir,
        check=False,
    )

    # Run the Box.md fix script
    fix_script = os.path.join(docs_dir, "fix_box.py")
    subprocess.run([sys.executable, fix_script])

    # Generate the js-api toctree dynamically
    toc_script = os.path.join(docs_dir, "generate_toc.py")
    subprocess.run([sys.executable, toc_script])

    # Preprocess README.md to fix relative paths pointing to js-api/_media
    preprocess_script = os.path.join(docs_dir, "preprocess_readme.py")
    subprocess.run([sys.executable, preprocess_script])


from pygments.lexer import RegexLexer
from pygments.token import *


class MlirLexer(RegexLexer):
    name = "MLIR"
    aliases = ["mlir"]
    filenames = ["*.mlir"]

    tokens = {
        "root": [
            (r"//.*?\n", Comment.Single),
            (r"%[a-zA-Z0-9_]+", Name.Variable),
            (r"@[a-zA-Z0-9_\.]+", Name.Function),
            (r"\^[a-zA-Z0-9_]+", Name.Label),
            (r"#[a-zA-Z0-9_\.]+", Name.Namespace),
            (r"![a-zA-Z0-9_\.]+", Name.Class),
            (r'"[^"]*"', String),
            (r"\b[a-zA-Z_][a-zA-Z0-9_\.]*\b", Keyword),
            (r"-?[0-9]+\.?[0-9]*", Number),
            (r"[=\(\)<>\[\]\{\}:,\.\*\-\+]", Punctuation),
            (r"\s+", Whitespace),
        ]
    }


def setup(app):
    app.connect("builder-inited", run_typedoc)
    app.add_lexer("mlir", MlirLexer)


jsdoc_config_path = "../typedoc.json"
myst_heading_anchors = 6
