"""Sphinx extension for integrating the ONNX9000 Interactive WASM Demo UI.
This extension provides the `.. interactive-demo::` directive which mounts
the Vanilla JS frontend into the generated Sphinx HTML.
"""

import os
import shutil
import subprocess

from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.application import Sphinx


class InteractiveDemoNode(nodes.General, nodes.Element):
    """A custom docutils node for the interactive demo."""

    pass


def visit_interactive_demo_node(self, node: InteractiveDemoNode) -> None:
    """HTML rendering for the interactive demo node."""
    data_attrs = " ".join(
        [f'data-{k}="{v}"' for k, v in node.attributes.items() if k != "ids" and k != "classes"]
    )

    # CSS trick removed to let it fit its div normally
    self.body.append(
        f"<style>"
        f".onnx9000-demo-container-wrapper {{ width: 100%; box-sizing: border-box; }}"
        f".onnx9000-demo-container {{ width: 100%; height: 85vh; min-height: 700px; border: 1px solid var(--border-color, #ccc); border-radius: 8px; margin: 2rem 0; box-shadow: 0 10px 30px rgba(0,0,0,0.1); overflow: hidden; }}"
        f"</style>"
        f'<div class="onnx9000-demo-container-wrapper">'
        f'<div id="interactive-demo-container" class="onnx9000-demo-container" {data_attrs}>'
        f"</div></div>"
    )


def depart_interactive_demo_node(self, node: InteractiveDemoNode) -> None:
    """HTML departure function for the interactive demo node."""
    pass


class InteractiveDemoDirective(Directive):
    """Directive to embed the interactive WASM UI demo.

    Usage:
        .. interactive-demo::
           :initial-source: keras
           :initial-target: onnx
    """

    has_content = False
    option_spec = {
        "initial-source": str,
        "initial-target": str,
    }

    def run(self) -> list[nodes.Node]:
        """Creates the node with the specified options."""
        node = InteractiveDemoNode()
        node["initial-source"] = self.options.get("initial-source", "keras")
        node["initial-target"] = self.options.get("initial-target", "onnx")
        return [node]


def build_frontend(app: Sphinx) -> None:
    """Hooks into Sphinx build process to run the frontend build via pnpm
    if it hasn't been built, and copies the artifacts into the Sphinx static directory.
    """
    frontend_dir = os.path.abspath(os.path.join(app.srcdir, "../apps/sphinx-demo-ui"))
    repo_root = os.path.abspath(os.path.join(app.srcdir, ".."))
    dist_dir = os.path.join(frontend_dir, "dist")

    # We only run build if 'dist' doesn't exist to speed up docs iteration.
    # In CI, it will build fresh.
    if not os.path.exists(dist_dir) or os.environ.get("FORCE_FRONTEND_BUILD") == "1":
        print("[onnx9000-demo] Building Vanilla JS frontend with Vite...")
        subprocess.check_call(
            ["pnpm", "turbo", "run", "build", "--filter", "@onnx9000/sphinx-demo-ui"],
            cwd=repo_root,
        )
        print("[onnx9000-demo] Frontend build complete.")

    # Normally Sphinx expects these in _static, but we can also just inject the raw paths if we copy them.
    # The safest way is to hook them via app.add_css_file and app.add_js_file
    for f in os.listdir(dist_dir):
        if f.endswith(".css"):
            app.add_css_file(f"demo-ui/{f}")
        elif f.endswith(".js") and not f.endswith("umd.cjs"):
            app.add_js_file(f"demo-ui/{f}", type="module")


def copy_frontend_assets(app: Sphinx, exc: Exception) -> None:
    """Copies the built Vite assets into Sphinx's output build directory _static/demo-ui."""
    if exc:
        return

    if app.builder.format != "html":
        return

    frontend_dir = os.path.abspath(os.path.join(app.srcdir, "../apps/sphinx-demo-ui"))
    dist_dir = os.path.join(frontend_dir, "dist")

    out_static_dir = os.path.join(app.outdir, "_static", "demo-ui")
    if os.path.exists(out_static_dir):
        shutil.rmtree(out_static_dir)

    shutil.copytree(dist_dir, out_static_dir)

    # Copy pyodide
    pyodide_src = os.path.abspath(os.path.join(app.srcdir, "../node_modules/pyodide"))
    if os.path.exists(pyodide_src):
        for item in os.listdir(pyodide_src):
            src_item = os.path.join(pyodide_src, item)
            dst_item = os.path.join(out_static_dir, item)
            if os.path.isdir(src_item):
                shutil.copytree(src_item, dst_item, dirs_exist_ok=True)
            else:
                shutil.copy2(src_item, dst_item)


def setup(app: Sphinx) -> dict[str, str]:
    """Extension setup."""
    app.add_node(
        InteractiveDemoNode, html=(visit_interactive_demo_node, depart_interactive_demo_node)
    )

    app.add_directive("interactive-demo", InteractiveDemoDirective)

    app.connect("builder-inited", build_frontend)
    app.connect("build-finished", copy_frontend_assets)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
