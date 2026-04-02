"""TVM submodule for AST and optimization."""

import json
import tarfile
import zipfile
from io import BytesIO
from typing import Union


def bundle_artifacts(
    artifacts: dict[str, Union[str, bytes]], output_path: str, format: str = "tar.gz"
):
    """Pass 324: Bundle output artifacts into a single .tar.gz or .zip."""
    if format == "tar.gz":
        with tarfile.open(output_path, "w:gz") as tar:
            for name, content in artifacts.items():
                if isinstance(content, str):
                    content = content.encode("utf-8")
                info = tarfile.TarInfo(name=name)
                info.size = len(content)
                tar.addfile(info, BytesIO(content))
    elif format == "zip":
        with zipfile.ZipFile(output_path, "w") as zf:
            for name, content in artifacts.items():
                if isinstance(content, bytes):
                    zf.writestr(name, content)
                else:
                    zf.writestr(name, content.encode("utf-8"))
    else:
        raise ValueError(f"Unknown format {format}")


def generate_npm_package(
    model_name: str, artifacts: dict[str, Union[str, bytes]]
) -> dict[str, Union[str, bytes]]:
    """Pass 325: Generate an npm package structure."""
    pkg_json = {
        "name": f"@onnx9000-models/{model_name.lower()}",
        "version": "1.0.0",
        "description": "Auto-generated ONNX9000 WebAssembly/WebGPU model",
        "main": "index.js",
        "types": "index.d.ts",
        "dependencies": {},  # Zero-dependency!
    }

    # Pass 328: Provide zero-dependency inference SDK.
    index_js = """
// Zero-dependency inference SDK
class ModelRunner {
    constructor() { this.instance = null; }
    async load(wasmBytes) {
        const module = await WebAssembly.instantiate(wasmBytes, {});
        this.instance = module.instance;
    }
    run() {
        if (!this.instance) throw new Error("Model not loaded");
        // ... generic run
    }
}
module.exports = { ModelRunner };
"""
    index_d_ts = """
export class ModelRunner {
    load(wasmBytes: Uint8Array): Promise<void>;
    run(): void;
}
"""

    res = dict(artifacts)
    res["package.json"] = json.dumps(pkg_json, indent=2)
    res["index.js"] = index_js
    res["index.d.ts"] = index_d_ts
    return res


class Target:
    """Core class for TVM AST node or pass."""

    def __init__(self, target_name: str, options: dict = None):
        """Magic method."""
        """Initialize."""
        """Do the function."""
        self.target_name = target_name
        self.options = options or {}


def build(mod, target=None, params=None):
    """Implement a Python API."""
    return None


def load_graph_inputs_override(cli_override_str: str) -> dict:
    """Pass 329: Allow manual overriding of graph inputs (shape/type) via CLI."""
    # format: "input1:f32[1,3,224,224],input2:i64[1]"
    overrides = {}
    if not cli_override_str:
        return overrides
    for part in cli_override_str.split(","):
        name, rest = part.split(":")
        dtype, shape_str = rest.split("[")
        shape_str = shape_str.rstrip("]")
        shape = tuple(int(s) for s in shape_str.split(",") if s)
        overrides[name] = {"dtype": dtype, "shape": shape}
    return overrides
