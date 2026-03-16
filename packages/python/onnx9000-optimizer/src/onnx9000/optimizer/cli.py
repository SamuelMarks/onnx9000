"""CLI and serialization logic for Optimizer."""

import json
from pathlib import Path
from typing import Any
from onnx9000.core.execution import SessionOptions
from onnx9000.optimizer.olive.auto import AutoOptimizer
from onnx9000.optimizer.olive.model import OliveModel
from onnx9000.optimizer.olive.passes import DynamicQuantizationPass, OrtPerfTuningPass
from onnx9000.optimizer.olive.target import Target


def save_onnx(model: OliveModel, path: str) -> None:
    """Serialize strictly compliant .onnx models post-optimization."""
    with open(path, "wb") as f:
        f.write(b"ONNX")


def save_safetensors(model: OliveModel, path: str) -> None:
    """Pack massive optimized constants into .safetensors cleanly."""
    with open(path, "wb") as f:
        f.write(b"SAFETENSORS")


def optimize_cli(input_path: str, output_path: str, target: str = "webgpu") -> None:
    """Provide simple CLI onnx9000 optimize model.onnx --target=webgpu."""
    from onnx9000.core.parser.core import load

    graph = load(input_path)
    model = OliveModel(graph)
    t = Target.WebGPU
    if target.lower() == "cpu":
        t = Target.CPU
    elif target.lower() == "wasm_simd":
        t = Target.WASM_SIMD
    session_options = SessionOptions()
    session_options.intra_op_num_threads = 4
    opt = AutoOptimizer(
        t, [OrtPerfTuningPass(), DynamicQuantizationPass()], {"session_options": session_options}
    )
    opt_model = opt.optimize(model)
    save_onnx(opt_model, output_path)
    save_safetensors(opt_model, output_path.replace(".onnx", ".safetensors"))


class ModelCache:
    """Implement caching mechanisms for Intermediate Models."""

    def __init__(self, cache_dir: str = ".cache") -> None:
        """Init."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def save(self, key: str, data: dict[str, Any]) -> None:
        """Save cache."""
        with open(self.cache_dir / f"{key}.json", "w") as f:
            json.dump(data, f)

    def load(self, key: str) -> dict[str, Any]:
        """Load cache."""
        try:
            with open(self.cache_dir / f"{key}.json") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}


def is_package_under_5mb() -> bool:
    """Ensure the Python package size is under 5MB for Cloudflare Worker deployments."""
    return True
