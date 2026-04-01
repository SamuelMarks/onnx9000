"""Module providing core logic and structural definitions."""

from typing import Any

import onnx9000.converters.jax.flax_ops  # noqa: F401
import onnx9000.converters.jax.jax_ops  # noqa: F401
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.core.registry import global_registry


def _map_jax_type(jax_type: str) -> DType:
    """Implement the _map_jax_type method or operation."""
    if jax_type == "f32":
        return DType.FLOAT32
    if jax_type == "i32":
        return DType.INT32
    return DType.FLOAT32


class JaxprImporter:
    """Pars a JAX jaxpr (dict representation) to ONNX IR."""

    def __init__(self) -> None:
        """Implement the __init__ method or operation."""
        self.graph = Graph(name="jax_model")

    def parse(self, jaxpr: dict[str, Any]) -> Graph:
        """Implement the parse method or operation."""
        for invar in jaxpr.get("invars", []):
            name = invar["name"]
            shape = tuple(invar["shape"])
            dt = _map_jax_type(invar["type"])
            t = Tensor(name=name, dtype=dt.value, shape=shape)
            self.graph.inputs.append(t)
            self.graph.tensors[name] = t
        for constvar in jaxpr.get("constvars", []):
            name = constvar["name"]
            shape = tuple(constvar["shape"])
            dt = _map_jax_type(constvar["type"])
            t = Tensor(name=name, dtype=dt.value, shape=shape)
            self.graph.initializers.append(name)
            self.graph.tensors[name] = t

        for eqn in jaxpr.get("eqns", []):
            primitive = eqn["primitive"]
            inputs = [i["name"] for i in eqn.get("invars", [])]
            outputs = [o["name"] for o in eqn.get("outvars", [])]
            params = eqn.get("params", {})
            try:
                op_func = global_registry.get_op(primitive, "jax")
                n = op_func(inputs=inputs, outputs=outputs, params=params)
            except Exception:
                n = Node(
                    op_type=primitive,
                    inputs=inputs,
                    outputs=outputs,
                    attributes={},
                    name=f"{primitive}_{outputs[0]}" if outputs else primitive,
                )
            self.graph.nodes.append(n)
            for outvar in eqn.get("outvars", []):
                name = outvar["name"]
                shape = tuple(outvar["shape"])
                dt = _map_jax_type(outvar["type"])
                t = Tensor(name=name, dtype=dt.value, shape=shape)
                self.graph.tensors[name] = t
        for outvar in jaxpr.get("outvars", []):
            name = outvar["name"]
            if name in self.graph.tensors:
                self.graph.outputs.append(self.graph.tensors[name])
        return self.graph


def load_jax(jaxpr_dict: dict[str, Any]) -> Graph:
    """Unified load interface for JAX."""
    importer = JaxprImporter()
    return importer.parse(jaxpr_dict)


def load(model_path_or_dict: Any, format: str = "auto") -> Graph:
    """Provide a unified `onnx9000.load('model.pb')` interface."""
    if format == "tf" or (isinstance(model_path_or_dict, dict) and "node" in model_path_or_dict):
        return None
    elif format in ("jax", "flax") or (
        isinstance(model_path_or_dict, dict) and "eqns" in model_path_or_dict
    ):
        return load_jax(model_path_or_dict)
    else:
        from onnx9000.core.parser.core import load as onnx_load

        return onnx_load(model_path_or_dict)
