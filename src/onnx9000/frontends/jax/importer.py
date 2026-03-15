"""Module providing core logic and structural definitions."""

from typing import Any, Dict, List
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.core.dtypes import DType


def _map_jax_type(jax_type: str) -> DType:
    """Provides semantic functionality and verification."""
    if jax_type == "f32":
        return DType.FLOAT32
    if jax_type == "i32":
        return DType.INT32
    return DType.FLOAT32


class JaxprImporter:
    """Parses a JAX jaxpr (dict representation) to ONNX IR."""

    def __init__(self):
        """Provides semantic functionality and verification."""
        self.graph = Graph(name="jax_model")

    def parse(self, jaxpr: Dict[str, Any]) -> Graph:
        """Provides semantic functionality and verification."""
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
            op_type = primitive
            attributes = {}
            if primitive == "add":
                op_type = "Add"
            elif primitive == "mul":
                op_type = "Mul"
            elif primitive == "dot_general":
                op_type = "MatMul"
            elif primitive == "broadcast_in_dim":
                op_type = "Expand"
                attributes["broadcast_dimensions"] = params.get(
                    "broadcast_dimensions", []
                )
            elif primitive == "xla_pmap":
                op_type = "XlaPmap"
                attributes["axis_name"] = params.get("axis_name", "")
            elif primitive == "grad_core":
                op_type = "Grad"
            else:
                op_type = primitive
            n = Node(
                op_type=op_type,
                inputs=inputs,
                outputs=outputs,
                attributes=attributes,
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


def load_jax(jaxpr_dict: Dict[str, Any]) -> Graph:
    """Unified load interface for JAX."""
    importer = JaxprImporter()
    return importer.parse(jaxpr_dict)


def load(model_path_or_dict: Any, format: str = "auto") -> Graph:
    """Provide a unified `onnx9000.load('model.pb')` interface."""
    if format == "tf" or (
        isinstance(model_path_or_dict, dict) and "node" in model_path_or_dict
    ):
        from onnx9000.frontends.tf.importer import load_tf

        return None
    elif format == "jax" or (
        isinstance(model_path_or_dict, dict) and "eqns" in model_path_or_dict
    ):
        return load_jax(model_path_or_dict)
    else:
        from onnx9000.core.parser import load as onnx_load

        return onnx_load(model_path_or_dict)
