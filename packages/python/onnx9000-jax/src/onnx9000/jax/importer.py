"""Implementation of importer (onnx9000 / jax / importer.py) providing JaxprImporter, load_jax, load."""

from onnx9000.core.dtypes import DType
from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.core.registry import global_registry


def _map_jax_type(jax_type: str) -> DType:
    """Execute map jax type operation.

    Args:
        jax_type: Parameter specifying jax type (str).

    Returns:
        Result of map jax type (DType).
    """
    if jax_type == "f32":
        return DType.FLOAT32
    if jax_type == "i32":
        return DType.INT32
    return DType.FLOAT32


class JaxprImporter:
    """Pars a JAX jaxpr (dict representation) to ONNX IR."""

    def __init__(self) -> None:
        """Initialize JaxprImporter with configured parameters and state."""
        self.graph = Graph(name="jax_model")

    def parse(
        self, jaxpr: dict[str, int | float | str | bool | bytes | list | dict | tuple | None]
    ) -> Graph:
        """Execute parse on JaxprImporter.

        Args:
            jaxpr: Parameter specifying jaxpr (dict[str, Union[int, float, str, bool, bytes, list, dict, tuple, None]]).

        Returns:
            Result of parse (Graph).
        """
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
                op_func = global_registry.get_op("jax", primitive)
                n = op_func(inputs=inputs, outputs=outputs, params=params)
            except (
                Onnx9000Error,
                OSError,
                ValueError,
                TypeError,
                RuntimeError,
                KeyError,
                IndexError,
                AttributeError,
                ImportError,
                LookupError,
                ArithmeticError,
                BufferError,
                AssertionError,
                StopIteration,
            ):
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


def load_jax(
    jaxpr_dict: dict[str, int | float | str | bool | bytes | list | dict | tuple | None],
) -> Graph:
    """Unified load interface for JAX."""
    importer = JaxprImporter()
    return importer.parse(jaxpr_dict)


def load(
    model_path_or_dict: int | float | str | bool | bytes | list | dict | tuple | None,
    format: str = "auto",
) -> Graph:
    """Load a model using a unified interface."""
    if format == "tf" or (isinstance(model_path_or_dict, dict) and "node" in model_path_or_dict):
        return None
    elif format in ("jax", "flax") or (
        isinstance(model_path_or_dict, dict) and "eqns" in model_path_or_dict
    ):
        return load_jax(model_path_or_dict)
    else:
        from onnx9000.core.parser.core import load as onnx_load

        return onnx_load(model_path_or_dict)
