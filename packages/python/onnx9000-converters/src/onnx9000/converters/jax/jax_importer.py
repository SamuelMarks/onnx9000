"""Advanced JAX importer."""

from typing import Any, Callable, Optional, Tuple
import jax
import jax.numpy as jnp
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.core.dtypes import DType


class JAXImporter:
    """Importer for JAX functions and jaxprs."""

    def __init__(self, func: Optional[Callable] = None) -> None:
        """Initialize the JAX importer."""
        self.func = func
        self.graph = Graph(name="jax_model")
        self.var_map: dict[Any, str] = {}
        self._var_counter = 0

    def get_var_name(self, var: Any) -> str:
        """Get or create a unique name for a JAX variable or constant."""
        if var in self.var_map:
            return self.var_map[var]

        if hasattr(var, "id"):
            name = f"var_{var.id}"
        elif isinstance(var, (int, float, bool)):
            name = f"const_{self._var_counter}"
            self._var_counter += 1
            # Add as initializer
            t = Tensor(name=name, dtype=DType.FLOAT32.value, shape=())
            self.graph.initializers.append(name)
            self.graph.tensors[name] = t
        else:
            name = f"var_{self._var_counter}"
            self._var_counter += 1

        self.var_map[var] = name
        return name

    def import_func(self, *args: Any, **kwargs: Any) -> Graph:
        """Import a JAX function by tracing it with given arguments."""
        if self.func is None:
            raise ValueError("Function must be provided to import_func.")

        # Phase 15: Pytree Resolution
        flat_args, in_tree = jax.tree_util.tree_flatten((args, kwargs))

        def flat_func(*f_args):
            f_args, f_kwargs = jax.tree_util.tree_unflatten(in_tree, f_args)
            return self.func(*f_args, **f_kwargs)

        jaxpr = jax.make_jaxpr(flat_func)(*flat_args)
        return self.import_jaxpr(jaxpr.jaxpr)

    def import_jaxpr(self, jaxpr: jax.core.Jaxpr) -> Graph:
        """Import a JAX jaxpr object."""
        # Handle inputs
        for var in jaxpr.invars:
            name = self.get_var_name(var)
            shape = var.aval.shape
            dtype = self._map_dtype(var.aval.dtype)
            t = Tensor(name=name, dtype=dtype.value, shape=shape)
            self.graph.inputs.append(t)
            self.graph.tensors[name] = t

        # Handle constants
        for var in jaxpr.constvars:
            name = self.get_var_name(var)
            shape = var.aval.shape
            dtype = self._map_dtype(var.aval.dtype)
            t = Tensor(name=name, dtype=dtype.value, shape=shape)
            self.graph.initializers.append(name)
            self.graph.tensors[name] = t

        # Handle equations
        for eqn in jaxpr.eqns:
            primitive = eqn.primitive.name
            inputs = [self.get_var_name(v) for v in eqn.invars]
            outputs = [self.get_var_name(v) for v in eqn.outvars]

            # Map primitive to ONNX op
            op_type = self._map_primitive(primitive)

            node = Node(
                op_type=op_type,
                inputs=inputs,
                outputs=outputs,
                attributes=eqn.params,
                name=f"{op_type}_{outputs[0]}" if outputs else op_type,
            )
            self.graph.nodes.append(node)

            # Add output tensors
            for var in eqn.outvars:
                name = self.get_var_name(var)
                shape = var.aval.shape
                dtype = self._map_dtype(var.aval.dtype)
                t = Tensor(name=name, dtype=dtype.value, shape=shape)
                self.graph.tensors[name] = t

        # Handle outputs
        for var in jaxpr.outvars:
            name = self.get_var_name(var)
            if name in self.graph.tensors:
                self.graph.outputs.append(self.graph.tensors[name])

        return self.graph

    def _map_dtype(self, jax_dtype: Any) -> DType:
        """Map JAX dtype to DType."""
        if jax_dtype == jnp.float32:
            return DType.FLOAT32
        if jax_dtype == jnp.int32:
            return DType.INT32
        return DType.FLOAT32

    def _map_primitive(self, primitive: str) -> str:
        """Map JAX primitive name to ONNX op type."""
        mapping = {
            "add": "Add",
            "sub": "Sub",
            "mul": "Mul",
            "div": "Div",
            "dot_general": "MatMul",
            "exp": "Exp",
            "log": "Log",
            "sin": "Sin",
            "cos": "Cos",
            "tanh": "Tanh",
            "relu": "Relu",
            "reshape": "Reshape",
            "transpose": "Transpose",
        }
        return mapping.get(primitive, primitive)
