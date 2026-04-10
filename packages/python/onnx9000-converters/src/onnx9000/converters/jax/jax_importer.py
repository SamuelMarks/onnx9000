"""JAX jaxpr to onnx9000 IR importer."""

from typing import Any, Callable, Optional

from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Variable


class JAXImporter:
    """Importer for JAX computation graphs (jaxpr)."""

    def __init__(self, graph_name: str = "jax_graph"):
        """Initialize the JAXImporter."""
        self.builder = Graph(graph_name)
        self.var_map = {}
        self._var_counter = 0

    def get_var_name(self, var: Any) -> str:
        """Get or create a unique name for a JAX variable.

        Args:
            var: The JAX variable or literal to get a name for.

        Returns:
            A unique string name for the variable.

        """
        # Try to use var itself as key, but fall back to id(var) for unhashable types
        try:
            key = var
            hash(key)
        except TypeError:
            key = id(var)

        if key in self.var_map:
            return self.var_map[key]

        name = f"v{self._var_counter}"
        self._var_counter += 1
        self.var_map[key] = name

        # Add to graph tensors if it has type info
        if hasattr(var, "aval"):
            shape = list(var.aval.shape)
            dtype = self._map_dtype(var.aval.dtype)
            self.builder.add_tensor(Variable(name, shape, dtype))

        return name

    def import_func(self, func: Callable, *args: Any, **kwargs: Any) -> Graph:
        """Trace a JAX function and import it into onnx9000 IR."""
        import jax

        def flat_func(*args, **kwargs):
            """Execute internal flattened function for JAX tracing."""
            return func(*args, **kwargs)

        jaxpr = jax.make_jaxpr(flat_func)(*args, **kwargs)
        return self.import_jaxpr(jaxpr.jaxpr, jaxpr.consts)

    def import_jaxpr(self, jaxpr: Any, consts: list[Any]) -> Graph:
        """Import a JAX jaxpr structure into onnx9000 IR."""
        # Add inputs
        for var in jaxpr.invars:
            name = self.get_var_name(var)
            self.builder.inputs.append(name)

        # Add constants
        for var, val in zip(jaxpr.constvars, consts):
            name = self.get_var_name(var)
            import numpy as np
            from onnx9000.core.ir import Constant

            c = Constant(name, values=np.array(val).tobytes(), shape=list(np.array(val).shape))
            self.builder.add_tensor(c)

        # Process equations
        for eqn in jaxpr.eqns:
            in_names = [self.get_var_name(v) for v in eqn.invars]
            out_names = [self.get_var_name(v) for v in eqn.outvars]
            params = dict(eqn.params.items())

            import onnx9000.converters.jax.jax_ops  # noqa: F401
            from onnx9000.core.registry import global_registry

            try:
                op_func = global_registry.get_op("jax", eqn.primitive.name)
                node = op_func(inputs=in_names, outputs=out_names, params=params)
            except Exception:
                # Fallback to uppercase
                op_type = eqn.primitive.name.capitalize()
                node = Node(
                    op_type=op_type,
                    inputs=in_names,
                    outputs=out_names,
                    attributes=params,
                    name=f"{op_type}_{out_names[0]}" if out_names else op_type,
                )

            self.builder.add_node(node)

        # Add outputs
        for var in jaxpr.outvars:
            name = self.get_var_name(var)
            self.builder.outputs.append(name)

        return self.builder

    def _map_dtype(self, jax_dtype: Any) -> DType:
        """Map JAX dtype to onnx9000 DType."""
        import numpy as np

        if jax_dtype == np.float32:
            return DType.FLOAT32
        if jax_dtype == np.int32:
            return DType.INT32
        return DType.FLOAT32
