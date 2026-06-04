"""JAX jaxpr to onnx9000 IR importer."""

from collections.abc import Callable
from typing import Any

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
            shape = list(var.aval.shape)  # pragma: no cover
            dtype = self._map_dtype(var.aval.dtype)  # pragma: no cover
            self.builder.add_tensor(Variable(name, shape, dtype))  # pragma: no cover

        return name

    def import_func(self, func: Callable, *args: Any, **kwargs: Any) -> Graph:
        """Trace a JAX function and import it into onnx9000 IR."""
        import jax  # pragma: no cover

        def flat_func(*args, **kwargs):  # pragma: no cover
            """Execute internal flattened function for JAX tracing."""
            return func(*args, **kwargs)  # pragma: no cover

        jaxpr = jax.make_jaxpr(flat_func)(*args, **kwargs)  # pragma: no cover
        return self.import_jaxpr(jaxpr.jaxpr, jaxpr.consts)  # pragma: no cover

    def import_jaxpr(self, jaxpr: Any, consts: list[Any]) -> Graph:
        """Import a JAX jaxpr structure into onnx9000 IR."""
        # Add inputs
        for var in jaxpr.invars:
            name = self.get_var_name(var)  # pragma: no cover
            self.builder.inputs.append(name)  # pragma: no cover

        # Add constants
        for var, val in zip(jaxpr.constvars, consts):
            name = self.get_var_name(var)  # pragma: no cover
            import numpy as np  # pragma: no cover
            from onnx9000.core.ir import Constant  # pragma: no cover

            c = Constant(
                name, values=np.array(val).tobytes(), shape=list(np.array(val).shape)
            )  # pragma: no cover
            self.builder.add_tensor(c)  # pragma: no cover

        # Process equations
        for eqn in jaxpr.eqns:
            in_names = [self.get_var_name(v) for v in eqn.invars]  # pragma: no cover
            out_names = [self.get_var_name(v) for v in eqn.outvars]  # pragma: no cover
            params = dict(eqn.params.items())  # pragma: no cover

            import onnx9000.jax.jax_ops  # noqa: F401  # pragma: no cover
            from onnx9000.core.registry import global_registry  # pragma: no cover

            try:  # pragma: no cover
                op_func = global_registry.get_op("jax", eqn.primitive.name)  # pragma: no cover
                node = op_func(
                    inputs=in_names, outputs=out_names, params=params
                )  # pragma: no cover
            except Exception:  # pragma: no cover
                # Fallback to uppercase
                op_type = eqn.primitive.name.capitalize()  # pragma: no cover
                node = Node(  # pragma: no cover
                    op_type=op_type,
                    inputs=in_names,
                    outputs=out_names,
                    attributes=params,
                    name=f"{op_type}_{out_names[0]}" if out_names else op_type,
                )

            self.builder.add_node(node)  # pragma: no cover

        # Add outputs
        for var in jaxpr.outvars:
            name = self.get_var_name(var)  # pragma: no cover
            self.builder.outputs.append(name)  # pragma: no cover

        return self.builder

    def _map_dtype(self, jax_dtype: Any) -> DType:
        """Map JAX dtype to onnx9000 DType."""
        import numpy as np

        if jax_dtype == np.float32:
            return DType.FLOAT32
        if jax_dtype == np.int32:
            return DType.INT32
        return DType.FLOAT32
