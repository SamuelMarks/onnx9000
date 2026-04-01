"""Numerical debugger for onnx9000."""

from typing import Dict, Any, List
import numpy as np
from onnx9000.core.ir import Graph, Tensor
from onnx9000.core.execution import ExecutionContext


class NumericalDebugger:
    """Utility to compare activations between source and target models/EPs."""

    def __init__(self, graph: Graph):
        self.graph = graph

    def compare(self, inputs: Dict[str, np.ndarray], ep1: Any, ep2: Any) -> Dict[str, float]:
        """Compare activations of all nodes between two execution providers."""
        # ep1 and ep2 should be ExecutionProviders

        ctx = ExecutionContext()
        # Convert numpy inputs to Tensor objects
        input_tensors = {k: Tensor(name=k, data=v, shape=v.shape) for k, v in inputs.items()}

        res1 = ep1.execute(self.graph, ctx, input_tensors)
        res2 = ep2.execute(self.graph, ctx, input_tensors)

        errors = {}
        for name in res1:
            if name in res2:
                t1 = res1[name].data
                t2 = res2[name].data
                if isinstance(t1, np.ndarray) and isinstance(t2, np.ndarray):
                    # Mean Absolute Error
                    mae = np.mean(np.abs(t1 - t2))
                    errors[name] = float(mae)

        return errors
