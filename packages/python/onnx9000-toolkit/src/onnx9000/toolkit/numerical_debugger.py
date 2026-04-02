"""Numerical debugger for onnx9000."""

from typing import Any

import numpy as np
from onnx9000.core.execution import ExecutionContext, SessionOptions
from onnx9000.core.ir import Graph, Tensor


class NumericalDebugger:
    """Utility to compare activations between source and target models/EPs."""

    def __init__(self, graph: Graph):
        """Initialize the NumericalDebugger with a graph.

        Args:
            graph: The ONNX graph to debug.
        """
        self.graph = graph

    def compare(self, inputs: dict[str, np.ndarray], ep1: Any, ep2: Any) -> dict[str, float]:
        """Compare activations of all nodes between two execution providers.

        Args:
            inputs: Dictionary mapping input names to numpy arrays.
            ep1: The first execution provider (e.g., CPU).
            ep2: The second execution provider (e.g., CUDA).

        Returns:
            Dictionary mapping tensor names to their Mean Absolute Error (MAE).
        """
        # ep1 and ep2 should be ExecutionProviders

        options = SessionOptions()
        ctx = ExecutionContext(options=options)
        # Convert numpy inputs to Tensor objects
        input_tensors = {k: Tensor(name=k, data=v, shape=list(v.shape)) for k, v in inputs.items()}

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
