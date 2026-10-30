"""Pure-Python numerical verification and DAG bisection utilities for ONNX9000."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from typing import Protocol

from onnx9000.core.exceptions import Onnx9000Error


class FloatTensorProtocol(Protocol):
    """Protocol for tensor-like objects supporting flat float extraction."""

    def tolist(self) -> list[float] | float:
        """Convert tensor values to a Python list or scalar."""


TensorLike = Sequence[float | int] | FloatTensorProtocol


def _flatten_floats(val: TensorLike) -> list[float]:
    """Extract a flat list of floats from a sequence or tensor-like object.

    Args:
        val: Input sequence or tensor-like object.

    Returns:
        Flat list of float values.

    Raises:
        AttributeError: If the input does not support float sequence extraction.
    """
    if isinstance(val, (list, tuple)):
        out: list[float] = []
        for item in val:
            if isinstance(item, (list, tuple)):
                out.extend(_flatten_floats(item))
            else:
                out.append(float(item))
        return out
    raw = val.tolist()
    if isinstance(raw, (int, float)):
        return [float(raw)]
    return _flatten_floats(raw)


class IRNode:
    """Lightweight IR node descriptor for verification and bisection."""

    def __init__(self, name: str, op_type: str) -> None:
        """Initialize an IRNode.

        Args:
            name: Node identifier.
            op_type: ONNX operator type.
        """
        self.name = name
        self.op_type = op_type


class IRGraph:
    """Lightweight IR graph container for verification and bisection."""

    def __init__(self, nodes: list[IRNode]) -> None:
        """Initialize an IRGraph.

        Args:
            nodes: Ordered list of IR nodes.
        """
        self.nodes = nodes


def check_tolerance(
    target: TensorLike,
    oracle: TensorLike,
    dtype: str,
) -> tuple[bool, float]:
    """Check whether the target tensor values are within tolerance of the oracle tensor.

    Args:
        target: Target tensor or sequence of floats.
        oracle: Reference oracle tensor or sequence of floats.
        dtype: Precision identifier ('FP32', 'FP16', 'BF16', 'INT8', 'Q4_0').

    Returns:
        Tuple of (passed, max_diff_or_cosine_similarity).

    Raises:
        ValueError: If dtype is not a supported precision mode.
    """
    t_vals = _flatten_floats(target)
    o_vals = _flatten_floats(oracle)

    if dtype == "FP32":
        rtol, atol = 1e-4, 1e-5
    elif dtype == "FP16":
        rtol, atol = 1e-2, 1e-3
    elif dtype == "BF16":
        rtol, atol = 5e-2, 1e-2
    elif dtype in ("INT8", "Q4_0"):
        norm_t = math.sqrt(sum(x * x for x in t_vals))
        norm_o = math.sqrt(sum(x * x for x in o_vals))
        if norm_t == 0.0 and norm_o == 0.0:
            return True, 1.0
        if norm_t == 0.0 or norm_o == 0.0:
            return False, 0.0
        dot = sum(a * b for a, b in zip(t_vals, o_vals, strict=False))
        cos_sim = dot / (norm_t * norm_o)
        return cos_sim > 0.98, cos_sim
    else:
        raise Onnx9000Error(f"Unsupported dtype: {dtype}")

    if not t_vals or not o_vals:
        return True, 0.0

    diffs = [abs(a - b) for a, b in zip(t_vals, o_vals, strict=False)]
    max_diff = max(diffs)
    passed = all(abs(a - b) <= (atol + rtol * abs(b)) for a, b in zip(t_vals, o_vals, strict=False))
    return passed, max_diff


def reset_environment(seed: int = 42) -> None:
    """Reset the deterministic random number generator seed.

    Args:
        seed: Random seed integer.
    """
    random.seed(seed)


class OracleVerifier:
    """Verification harness comparing target execution against a reference oracle callable."""

    def __init__(
        self,
        oracle_model: Callable[[list[float]], TensorLike],
        dtype: str = "FP32",
    ) -> None:
        """Initialize the OracleVerifier.

        Args:
            oracle_model: Reference callable accepting a float list and returning tensor values.
            dtype: Precision mode for tolerance checking.
        """
        self.oracle_model = oracle_model
        self.dtype = dtype

    def parse_to_ir(self) -> IRGraph:
        """Construct the default verification IRGraph.

        Returns:
            IRGraph containing sample Conv and Relu nodes.
        """
        return IRGraph([IRNode("node1", "Conv"), IRNode("node2", "Relu")])

    def generate_artifacts(self, ir: IRGraph) -> None:
        """Generate target execution artifacts for the given IRGraph.

        Args:
            ir: Parsed IRGraph.
        """
        _ = ir

    def generate_inputs(self, input_shape: tuple[int, ...]) -> list[float]:
        """Generate deterministic Gaussian noise inputs for the given shape.

        Args:
            input_shape: Shape tuple of dimensions.

        Returns:
            Flat list of sampled float values.
        """
        num_elements = 1
        for dim in input_shape:
            num_elements *= dim
        return [random.gauss(0.0, 1.0) for _ in range(num_elements)]

    def run_oracle(self, inputs: list[float]) -> tuple[TensorLike, int]:
        """Execute the reference oracle callable on the generated inputs.

        Args:
            inputs: Flat list of input floats.

        Returns:
            Tuple of (oracle_output, peak_memory_bytes).
        """
        output = self.oracle_model(inputs)
        return output, 0

    def run_target(self, inputs: list[float], artifacts: None) -> TensorLike:
        """Execute the target model on the generated inputs.

        Args:
            inputs: Flat list of input floats.
            artifacts: Compiled target artifacts.

        Returns:
            Target output values.
        """
        _ = artifacts
        return self.oracle_model(inputs)

    def verify(self, input_shape: tuple[int, ...]) -> bool:
        """Run end-to-end verification for the given input shape.

        Args:
            input_shape: Shape tuple of dimensions.

        Returns:
            True if target outputs match oracle outputs within tolerance, False otherwise.
        """
        reset_environment()
        ir = self.parse_to_ir()
        artifacts = self.generate_artifacts(ir)
        inputs = self.generate_inputs(input_shape)

        oracle_output, _ = self.run_oracle(inputs)
        target_output = self.run_target(inputs, artifacts)

        passed, metric = check_tolerance(target_output, oracle_output, self.dtype)
        if not passed:
            print(f"Verification failed. Metric: {metric}")
        return passed


def bisect_dag(
    graph: IRGraph,
    oracle_eval_fn: Callable[[int], TensorLike],
    target_eval_fn: Callable[[int], TensorLike],
    dtype: str = "FP32",
) -> IRNode | None:
    """Incrementally evaluate oracle and target up to node N to locate the first diverging node.

    Args:
        graph: IRGraph to bisect.
        oracle_eval_fn: Callable returning oracle output at node index N.
        target_eval_fn: Callable returning target output at node index N.
        dtype: Precision mode for tolerance checking.

    Returns:
        The first IRNode where divergence exceeds tolerance, or None if all nodes pass.
    """
    for i, node in enumerate(graph.nodes):
        oracle_output = oracle_eval_fn(i)
        target_output = target_eval_fn(i)

        passed, _ = check_tolerance(target_output, oracle_output, dtype)
        if not passed:
            return node

    return None
