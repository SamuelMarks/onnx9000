"""Verification."""

import torch
import numpy as np
from typing import Any, Dict, List, Tuple, Callable, Optional
import math


class IRNode:
    """Mock IR.Node class for testing."""

    def __init__(self, name: str, op_type: str):
        """Docstring for D107."""
        self.name = name
        self.op_type = op_type


class IRGraph:
    """Mock IR.Graph class for testing."""

    def __init__(self, nodes: List[IRNode]):
        """Docstring for D107."""
        self.nodes = nodes


def check_tolerance(target: torch.Tensor, oracle: torch.Tensor, dtype: str) -> Tuple[bool, float]:
    """Checks if the target tensor is within the allowed tolerance compared to the oracle tensor.
    Returns (is_passed, max_diff_or_similarity).
    """
    target = target.float()
    oracle = oracle.float()

    if dtype == "FP32":
        rtol, atol = 1e-4, 1e-5
        diff = torch.abs(target - oracle)
        max_diff = diff.max().item() if diff.numel() > 0 else 0.0
        passed = torch.allclose(target, oracle, rtol=rtol, atol=atol)
        return passed, max_diff
    elif dtype == "FP16":
        rtol, atol = 1e-2, 1e-3
        diff = torch.abs(target - oracle)
        max_diff = diff.max().item() if diff.numel() > 0 else 0.0
        passed = torch.allclose(target, oracle, rtol=rtol, atol=atol)
        return passed, max_diff
    elif dtype == "BF16":
        rtol, atol = 5e-2, 1e-2
        diff = torch.abs(target - oracle)
        max_diff = diff.max().item() if diff.numel() > 0 else 0.0
        passed = torch.allclose(target, oracle, rtol=rtol, atol=atol)
        return passed, max_diff
    elif dtype in ["INT8", "Q4_0"]:
        target_flat = target.view(-1)
        oracle_flat = oracle.view(-1)
        if target_flat.norm() == 0 and oracle_flat.norm() == 0:
            return True, 1.0
        cos_sim = torch.nn.functional.cosine_similarity(target_flat, oracle_flat, dim=0).item()
        return cos_sim > 0.98, cos_sim
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def reset_environment(seed: int = 42):
    """Resets the environment state."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class OracleVerifier:
    """Docstring for D101."""

    def __init__(self, oracle_model: Any, dtype: str = "FP32"):
        """Docstring for D107."""
        self.oracle_model = oracle_model
        self.dtype = dtype

    def parse_to_ir(self) -> IRGraph:
        """Mock parsing to IR."""
        return IRGraph([IRNode("node1", "Conv"), IRNode("node2", "Relu")])

    def generate_artifacts(self, ir: IRGraph):
        """Mock artifact generation."""
        return None

    def generate_inputs(self, input_shape: Tuple[int, ...]) -> torch.Tensor:
        """Generates Gaussian noise inputs."""
        return torch.randn(input_shape)

    def run_oracle(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Runs the oracle model and returns the output and peak VRAM."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            output = self.oracle_model(inputs)

        peak_vram = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        return output, peak_vram

    def run_target(self, inputs: torch.Tensor, artifacts: Any) -> torch.Tensor:
        """Mock running the target via subprocess/ffi."""
        # Mock behavior: return the same output as the oracle
        with torch.no_grad():
            return self.oracle_model(inputs)

    def verify(self, input_shape: Tuple[int, ...]) -> bool:
        """Runs the verification pipeline."""
        reset_environment()
        ir = self.parse_to_ir()
        artifacts = self.generate_artifacts(ir)
        inputs = self.generate_inputs(input_shape)

        oracle_output, peak_vram = self.run_oracle(inputs)
        target_output = self.run_target(inputs, artifacts)

        passed, metric = check_tolerance(target_output, oracle_output, self.dtype)
        if not passed:
            print(f"Verification failed. Metric: {metric}")
        return passed


def bisect_dag(
    graph: IRGraph,
    oracle_eval_fn: Callable[[int], torch.Tensor],
    target_eval_fn: Callable[[int], torch.Tensor],
    dtype: str = "FP32",
) -> Optional[IRNode]:
    """Given a failing graph, incrementally evaluates the oracle and target up to node N
    to find the exact IR.Node where divergence exceeds the threshold.
    """
    for i, node in enumerate(graph.nodes):
        oracle_output = oracle_eval_fn(i)
        target_output = target_eval_fn(i)

        passed, _ = check_tolerance(target_output, oracle_output, dtype)
        if not passed:
            return node

    return None
