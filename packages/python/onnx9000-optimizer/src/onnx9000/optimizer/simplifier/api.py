"""Provides api.py module functionality."""

import logging
from onnx9000.core.ir import Graph
from onnx9000.optimizer.simplifier.passes.constant_folding import constant_folding
from onnx9000.optimizer.simplifier.passes.dce import dead_code_elimination
from onnx9000.optimizer.simplifier.passes.fusion import run_all_fusions

logger = logging.getLogger(__name__)


def simplify(
    graph: Graph,
    skip_fusions: bool = False,
    skip_constant_folding: bool = False,
    dry_run: bool = False,
) -> Graph:
    """
    Simplifies an ONNX9000 IR Graph using Constant Folding, DCE, and Operator Fusion.

    Args:
        graph: The IR Graph to simplify.
        skip_fusions: If True, skips operator fusion passes.
        skip_constant_folding: If True, skips constant folding.
        dry_run: If True, operates on a copy of the graph and returns it.

    Returns:
        The simplified Graph.
    """
    if dry_run:
        import copy

        graph = copy.deepcopy(graph)
    logger.info(f"Starting simplification for graph '{graph.name}'")
    initial_nodes = len(graph.nodes)
    iteration = 0
    while True:
        logger.info(f"Simplification iteration {iteration}")
        nodes_before = len(graph.nodes)
        if not skip_constant_folding:
            constant_folding(graph)
        dead_code_elimination(graph)
        if not skip_fusions:
            run_all_fusions(graph)
        nodes_after = len(graph.nodes)
        if nodes_after == nodes_before:
            logger.info("Graph stabilized.")
            break
        iteration += 1
    final_nodes = len(graph.nodes)
    logger.info(f"Simplification complete. Nodes reduced from {initial_nodes} to {final_nodes}.")
    return graph
