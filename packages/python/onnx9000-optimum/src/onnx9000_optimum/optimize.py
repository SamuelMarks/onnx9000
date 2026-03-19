import json
import logging
import os
import sys
from typing import Optional

logger = logging.getLogger(__name__)


def optimize_model(
    model_path: str, level: str = "O1", disable_fusion: bool = False, optimize_size: bool = False
):
    """Optimize ONNX model using onnx9000 core optimizer passes."""
    try:
        from onnx9000.core.parser.core import load as load_onnx
        from onnx9000.core.serializer import save as save_onnx
    except ImportError:
        logger.error("onnx9000 core is missing.")
        sys.exit(1)

    try:
        from onnx9000.optimizer.simplifier.api import simplify
    except ImportError:
        logger.error("onnx9000-optimizer is missing.")
        sys.exit(1)

    print(f"Loading {model_path}...")
    graph = load_onnx(model_path)

    orig_graph_copy = graph.clone() if level in ["O3", "O4"] else graph

    # Configure optimizer rules based on level

    if level in ["O3", "O4"]:
        from onnx9000.optimizer.simplifier.passes.webgpu import (
            fp16_cast,
            fuse_geglu,
            fuse_swiglu,
            generate_execution_schedule,
            generate_html_report,
            inject_web_worker_boundaries,
            optimize_for_webgpu,
            replace_gather_with_lookup,
        )

        fuse_swiglu(graph)
        fuse_geglu(graph)
        optimize_for_webgpu(graph)
        replace_gather_with_lookup(graph)
        inject_web_worker_boundaries(graph)

    if level == "O4":
        fp16_cast(graph)

    print(f"Optimizing graph with level {level}...")

    graph = simplify(
        graph,
        skip_fusions=disable_fusion,
        skip_constant_folding=False,
        strip_metadata=optimize_size,
        max_iterations=5,
    )

    out_path = model_path.replace(".onnx", f"_optimized_{level}.onnx")
    print(f"Saving optimized graph to {out_path}...")
    save_onnx(graph, out_path)

    if level in ["O3", "O4"]:
        generate_html_report(orig_graph_copy, graph, out_path.replace(".onnx", "_report.html"))
        generate_execution_schedule(graph, out_path.replace(".onnx", "_schedule.json"))

    # Collect some statistics
    try:
        orig_size = os.path.getsize(model_path)
        new_size = os.path.getsize(out_path)
        print(
            f"Optimization complete. Size reduction: {orig_size / 1024 / 1024:.2f}MB -> {new_size / 1024 / 1024:.2f}MB"
        )
    except Exception:
        return False
