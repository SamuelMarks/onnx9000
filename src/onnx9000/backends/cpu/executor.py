"""CPU executor implementation."""

import numpy as np
import concurrent.futures
from typing import Dict, Any, List, Optional
import logging

from onnx9000.core.ir import Graph, Node
from onnx9000.backends.cpu.memory import MemoryPlanner
from onnx9000.backends.cpu.ops import OP_REGISTRY

logger = logging.getLogger(__name__)


class Executor:
    """Zero-Overhead Python Dispatcher for CPU execution."""

    def __init__(self, graph: Graph, use_threadpool: bool = False) -> None:
        """Initialize the executor."""
        self.graph = graph
        self.planner = MemoryPlanner()
        self.constants: Dict[str, np.ndarray] = {}
        self.use_threadpool = use_threadpool
        self._init_memory()

    def _init_memory(self) -> None:
        """Plan memory for the graph."""
        # 1. Cache initializers
        for init_name in self.graph.initializers:
            tensor = self.graph.tensors.get(init_name)
            if tensor and tensor.data is not None:
                self.constants[init_name] = tensor.data

        # 2. Allocate static memory for intermediates
        for node in self.graph.nodes:
            for out_name in node.outputs:
                tensor = self.graph.tensors.get(out_name)
                if tensor:
                    # Very simple static planning if dimensions are known statically
                    shape = []
                    is_dynamic = False
                    for dim in tensor.shape:
                        if hasattr(dim, "value") and isinstance(dim.value, str):
                            is_dynamic = True
                            break
                        shape.append(int(getattr(dim, "value", dim)))

                    if not is_dynamic:
                        # Assuming float32 as default if dtype mapping is complex
                        dtype = np.float32
                        if tensor.dtype:
                            # Try to infer numpy dtype from core dtype
                            dtype = np.dtype(
                                "float32"
                            )  # Placeholder, could map exactly
                        size_in_bytes = np.prod(shape, dtype=int) * dtype.itemsize
                        self.planner.allocate_static(
                            out_name, size_in_bytes, tuple(shape), dtype
                        )

        self.planner.build_arena()

    def _execute_node(self, node: Node, run_context: Dict[str, np.ndarray]) -> None:
        """Execute a single node."""
        inputs = []
        for inp in node.inputs:
            if inp in run_context:
                inputs.append(run_context[inp])
            elif inp in self.constants:
                inputs.append(self.constants[inp])
            else:
                inputs.append(self.planner.get_tensor(inp))

        op_fn = OP_REGISTRY.get(node.op_type)
        if op_fn is None:
            raise RuntimeError(
                f"Operation {node.op_type} not implemented in CPU backend"
            )

        outputs = op_fn(inputs, node.attributes)

        for idx, out_name in enumerate(node.outputs):
            if idx < len(outputs):
                out_data = outputs[idx]
                if out_name in self.planner.offsets:
                    self.planner.set_tensor(out_name, out_data)
                else:
                    run_context[out_name] = out_data

    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Run the graph execution loop."""
        run_context: Dict[str, np.ndarray] = inputs.copy()

        if self.use_threadpool:
            # Step 013: Implement thread-pool orchestration via Python concurrent.futures
            # Determine independent nodes. For simplicity, we just use ThreadPoolExecutor
            # and submit tasks if their dependencies are met.
            # Building a dependency graph
            dep_count = {i: 0 for i in range(len(self.graph.nodes))}
            node_to_idx = {node: i for i, node in enumerate(self.graph.nodes)}
            produced_by = {}
            for i, node in enumerate(self.graph.nodes):
                for out in node.outputs:
                    produced_by[out] = i

            for i, node in enumerate(self.graph.nodes):
                for inp in node.inputs:
                    if inp in produced_by:
                        dep_count[i] += 1

            ready_nodes = [i for i, count in dep_count.items() if count == 0]
            completed_nodes = set()

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {}
                while len(completed_nodes) < len(self.graph.nodes):
                    # Submit ready nodes
                    for idx in ready_nodes:
                        if idx not in futures and idx not in completed_nodes:
                            node = self.graph.nodes[idx]
                            futures[idx] = executor.submit(
                                self._execute_node, node, run_context
                            )

                    # Wait for at least one to complete
                    done, _ = concurrent.futures.wait(
                        futures.values(), return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    for future in done:
                        # Find which node this future belongs to
                        done_idx = None
                        for idx, f in futures.items():
                            if f == future:
                                done_idx = idx
                                break

                        if done_idx is not None:
                            del futures[done_idx]
                            completed_nodes.add(done_idx)
                            # Update dependencies
                            node = self.graph.nodes[done_idx]
                            for i, other_node in enumerate(self.graph.nodes):
                                if i not in completed_nodes and i not in futures:
                                    for out in node.outputs:
                                        if out in other_node.inputs:
                                            dep_count[i] -= 1
                                            if dep_count[i] == 0:
                                                if i not in ready_nodes:
                                                    ready_nodes.append(i)

        else:
            # Step 004 & 014: Fast linear execution loop
            for node in self.graph.nodes:
                self._execute_node(node, run_context)

        # Collect outputs
        results = {}
        for out_name in self.graph.outputs:
            if out_name in run_context:
                results[out_name] = run_context[out_name]
            elif out_name in self.constants:
                results[out_name] = self.constants[out_name]
            else:
                results[out_name] = self.planner.get_tensor(out_name)

        return results
