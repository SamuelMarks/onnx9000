"""Module providing core logic and structural definitions."""

from onnx9000.core.exceptions import CompilationError
from onnx9000.core.ir import Graph


def plan_memory(graph: Graph) -> None:
    """
    Performs liveness analysis to determine tensor lifespans and
    assigns memory buffer IDs to allow C++ memory re-use.
    """
    # 1. Determine topological ordering (assume graph.nodes is already sorted by parser)

    # 2. Find first and last usage of each tensor
    # Tensors are created at the node that outputs them, or at idx 0 if input/initializer.
    # They die at the last node that uses them as input.

    lifespans: dict[str, list[int]] = {}

    for name in graph.inputs + graph.initializers:
        lifespans[name] = [0, 0]  # Start at 0, currently ends at 0

    for i, node in enumerate(graph.nodes):
        node_idx = i + 1

        for out in node.outputs:
            if out not in lifespans:
                lifespans[out] = [node_idx, node_idx]
            else:
                lifespans[out][0] = node_idx
                lifespans[out][1] = node_idx

        for inp in node.inputs:
            if inp not in lifespans:
                raise CompilationError(
                    f"Tensor {inp} used before creation in node {node.name}"
                )
            lifespans[inp][1] = max(lifespans[inp][1], node_idx)

    # Outputs must stay alive until the very end
    end_idx = len(graph.nodes) + 1
    for out in graph.outputs:
        if out in lifespans:
            lifespans[out][1] = end_idx

    for name, span in lifespans.items():
        if name in graph.tensors:
            graph.tensors[name].lifespan = (span[0], span[1])

    # 3. Greedy interval coloring for memory reuse
    intervals = []
    for _name, tensor in graph.tensors.items():
        if (
            not tensor.is_initializer
            and _name not in graph.inputs
            and tensor.lifespan[0] != -1
        ):
            intervals.append((tensor.lifespan[0], tensor.lifespan[1], tensor))

    intervals.sort(key=lambda x: x[0])

    active_buffers: list[tuple[int, int]] = []
    buffer_count = 0

    for start, end, tensor in intervals:
        reusable_id = -1
        for i, (active_end, buf_id) in enumerate(active_buffers):
            # Start must be > active_end to avoid in-place overlap hazards
            # In-place sharing (start == active_end) is only safe for element-wise ops,
            # but highly unsafe for MatMul/Conv which read inputs multiple times.
            if start > active_end:
                reusable_id = buf_id
                active_buffers[i] = (end, buf_id)
                break

        if reusable_id != -1:
            tensor.buffer_id = reusable_id
        else:
            tensor.buffer_id = buffer_count
            active_buffers.append((end, buffer_count))
            buffer_count += 1
