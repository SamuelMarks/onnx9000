"""Advanced static memory arena planning and simulation."""

from typing import Optional, Union

from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Constant, Graph, Node, Tensor, Variable
from onnx9000.core.profiler import dtype_size, resolve_volume


class MemoryBlock:
    """Represents the Memory Block class."""

    def __init__(self, offset: int, size: int):
        """Initialize the instance."""
        self.offset = offset
        self.size = size
        self.free = True
        self.tensor_name: Optional[str] = None

    def __repr__(self):
        """Execute repr magic method operation."""
        return f"Block(offset={self.offset}, size={self.size}, free={self.free}, tensor={self.tensor_name})"


class ArenaSimulator:
    """Represents the Arena Simulator class."""

    def __init__(self, alignment: int = 256):
        """Initialize the instance."""
        self.alignment = alignment
        self.blocks: list[MemoryBlock] = []
        self.peak_memory = 0
        self.tensor_offsets: dict[str, int] = {}

    def _align(self, size: int) -> int:
        """Execute the align operation."""
        if size % self.alignment == 0:
            return size
        return size + (self.alignment - (size % self.alignment))

    def allocate_first_fit(self, name: str, size: int) -> int:
        """Execute the allocate first fit operation."""
        aligned_size = self._align(size)

        for b in self.blocks:
            if b.free and b.size >= aligned_size:
                b.free = False
                b.tensor_name = name
                if b.size > aligned_size:
                    new_block = MemoryBlock(b.offset + aligned_size, b.size - aligned_size)
                    b.size = aligned_size
                    self.blocks.append(new_block)
                    self.blocks.sort(key=lambda x: x.offset)
                self.tensor_offsets[name] = b.offset
                return b.offset

        last_offset = self.blocks[-1].offset + self.blocks[-1].size if self.blocks else 0
        new_block = MemoryBlock(last_offset, aligned_size)
        new_block.free = False
        new_block.tensor_name = name
        self.blocks.append(new_block)
        self.tensor_offsets[name] = new_block.offset

        self.peak_memory = max(self.peak_memory, last_offset + aligned_size)
        return new_block.offset

    def allocate_best_fit(self, name: str, size: int) -> int:
        """Execute the allocate best fit operation."""
        aligned_size = self._align(size)

        best_block = None
        for b in self.blocks:
            if b.free and b.size >= aligned_size:
                if best_block is None or b.size < best_block.size:
                    best_block = b

        if best_block:
            best_block.free = False
            best_block.tensor_name = name
            if best_block.size > aligned_size:
                new_block = MemoryBlock(
                    best_block.offset + aligned_size, best_block.size - aligned_size
                )
                best_block.size = aligned_size
                self.blocks.append(new_block)
                self.blocks.sort(key=lambda x: x.offset)
            self.tensor_offsets[name] = best_block.offset
            return best_block.offset

        last_offset = self.blocks[-1].offset + self.blocks[-1].size if self.blocks else 0
        new_block = MemoryBlock(last_offset, aligned_size)
        new_block.free = False
        new_block.tensor_name = name
        self.blocks.append(new_block)
        self.tensor_offsets[name] = new_block.offset

        self.peak_memory = max(self.peak_memory, last_offset + aligned_size)
        return new_block.offset

    def free(self, name: str):
        """Execute the free operation."""
        for b in self.blocks:
            if b.tensor_name == name:
                b.free = True
                b.tensor_name = None
                break

        i = 0
        while i < len(self.blocks) - 1:
            if self.blocks[i].free and self.blocks[i + 1].free:
                self.blocks[i].size += self.blocks[i + 1].size
                self.blocks.pop(i + 1)
            else:
                i += 1

    def calculate_fragmentation(self) -> float:
        """Execute the calculate fragmentation operation."""
        total_free = sum(b.size for b in self.blocks if b.free)
        if self.peak_memory == 0:
            return 0.0
        return (total_free / self.peak_memory) * 100.0


def simulate_memory_plan(
    graph: Graph, strategy: str = "first_fit", dynamic_overrides: dict[str, int] = None
) -> ArenaSimulator:
    """Simulate memory allocation for a graph based on topological node execution.

    Computes exact lifetimes and buffer offsets.
    """
    arena = ArenaSimulator(alignment=256)

    lifetimes = {}

    for i, n in enumerate(graph.nodes):
        for out in n.outputs:
            if out not in lifetimes:
                lifetimes[out] = [i, i]
        for inp in n.inputs:
            if inp in lifetimes:
                lifetimes[inp][1] = max(lifetimes[inp][1], i)

    for inp in graph.inputs:
        inp_name = inp if isinstance(inp, str) else inp.name
        if inp_name not in lifetimes:
            lifetimes[inp_name] = [0, len(graph.nodes)]
        else:
            lifetimes[inp_name][1] = len(graph.nodes)

    active_tensors = set()

    for inp in graph.inputs:
        inp_name = inp if isinstance(inp, str) else inp.name
        if inp_name in graph.tensors and not (
            isinstance(graph.tensors[inp_name], Constant) or graph.tensors[inp_name].is_initializer
        ):
            v = resolve_volume(graph.tensors[inp_name].shape, dynamic_overrides)
            if isinstance(v, int) and v > 0:
                size = v * dtype_size(graph.tensors[inp_name].dtype)
                if strategy == "first_fit":
                    arena.allocate_first_fit(inp_name, size)
                else:
                    arena.allocate_best_fit(inp_name, size)
                active_tensors.add(inp_name)

    for i, n in enumerate(graph.nodes):
        in_place_possible = False
        if n.op_type in ["Relu", "Sigmoid", "Tanh"]:
            if n.inputs and lifetimes.get(n.inputs[0], [0, 0])[1] <= i:
                in_place_possible = True
                arena.tensor_offsets[n.outputs[0]] = arena.tensor_offsets.get(n.inputs[0], 0)
                for b in arena.blocks:
                    if b.tensor_name == n.inputs[0]:
                        b.tensor_name = n.outputs[0]
                        active_tensors.add(n.outputs[0])
                        active_tensors.discard(n.inputs[0])
                        break

        view_possible = False
        if not in_place_possible and n.op_type in [
            "Reshape",
            "Flatten",
            "Squeeze",
            "Unsqueeze",
        ]:
            if n.inputs and n.inputs[0] in arena.tensor_offsets:
                view_possible = True
                arena.tensor_offsets[n.outputs[0]] = arena.tensor_offsets[n.inputs[0]]
                if n.outputs[0] in lifetimes and n.inputs[0] in lifetimes:
                    lifetimes[n.inputs[0]][1] = max(
                        lifetimes[n.outputs[0]][1], lifetimes[n.inputs[0]][1]
                    )

        if not in_place_possible and not view_possible:
            for out in n.outputs:
                if out in graph.tensors:
                    v = resolve_volume(graph.tensors[out].shape, dynamic_overrides)
                    if isinstance(v, int) and v > 0:
                        size = v * dtype_size(graph.tensors[out].dtype)
                        if strategy == "first_fit":
                            arena.allocate_first_fit(out, size)
                        else:
                            arena.allocate_best_fit(out, size)
                        active_tensors.add(out)

        for active in list(active_tensors):
            if lifetimes.get(active, [0, 0])[1] <= i:
                arena.free(active)
                active_tensors.discard(active)

    return arena


Graph.simulate_memory_plan = simulate_memory_plan
