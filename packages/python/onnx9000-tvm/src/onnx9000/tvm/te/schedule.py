"""TVM submodule for AST and optimization."""

from typing import Any, Optional

from .tensor import ComputeOp, IterVar, Tensor


class Stage:
    """Core class for TVM AST node or pass."""

    def __init__(self, op: Any):
        """Magic method."""
        """Initialize."""
        """Do the function."""
        self.op = op
        self.axes: list[IterVar] = list(op.axis) if hasattr(op, "axis") else []
        self.relations: list[dict[str, Any]] = []
        self.inlined = False
        self.storage_alignments: list[Any] = []
        self.double_buffer = False

    def split(
        self, parent: IterVar, factor: Optional[int] = None, nparts: Optional[int] = None
    ) -> tuple[IterVar, IterVar]:
        """Evaluates or manipulates TVM AST nodes."""
        if parent not in self.axes:
            raise ValueError(f"Axis {parent} not found in stage.")
        idx = self.axes.index(parent)
        outer = IterVar(f"{parent.name}.outer")
        inner = IterVar(f"{parent.name}.inner")

        self.axes[idx : idx + 1] = [outer, inner]
        self.relations.append(
            {
                "type": "split",
                "parent": parent,
                "outer": outer,
                "inner": inner,
                "factor": factor,
                "nparts": nparts,
            }
        )
        return outer, inner

    def fuse(self, *args: IterVar) -> IterVar:
        """Do the function."""
        for a in args:
            if a not in self.axes:
                raise ValueError(f"Axis {a} not found in stage.")

        fused = IterVar(f"{args[0].name}.fused")
        # Find first index
        idx = self.axes.index(args[0])
        # Remove all fused args and insert the new one
        self.axes = [ax for ax in self.axes if ax not in args]
        self.axes.insert(idx, fused)

        self.relations.append({"type": "fuse", "args": list(args), "fused": fused})
        return fused

    def reorder(self, *args: IterVar):
        """Do the function."""
        if set(args) != set(self.axes):
            raise ValueError("reorder must specify all axes")
        self.axes = list(args)
        self.relations.append({"type": "reorder", "args": list(args)})

    def bind(self, axis: IterVar, thread_env: str):
        """Do the function."""
        if axis not in self.axes:
            raise ValueError(f"Axis {axis} not found in stage.")
        self.relations.append({"type": "bind", "axis": axis, "thread": thread_env})

    def compute_at(self, parent_stage: "Stage", scope: IterVar):
        """Do the function."""
        self.relations.append({"type": "compute_at", "parent": parent_stage, "scope": scope})

    def compute_inline(self):
        """Do the function."""
        self.inlined = True
        self.relations.append({"type": "compute_inline"})

    def tile(
        self, x_parent: IterVar, y_parent: IterVar, x_factor: int, y_factor: int
    ) -> tuple[IterVar, IterVar, IterVar, IterVar]:
        """Evaluates or manipulates TVM AST nodes."""
        xo, xi = self.split(x_parent, factor=x_factor)
        yo, yi = self.split(y_parent, factor=y_factor)

        # Naive reorder just inserts them where they were or puts them at the end.
        # To avoid ValueError from reorder checking all axes, let's just set the relations here.
        # This is a dummy implementation anyway.
        self.relations.append({"type": "tile", "args": [xo, yo, xi, yi]})
        return xo, yo, xi, yi

    def unroll(self, axis: IterVar):
        """Do the function."""
        self.relations.append({"type": "unroll", "axis": axis})

    def vectorize(self, axis: IterVar):
        """Do the function."""
        self.relations.append({"type": "vectorize", "axis": axis})

    def tensorize(self, axis: IterVar, intrin: Any):
        """Do the function."""
        self.relations.append({"type": "tensorize", "axis": axis, "intrin": intrin})

    def set_double_buffer(self):
        """Do the function."""
        self.double_buffer = True

    def storage_align(self, axis: IterVar, factor: int, offset: int):
        """Do the function."""
        self.storage_alignments.append((axis, factor, offset))


class Schedule:
    """Core class for TVM AST node or pass."""

    def __init__(self, stages: list[Stage]):
        """Magic method."""
        """Initialize."""
        """Do the function."""
        self.stages = stages
        self.stage_map = {s.op: s for s in stages}

    def __getitem__(self, tensor: Tensor) -> Stage:
        """Do the function."""
        if tensor.op not in self.stage_map:
            raise ValueError(f"Tensor {tensor} not in schedule.")
        return self.stage_map[tensor.op]

    def cache_read(self, tensor: Tensor, scope: str, readers: list[Tensor]) -> Tensor:
        """Do the function."""
        # Simplistic mocking
        cached_tensor = Tensor(
            tensor.shape, tensor.dtype, f"cache_read_{getattr(tensor.op, 'name', str(tensor.op))}"
        )
        stage = Stage(cached_tensor.op)
        self.stages.append(stage)
        self.stage_map[cached_tensor.op] = stage
        return cached_tensor

    def cache_write(self, tensor: Tensor, scope: str) -> Tensor:
        """Do the function."""
        cached_tensor = Tensor(
            tensor.shape, tensor.dtype, f"cache_write_{getattr(tensor.op, 'name', str(tensor.op))}"
        )
        stage = Stage(cached_tensor.op)
        self.stages.append(stage)
        self.stage_map[cached_tensor.op] = stage
        return cached_tensor


from typing import Union


def create_schedule(ops: Union[Any, list[Any]]) -> Schedule:
    """Do the function."""
    if not isinstance(ops, list):
        ops = [ops]
    # In reality, this does a post-order traversal to collect all ops
    stages = [Stage(op.op) for op in ops]
    return Schedule(stages)
