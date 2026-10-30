"""TVM Tensor Expression (TE) loop schedule primitives."""

from __future__ import annotations

from onnx9000.core.exceptions import CompilationError

from .tensor import AxisOpProtocol, IterVar, NamedOpProtocol, Tensor, TensorOpType


class Stage:
    """Represents a single loop nest stage in a TE schedule."""

    def __init__(self, op: TensorOpType) -> None:
        """Initialize a schedule stage for an operation."""
        self.op = op
        self.axes: list[IterVar] = list(op.axis) if isinstance(op, AxisOpProtocol) else []
        self.relations: list[dict[str, str | int | IterVar | list[IterVar] | Stage | None]] = []
        self.inlined = False
        self.storage_alignments: list[tuple[IterVar, int, int]] = []
        self.double_buffer = False

    def split(
        self, parent: IterVar, factor: int | None = None, nparts: int | None = None
    ) -> tuple[IterVar, IterVar]:
        """Split a loop axis into outer and inner axes."""
        if parent not in self.axes:
            raise CompilationError(f"Axis {parent} not found in stage.")
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
        """Fuse multiple loop axes into a single axis."""
        for a in args:
            if a not in self.axes:
                raise CompilationError(f"Axis {a} not found in stage.")

        fused = IterVar(f"{args[0].name}.fused")
        idx = self.axes.index(args[0])
        self.axes = [ax for ax in self.axes if ax not in args]
        self.axes.insert(idx, fused)

        self.relations.append({"type": "fuse", "args": list(args), "fused": fused})
        return fused

    def reorder(self, *args: IterVar) -> None:
        """Reorder the loop axes of the stage."""
        if set(args) != set(self.axes):
            raise CompilationError("reorder must specify all axes")
        self.axes = list(args)
        self.relations.append({"type": "reorder", "args": list(args)})

    def bind(self, axis: IterVar, thread_env: str) -> None:
        """Bind a loop axis to a hardware thread index."""
        if axis not in self.axes:
            raise CompilationError(f"Axis {axis} not found in stage.")
        self.relations.append({"type": "bind", "axis": axis, "thread": thread_env})

    def compute_at(self, parent_stage: Stage, scope: IterVar) -> None:
        """Attach this stage's computation at a loop scope of a parent stage."""
        self.relations.append({"type": "compute_at", "parent": parent_stage, "scope": scope})

    def compute_inline(self) -> None:
        """Mark this stage to be computed inline at its consumers."""
        self.inlined = True
        self.relations.append({"type": "compute_inline"})

    def tile(
        self, x_parent: IterVar, y_parent: IterVar, x_factor: int, y_factor: int
    ) -> tuple[IterVar, IterVar, IterVar, IterVar]:
        """Tile two loop axes by the specified factors."""
        xo, xi = self.split(x_parent, factor=x_factor)
        yo, yi = self.split(y_parent, factor=y_factor)
        self.relations.append({"type": "tile", "args": [xo, yo, xi, yi]})
        return xo, yo, xi, yi

    def unroll(self, axis: IterVar) -> None:
        """Unroll the specified loop axis."""
        self.relations.append({"type": "unroll", "axis": axis})

    def vectorize(self, axis: IterVar) -> None:
        """Vectorize the specified loop axis."""
        self.relations.append({"type": "vectorize", "axis": axis})

    def tensorize(self, axis: IterVar, intrin: str) -> None:
        """Lower the loop axis using a hardware tensor intrinsic."""
        self.relations.append({"type": "tensorize", "axis": axis, "intrin": intrin})

    def set_double_buffer(self) -> None:
        """Enable double buffering for this stage."""
        self.double_buffer = True

    def storage_align(self, axis: IterVar, factor: int, offset: int) -> None:
        """Set storage alignment requirements for an axis."""
        self.storage_alignments.append((axis, factor, offset))


class Schedule:
    """Collection of stages representing a complete loop schedule."""

    def __init__(self, stages: list[Stage]) -> None:
        """Initialize the schedule with a list of stages."""
        self.stages = stages
        self.stage_map = {s.op: s for s in stages}

    def __getitem__(self, tensor: Tensor) -> Stage:
        """Retrieve the Stage corresponding to a Tensor."""
        if tensor.op not in self.stage_map:
            raise CompilationError(f"Tensor {tensor} not in schedule.")
        return self.stage_map[tensor.op]

    def cache_read(self, tensor: Tensor, scope: str, readers: list[Tensor]) -> Tensor:
        """Create a read cache stage for the given tensor in the specified memory scope."""
        op_name = tensor.op.name if isinstance(tensor.op, NamedOpProtocol) else str(tensor.op)
        cached_tensor = Tensor(tensor.shape, tensor.dtype, f"cache_read_{op_name}")
        stage = Stage(cached_tensor.op)
        self.stages.append(stage)
        self.stage_map[cached_tensor.op] = stage
        return cached_tensor

    def cache_write(self, tensor: Tensor, scope: str) -> Tensor:
        """Create a write cache stage for the given tensor in the specified memory scope."""
        op_name = tensor.op.name if isinstance(tensor.op, NamedOpProtocol) else str(tensor.op)
        cached_tensor = Tensor(tensor.shape, tensor.dtype, f"cache_write_{op_name}")
        stage = Stage(cached_tensor.op)
        self.stages.append(stage)
        self.stage_map[cached_tensor.op] = stage
        return cached_tensor


def create_schedule(ops: Tensor | list[Tensor]) -> Schedule:
    """Create a Schedule from one or more output Tensors."""
    if not isinstance(ops, list):
        ops = [ops]
    stages = [Stage(op.op) for op in ops]
    return Schedule(stages)
