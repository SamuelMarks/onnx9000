"""Typed MLIR AST/IR dialect definitions (`tosa`, `linalg`, `tensor`, `func`) for ONNX9000."""

from dataclasses import dataclass, field
from enum import Enum


class Dialect(str, Enum):
    """Supported MLIR dialects in ONNX9000."""

    FUNC = "func"
    TOSA = "tosa"
    LINALG = "linalg"
    TENSOR = "tensor"


@dataclass(frozen=True)
class MLIRType:
    """Represents a shaped tensor or scalar type in MLIR.

    Attributes:
        element_type: Element type identifier (e.g., `"f32"`, `"i64"`).
        shape: Optional tuple of static dimensions (`-1` for dynamic).
    """

    element_type: str
    shape: tuple[int, ...] | None = None

    def format(self) -> str:
        """Render the MLIR type in textual assembly syntax.

        Returns:
            Formatted MLIR type string (e.g., `"tensor<1x4xf32>"` or `"f32"`).
        """
        if self.shape is None:
            return self.element_type
        dims = "x".join("?" if d < 0 else str(d) for d in self.shape)
        prefix = f"{dims}x" if dims else ""
        return f"tensor<{prefix}{self.element_type}>"


@dataclass(frozen=True)
class MLIRValue:
    """SSA value in an MLIR block or region.

    Attributes:
        name: SSA identifier (without leading `%`).
        mlir_type: Associated `MLIRType`.
    """

    name: str
    mlir_type: MLIRType


@dataclass
class MLIROp:
    """Single MLIR operation node within a dialect.

    Attributes:
        dialect: Target `Dialect`.
        name: Operation mnemonic within the dialect.
        operands: Input SSA values.
        results: Output SSA values.
        attributes: Dictionary of named string/int/float attributes.
    """

    dialect: Dialect
    name: str
    operands: list[MLIRValue] = field(default_factory=list)
    results: list[MLIRValue] = field(default_factory=list)
    attributes: dict[str, str | int | float] = field(default_factory=dict)

    def format(self) -> str:
        """Render the operation in textual MLIR syntax.

        Returns:
            Formatted MLIR operation line.
        """
        lhs = ""
        if self.results:
            res_names = ", ".join(f"%{r.name}" for r in self.results)
            lhs = f"{res_names} = "
        ops_str = ", ".join(f"%{o.name}" for o in self.operands)
        attr_str = ""
        if self.attributes:
            pairs = ", ".join(f"{k} = {v}" for k, v in sorted(self.attributes.items()))
            attr_str = f" {{{pairs}}}"
        ret_types = ", ".join(r.mlir_type.format() for r in self.results)
        sig = (
            f" : ({', '.join(o.mlir_type.format() for o in self.operands)}) -> ({ret_types})"
            if self.results
            else ""
        )
        return f'{lhs}"{self.dialect.value}.{self.name}"({ops_str}){attr_str}{sig}'


@dataclass
class MLIRModule:
    """Top-level MLIR module container holding dialect operations."""

    name: str = "main"
    ops: list[MLIROp] = field(default_factory=list)

    def add_op(self, op: MLIROp) -> None:
        """Append an operation to the module body.

        Args:
            op: The `MLIROp` to append.
        """
        self.ops.append(op)

    def emit(self) -> str:
        """Emit textual MLIR assembly for the entire module.

        Returns:
            Complete `module @name { ... }` textual MLIR representation.
        """
        lines = [f"module @{self.name} {{"]
        for op in self.ops:
            lines.append(f"  {op.format()}")
        lines.append("}")
        return chr(10).join(lines)


def run() -> str:
    """Execute the default MLIR module status check.

    Returns:
        MLIR status confirmation string.
    """
    return "[mlir] processed"


__all__ = ["Dialect", "MLIRModule", "MLIROp", "MLIRType", "MLIRValue", "run"]
