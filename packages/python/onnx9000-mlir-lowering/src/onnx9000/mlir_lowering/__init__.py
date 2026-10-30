"""ONNX Graph to MLIR dialect lowering passes (`tosa`, `linalg`) for ONNX9000."""

from onnx9000.core.ir import Graph, Tensor, ValueInfo
from onnx9000.mlir import Dialect, MLIRModule, MLIROp, MLIRType, MLIRValue

_TOSA_OP_MAP: dict[str, str] = {
    "Add": "add",
    "Sub": "sub",
    "Mul": "mul",
    "Relu": "clamp",
    "MatMul": "matmul",
    "Conv": "conv2d",
    "Sigmoid": "sigmoid",
    "Tanh": "tanh",
    "Reshape": "reshape",
    "Transpose": "transpose",
}


def _value_to_mlir(item: Tensor | ValueInfo | str, default_type: MLIRType) -> MLIRValue:
    """Convert an ONNX Graph input/output/tensor reference into an `MLIRValue`.

    Args:
        item: Tensor, ValueInfo, or string name.
        default_type: Fallback `MLIRType` when shape is unknown.

    Returns:
        Typed `MLIRValue`.
    """
    if isinstance(item, (Tensor, ValueInfo)):
        dims = tuple(d if isinstance(d, int) else -1 for d in item.shape) if item.shape else None
        return MLIRValue(name=item.name, mlir_type=MLIRType("f32", dims))
    return MLIRValue(name=str(item), mlir_type=default_type)


class MLIRLoweringPass:
    """Lowers an ONNX9000 `Graph` into a target `Dialect` `MLIRModule`."""

    def __init__(self, target_dialect: Dialect = Dialect.TOSA) -> None:
        """Initialize the lowering pass with a target MLIR dialect.

        Args:
            target_dialect: Target MLIR dialect (`Dialect.TOSA` or `Dialect.LINALG`).
        """
        self.target_dialect = target_dialect

    def lower(self, graph: Graph) -> MLIRModule:
        """Lower an ONNX9000 `Graph` into an `MLIRModule`.

        Args:
            graph: Source ONNX9000 `Graph`.

        Returns:
            Constructed `MLIRModule` containing lowered dialect operations.
        """
        module = MLIRModule(name=graph.name or "onnx_graph")
        default_ty = MLIRType("f32")
        for node in graph.nodes:
            operands = [_value_to_mlir(inp, default_ty) for inp in node.inputs]
            results = [_value_to_mlir(out, default_ty) for out in node.outputs]
            mnemonic = _TOSA_OP_MAP.get(node.op_type, node.op_type.lower())
            attrs: dict[str, str | int | float] = {}
            if node.op_type == "Relu" and self.target_dialect == Dialect.TOSA:
                attrs["min_fp"] = 0.0
            module.add_op(
                MLIROp(
                    dialect=self.target_dialect,
                    name=mnemonic,
                    operands=operands,
                    results=results,
                    attributes=attrs,
                )
            )
        return module


def lower_to_tosa(graph: Graph) -> MLIRModule:
    """Lower an ONNX9000 `Graph` to the MLIR `tosa` dialect.

    Args:
        graph: Source ONNX9000 `Graph`.

    Returns:
        Lowered `MLIRModule`.
    """
    return MLIRLoweringPass(Dialect.TOSA).lower(graph)


def lower_to_linalg(graph: Graph) -> MLIRModule:
    """Lower an ONNX9000 `Graph` to the MLIR `linalg` dialect.

    Args:
        graph: Source ONNX9000 `Graph`.

    Returns:
        Lowered `MLIRModule`.
    """
    return MLIRLoweringPass(Dialect.LINALG).lower(graph)


def run() -> str:
    """Execute the default MLIR lowering module status check.

    Returns:
        MLIR lowering status confirmation string.
    """
    return "[mlir-lowering] processed"


__all__ = ["MLIRLoweringPass", "lower_to_linalg", "lower_to_tosa", "run"]
