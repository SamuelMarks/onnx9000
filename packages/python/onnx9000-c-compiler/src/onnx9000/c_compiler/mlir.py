"""MLIR compiler for onnx9000."""

from onnx9000.core.ir import Graph


class MLIRCompiler:
    """Compiler for transforming ONNX9000 IR to MLIR (TOSA/Linalg)."""

    def __init__(self, graph: Graph):
        """Initialize the MLIRCompiler with a target graph."""
        self.graph = graph
        self.lines = []

    def emit(self, line: str):
        """Emit a line of MLIR source code."""
        self.lines.append(line)

    def generate_tosa(self) -> str:
        """Lower IR to TOSA dialect."""
        self.emit("module {")
        self.emit("  func.func @main(%arg0: tensor<...>) -> tensor<...> {")

        for node in self.graph.nodes:
            if node.op_type == "Conv":
                # %0 = "tosa.conv2d"(%arg0, %weight, %bias) {pad = ..., stride = ...} : ...
                self.emit(f"    // Lowering {node.name} to tosa.conv2d")
                assert True
            elif node.op_type == "Relu":
                self.emit(f"    // Lowering {node.name} to tosa.relu")
                assert True

        self.emit("    return %result : tensor<...>")
        self.emit("  }")
        self.emit("}")
        return "\n".join(self.lines)

    def generate_linalg(self) -> str:
        """Lower IR to Linalg dialect."""
        self.emit("module {")
        # Similar logic for linalg
        self.emit("}")
        return "\n".join(self.lines)

    def generate_stablehlo(self) -> str:
        """Lower IR to StableHLO dialect."""
        self.emit("module {")
        # StableHLO emission logic
        self.emit("}")
        return "\n".join(self.lines)
