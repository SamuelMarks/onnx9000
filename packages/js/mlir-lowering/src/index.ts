/**
 * @fileoverview ONNX to MLIR dialect lowering pass pipeline for `@onnx9000/mlir-lowering`.
 * Lowers ONNX operators into `mhlo`, `tosa`, and `linalg` dialect operations.
 */

/**
 * Target MLIR lowering dialect.
 */
export type LoweringTargetDialect = 'mhlo' | 'tosa' | 'linalg';

/**
 * Lowered MLIR instruction emitted from an ONNX node.
 */
export interface LoweredMlirOp {
  /** Source ONNX operator type. */
  onnxOp: string;
  /** Target MLIR dialect operation name. */
  mlirOp: string;
}

/**
 * ONNX-to-MLIR dialect lowering pass runner.
 */
export class Mlirlowering {
  /** Mapping from ONNX operator types to target dialect operations. */
  private static readonly OP_TABLE: Readonly<
    Record<LoweringTargetDialect, Readonly<Record<string, string>>>
  > = {
    mhlo: {
      Add: 'mhlo.add',
      Sub: 'mhlo.subtract',
      Mul: 'mhlo.multiply',
      MatMul: 'mhlo.dot_general',
      Relu: 'mhlo.maximum',
    },
    tosa: {
      Add: 'tosa.add',
      Sub: 'tosa.sub',
      Mul: 'tosa.mul',
      MatMul: 'tosa.matmul',
      Relu: 'tosa.clamp',
    },
    linalg: {
      Add: 'linalg.generic',
      Sub: 'linalg.generic',
      Mul: 'linalg.generic',
      MatMul: 'linalg.matmul',
      Relu: 'linalg.generic',
    },
  };

  /**
   * Lowers a sequence of ONNX operator names into the specified MLIR dialect.
   * @param onnxOps List of ONNX operator names.
   * @param dialect Target MLIR dialect (defaults to `'mhlo'`).
   * @returns Array of lowered MLIR operation descriptors.
   */
  public lowerOps(onnxOps: string[], dialect: LoweringTargetDialect = 'mhlo'): LoweredMlirOp[] {
    const table = Mlirlowering.OP_TABLE[dialect];
    return onnxOps.map((op) => ({
      onnxOp: op,
      mlirOp: table[op] ?? `${dialect}.custom_${op.toLowerCase()}`,
    }));
  }

  /**
   * Runs a default ONNX-to-MLIR lowering pass.
   * @returns Status string confirming MLIR lowering.
   */
  public run(): string {
    this.lowerOps(['MatMul', 'Add', 'Relu'], 'mhlo');
    return '[mlir-lowering] processed';
  }
}
