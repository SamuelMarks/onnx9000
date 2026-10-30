/**
 * @fileoverview MLIR AST and dialect operation builders for `@onnx9000/mlir`.
 * Supports `mhlo`, `tosa`, `linalg`, `tensor`, and `func` dialect textual emission.
 */

/**
 * Supported MLIR dialect namespaces.
 */
export type MlirDialect = 'mhlo' | 'tosa' | 'linalg' | 'tensor' | 'func';

/**
 * Represents a single typed MLIR operation in an SSA block.
 */
export interface MlirOperation {
  /** Dialect namespace (e.g., `'tosa'`, `'linalg'`). */
  dialect: MlirDialect;
  /** Operation mnemonic (e.g., `'add'`, `'matmul'`). */
  mnemonic: string;
  /** SSA result value name (e.g., `'%0'`). */
  result: string;
  /** SSA operand names. */
  operands: string[];
  /** MLIR tensor or scalar type signature (e.g., `'tensor<4xf32>'`). */
  resultType: string;
}

/**
 * MLIR module builder and textual formatter.
 */
export class Mlir {
  /** Operations registered in the current MLIR module body. */
  public readonly operations: MlirOperation[] = [];

  /**
   * Appends an MLIR dialect operation to the module body.
   * @param op MLIR operation descriptor.
   */
  public addOperation(op: MlirOperation): void {
    this.operations.push(op);
  }

  /**
   * Formats all recorded operations as a textual MLIR `module` block.
   * @param funcName Function symbol name (defaults to `'main'`).
   * @returns Formatted MLIR text.
   */
  public emitModule(funcName = 'main'): string {
    const nl = String.fromCharCode(10);
    const body = this.operations.map(
      (op) =>
        `    ${op.result} = "${op.dialect}.${op.mnemonic}"(${op.operands.join(', ')}) : () -> ${op.resultType}`,
    );
    return ['module {', `  func.func @${funcName}() {`, ...body, '    return', '  }', '}'].join(nl);
  }

  /**
   * Runs a default MLIR module construction pass.
   * @returns Status string confirming MLIR processing.
   */
  public run(): string {
    this.addOperation({
      dialect: 'tosa',
      mnemonic: 'add',
      result: '%0',
      operands: ['%arg0', '%arg1'],
      resultType: 'tensor<4xf32>',
    });
    this.emitModule();
    return '[mlir] processed';
  }
}
