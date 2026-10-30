/**
 * @fileoverview Typed TypeScript DSL for authoring ONNX graphs in `@onnx9000/onnx-script`.
 * Enables fluent construction of ONNX operator nodes and symbolic values.
 */

/**
 * Recorded operator node in an ONNX Script function body.
 */
export interface ScriptNode {
  /** ONNX operator type. */
  opType: string;
  /** Input variable names. */
  inputs: string[];
  /** Output variable name. */
  output: string;
}

/**
 * Fluent builder for constructing ONNX graphs via a typed TypeScript DSL.
 */
export class Onnxscript {
  /** Ordered list of recorded script nodes. */
  public readonly nodes: ScriptNode[] = [];
  /** Counter for generating unique SSA variable names. */
  private counter = 0;

  /**
   * Records an operator invocation in the script graph and returns its SSA output name.
   * @param opType ONNX operator type (e.g., `'Add'`, `'Relu'`, `'MatMul'`).
   * @param inputs Input variable names.
   * @returns Newly allocated SSA variable name (e.g., `'%v0'`).
   */
  public op(opType: string, inputs: string[]): string {
    const output = `%v${this.counter++}`;
    this.nodes.push({ opType, inputs: [...inputs], output });
    return output;
  }

  /**
   * Runs a default ONNX Script graph construction pass.
   * @returns Status string confirming script execution.
   */
  public run(): string {
    const sum = this.op('Add', ['x', 'b']);
    this.op('Relu', [sum]);
    return '[onnx-script] processed';
  }
}
