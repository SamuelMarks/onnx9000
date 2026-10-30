/**
 * @fileoverview ONNX graph and model structural validator for `@onnx9000/onnx-checker`.
 * Validates IR version bounds, opset imports, and topological tensor dependencies.
 */

/**
 * Minimal ONNX model descriptor for structural validation.
 */
export interface CheckableModel {
  /** ONNX IR version (valid range: 3..10). */
  irVersion: number;
  /** Imported opset versions (must be non-empty). */
  opsetVersions: number[];
  /** Graph input tensor names. */
  inputs: string[];
  /** Ordered nodes with input and output tensor names. */
  nodes: { opType: string; inputs: string[]; outputs: string[] }[];
}

/**
 * ONNX model checker verifying structural and topological invariants.
 */
export class Onnxchecker {
  /**
   * Validates an ONNX model descriptor and returns any diagnostic error messages.
   * @param model Model descriptor to validate.
   * @returns Array of validation error strings (empty if valid).
   */
  public validate(model: CheckableModel): string[] {
    const errors: string[] = [];
    if (model.irVersion < 3 || model.irVersion > 10) {
      errors.push(`Unsupported irVersion: ${model.irVersion}`);
    }
    if (model.opsetVersions.length === 0) {
      errors.push('Missing opset import');
    }
    const knownTensors = new Set<string>(model.inputs);
    for (const node of model.nodes) {
      for (const inp of node.inputs) {
        if (!knownTensors.has(inp)) {
          errors.push(`Unresolved input '${inp}' in node '${node.opType}'`);
        }
      }
      for (const out of node.outputs) {
        knownTensors.add(out);
      }
    }
    return errors;
  }

  /**
   * Runs a default ONNX model validation check.
   * @returns Status string confirming checker execution.
   */
  public run(): string {
    this.validate({
      irVersion: 8,
      opsetVersions: [18],
      inputs: ['x'],
      nodes: [{ opType: 'Relu', inputs: ['x'], outputs: ['y'] }],
    });
    return '[onnx-checker] processed';
  }
}
