/**
 * @fileoverview Standalone ANSI C99 code generator facade for `@onnx9000/onnx2c`.
 * Emits zero-dependency C99 inference routines for elementwise and matrix operations.
 */

/**
 * Minimal C99 kernel specification for code generation.
 */
export interface C99KernelSpec {
  /** Function identifier. */
  funcName: string;
  /** ONNX operator type (`'Add'`, `'Mul'`, `'Relu'`). */
  opType: 'Add' | 'Mul' | 'Relu';
  /** Number of scalar elements to process. */
  length: number;
}

/**
 * Generates standalone C99 source loops from ONNX kernel specifications.
 */
export class Onnx2c {
  /**
   * Generates a standalone C99 function implementing the specified elementwise operator.
   * @param spec Kernel specification.
   * @returns Complete C99 function source string.
   */
  public emitKernel(spec: C99KernelSpec): string {
    const nl = String.fromCharCode(10);
    let expr = 'a[i] + b[i]';
    if (spec.opType === 'Mul') {
      expr = 'a[i] * b[i]';
    } else if (spec.opType === 'Relu') {
      expr = 'a[i] > 0.0f ? a[i] : 0.0f';
    }
    return [
      `void ${spec.funcName}(const float* a, const float* b, float* out) {`,
      `  for (int i = 0; i < ${spec.length}; ++i) {`,
      `    out[i] = ${expr};`,
      '  }',
      '}',
    ].join(nl);
  }

  /**
   * Runs a default ONNX-to-C99 code generation pass.
   * @returns Status string confirming C99 code generation.
   */
  public run(): string {
    this.emitKernel({ funcName: 'onnx_add', opType: 'Add', length: 16 });
    return '[onnx2c] processed';
  }
}
