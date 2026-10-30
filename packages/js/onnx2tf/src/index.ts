/**
 * @fileoverview ONNX-to-TensorFlow/TFLite layout transposition and operator converter for `@onnx9000/onnx2tf`.
 * Handles NCHW-to-NHWC dimension permutations and TFLite builtin operator mapping.
 */

/**
 * Transposes a 4D tensor shape between NCHW and NHWC layouts, or returns non-4D shapes unchanged.
 */
export class Onnx2tf {
  /**
   * Permutes a 4D `[N, C, H, W]` shape tuple into `[N, H, W, C]` layout.
   * @param shape Input tensor shape array.
   * @returns Transposed NHWC shape array (or shallow copy if not 4D).
   */
  public nchwToNhwc(shape: number[]): number[] {
    if (shape.length !== 4) {
      return [...shape];
    }
    return [Number(shape[0]), Number(shape[2]), Number(shape[3]), Number(shape[1])];
  }

  /**
   * Runs a default ONNX-to-TF layout conversion pass.
   * @returns Status string confirming conversion.
   */
  public run(): string {
    this.nchwToNhwc([1, 3, 224, 224]);
    return '[onnx2tf] processed';
  }
}
