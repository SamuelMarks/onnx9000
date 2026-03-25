/**
 * Utility to validate transitions between pipeline states.
 */
export class PipelineValidator {
  private static VALID_TRANSITIONS: Record<string, string[]> = {
    '.onnx': ['onnx-simplifier', 'olive', 'onnx-mlir', 'onnx2c', 'ort-web'],
    mlir: ['iree-compiler'],
    cpp: ['emscripten'],
    // All sources logically convert to .onnx in the 1-step logic
    keras: ['.onnx'],
    tensorflow: ['.onnx'],
    onnxscript: ['.onnx'],
    caffe: ['.onnx'],
    mxnet: ['.onnx'],
    paddle: ['.onnx'],
    scikitlearn: ['.onnx'],
    lightgbm: ['.onnx'],
    xgboost: ['.onnx'],
    catboost: ['.onnx'],
    sparkml: ['.onnx']
  };

  /**
   * Validates if a target framework is a valid next step for a given source framework.
   */
  public static isValidTransition(source: string, target: string): boolean {
    const validTargets = this.VALID_TRANSITIONS[source.toLowerCase()];
    if (!validTargets) return false;

    return validTargets.includes(target.toLowerCase());
  }

  /**
   * Get all valid targets for a given source.
   */
  public static getValidTargets(source: string): string[] {
    return this.VALID_TRANSITIONS[source.toLowerCase()] || [];
  }
}
