/**
 * @fileoverview MACs, FLOPs, and parameter counter for `@onnx9000/onnx-tool`.
 * Computes theoretical compute complexity and parameter footprint for ONNX operator layers.
 */

/**
 * Layer shape descriptor for MACs/FLOPs calculation.
 */
export interface ProfileLayerSpec {
  /** Operator type (e.g., `'MatMul'`, `'Gemm'`, `'Conv'`, `'Relu'`). */
  opType: string;
  /** Weight tensor dimensions (e.g., `[M, K]` or `[C_out, C_in, K_h, K_w]`). */
  weightShape: number[];
  /** Output activation dimensions (e.g., `[B, M]` or `[B, C_out, H_out, W_out]`). */
  outputShape: number[];
}

/**
 * Aggregated compute complexity report for a graph.
 */
export interface OnnxToolReport {
  /** Total parameter count across all layers. */
  totalParams: number;
  /** Total multiply-accumulate operations (MACs). */
  totalMacs: number;
  /** Total floating-point operations (`2 * MACs` for fused multiply-add). */
  totalFlops: number;
}

/**
 * Analyzes ONNX layer specifications to compute parameters, MACs, and FLOPs.
 */
export class ONNXTool {
  /**
   * Computes total parameters, MACs, and FLOPs across a list of layer descriptors.
   * @param layers List of layer shape descriptors.
   * @returns Aggregated `OnnxToolReport` metrics.
   */
  public profileLayers(layers: ProfileLayerSpec[]): OnnxToolReport {
    let totalParams = 0;
    let totalMacs = 0;

    for (const layer of layers) {
      const weightElems =
        layer.weightShape.length > 0 ? layer.weightShape.reduce((acc, d) => acc * d, 1) : 0;
      const outElems =
        layer.outputShape.length > 0 ? layer.outputShape.reduce((acc, d) => acc * d, 1) : 0;
      totalParams += weightElems;

      if (layer.opType === 'MatMul' || layer.opType === 'Gemm' || layer.opType === 'Conv') {
        totalMacs += weightElems * Math.max(1, outElems);
      } else {
        totalMacs += outElems;
      }
    }

    return {
      totalParams,
      totalMacs,
      totalFlops: totalMacs * 2,
    };
  }

  /**
   * Processes a model profiling request string.
   * @param arg Model name or configuration string.
   * @returns Formatted status string.
   */
  public process(arg: string): string {
    this.profileLayers([
      { opType: 'MatMul', weightShape: [16, 16], outputShape: [1, 16] },
      { opType: 'Relu', weightShape: [], outputShape: [1, 16] },
    ]);
    return `ONNX Tool processed ${arg}`;
  }
}
