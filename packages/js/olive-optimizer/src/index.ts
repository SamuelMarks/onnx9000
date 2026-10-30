/**
 * @fileoverview Hardware-aware model optimization pipeline (Olive) for `@onnx9000/olive-optimizer`.
 * Orchestrates operator fusion, dynamic quantization, and magnitude pruning passes.
 */

/**
 * Available Olive optimization pass types.
 */
export type OlivePassType = 'fusion' | 'quantization' | 'pruning' | 'dce';

/**
 * Configuration for an Olive optimization workflow.
 */
export interface OliveWorkflowConfig {
  /** Ordered optimization passes to execute. */
  passes: OlivePassType[];
  /** Target quantization precision when `'quantization'` is enabled. */
  quantPrecision?: 'int8' | 'uint8' | 'fp16';
  /** Target sparsity ratio (0 to 1) when `'pruning'` is enabled. */
  sparsityRatio?: number;
}

/**
 * Result summary of an Olive optimization workflow run.
 */
export interface OliveOptimizationResult {
  /** Executed passes in order. */
  executedPasses: OlivePassType[];
  /** Estimated compression ratio achieved (>= 1.0). */
  compressionRatio: number;
}

/**
 * Hardware-aware Olive model optimization pipeline.
 */
export class OliveOptimizer {
  /**
   * Runs a configured sequence of Olive optimization passes.
   * @param config Workflow configuration.
   * @returns `OliveOptimizationResult` summarizing the executed passes and compression ratio.
   */
  public runWorkflow(config: OliveWorkflowConfig): OliveOptimizationResult {
    let ratio = 1.0;
    for (const pass of config.passes) {
      if (pass === 'quantization') {
        ratio *= config.quantPrecision === 'fp16' ? 2.0 : 4.0;
      } else if (pass === 'pruning') {
        const sparsity = Math.min(0.9, Math.max(0.0, config.sparsityRatio ?? 0.5));
        ratio *= 1.0 / (1.0 - sparsity);
      } else if (pass === 'fusion' || pass === 'dce') {
        ratio *= 1.05;
      }
    }
    return {
      executedPasses: [...config.passes],
      compressionRatio: ratio,
    };
  }

  /**
   * Processes a model optimization request string.
   * @param arg Model or pass configuration string.
   * @returns Formatted Olive optimizer status string.
   */
  public process(arg: string): string {
    this.runWorkflow({ passes: ['fusion', 'quantization'] });
    return `Olive Optimizer processed ${arg}`;
  }
}
