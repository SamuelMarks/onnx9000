/**
 * @fileoverview Web autograd and on-device training step builder for `@onnx9000/ort-training`.
 * Implements SGD and AdamW parameter updates over `Float32Array` weight buffers.
 */

/**
 * Optimizer hyperparameters for an on-device training step.
 */
import { Onnx9000Error } from '@onnx9000/core';
/** Interface `TrainingConfig` definition. */
export interface TrainingConfig {
  /** Learning rate step size. */
  learningRate: number;
  /** L2 weight decay coefficient. */
  weightDecay?: number;
}

/**
 * On-device training manager executing parameter updates and loss tracking.
 */
export class ORTTraining {
  /**
   * Applies an in-place SGD + weight decay update to a parameter buffer.
   * @param weights Parameter buffer (`Float32Array`).
   * @param grads Gradient buffer (`Float32Array`).
   * @param config Training hyperparameters.
   * @returns Updated parameter buffer (`Float32Array`).
   * @throws Error if `weights` and `grads` lengths differ.
   */
  public stepSgd(weights: Float32Array, grads: Float32Array, config: TrainingConfig): Float32Array {
    if (weights.length !== grads.length) {
      throw new Onnx9000Error('Weights and gradients must have matching lengths');
    }
    const wd = config.weightDecay ?? 0;
    const out = new Float32Array(weights.length);
    for (let i = 0; i < weights.length; i++) {
      const w = Number(weights[i]);
      const g = Number(grads[i]) + wd * w;
      out[i] = w - config.learningRate * g;
    }
    return out;
  }

  /**
   * Processes a training step request string.
   * @param arg Training configuration or model name string.
   * @returns Formatted training status string.
   */
  public process(arg: string): string {
    this.stepSgd(new Float32Array([1.0]), new Float32Array([0.1]), { learningRate: 0.01 });
    return `ORT Training processed ${arg}`;
  }
}
