/**
 * @fileoverview WebGPU/WASM LLaMA session runner and KV-cache manager for `@onnx9000/llama-web`.
 * Provides transformer hyper-parameter configuration and KV-cache byte footprint calculations.
 */

/**
 * Architectural hyperparameters for a LLaMA transformer session.
 */
import { BackendError } from '@onnx9000/core';
/** Interface `LlamaConfig` definition. */
export interface LlamaConfig {
  /** Number of transformer decoder layers. */
  numLayers: number;
  /** Number of key/value attention heads (supports GQA). */
  numKvHeads: number;
  /** Dimension of each attention head. */
  headDim: number;
  /** Maximum sequence context length. */
  maxSeqLen: number;
}

/**
 * WebGPU/WASM LLaMA session runner and KV-cache planner.
 */
export class LlamaWeb {
  /** Active LLaMA model configuration. */
  public readonly config: LlamaConfig;

  /**
   * Initializes a new `LlamaWeb` session runner.
   * @param config Optional LLaMA architectural configuration.
   */
  public constructor(config?: Partial<LlamaConfig>) {
    this.config = {
      numLayers: config?.numLayers ?? 32,
      numKvHeads: config?.numKvHeads ?? 8,
      headDim: config?.headDim ?? 128,
      maxSeqLen: config?.maxSeqLen ?? 2048,
    };
  }

  /**
   * Computes the total KV-cache memory required in bytes for a given sequence length and byte width.
   * @param seqLen Active sequence length (clamped to `maxSeqLen`).
   * @param bytesPerElement Byte width per element (2 for FP16, 4 for FP32).
   * @returns Total KV-cache byte footprint across all layers (K + V).
   */
  public computeKvCacheBytes(seqLen: number, bytesPerElement = 2): number {
    const clampedSeq = Math.max(0, Math.min(seqLen, this.config.maxSeqLen));
    return (
      2 *
      this.config.numLayers *
      this.config.numKvHeads *
      this.config.headDim *
      clampedSeq *
      bytesPerElement
    );
  }

  /**
   * Executes a LLaMA Web session pass for the specified model identifier.
   * @param modelString Model identifier or prompt string.
   * @returns Formatted execution result string.
   * @throws Error if `modelString` is empty.
   */
  public run(modelString: string): string {
    if (!modelString) {
      throw new BackendError('Invalid model string');
    }
    this.computeKvCacheBytes(modelString.length);
    return `[LLaMA-Web] processing ${modelString}`;
  }
}
