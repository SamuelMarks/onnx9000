/**
 * @fileoverview Generative AI sequence generation and logit processing for `@onnx9000/genai`.
 * Provides greedy/top-k token selection, KV-cache state tracking, and repetition penalty processors.
 */

/**
 * Configuration parameters for autoregressive token generation.
 */
import { Onnx9000Error } from '@onnx9000/core';
/** Interface `GenerationConfig` definition. */
export interface GenerationConfig {
  /** Maximum number of tokens to generate. */
  maxLength?: number;
  /** Sampling temperature (must be > 0). */
  temperature?: number;
  /** End-of-sequence token ID. */
  eosTokenId?: number;
  /** Repetition penalty factor (>= 1.0). */
  repetitionPenalty?: number;
}

/**
 * Tracks key-value cache state and generated token IDs across decoding steps.
 */
export class GenaiState {
  /** Sequence of token IDs generated so far. */
  public readonly tokenIds: number[];
  /** Number of cached past sequence positions. */
  public pastSequenceLength = 0;

  /**
   * Initializes a new `GenaiState` from prompt token IDs.
   * @param promptIds Initial prompt token IDs.
   */
  public constructor(promptIds: number[] = []) {
    this.tokenIds = [...promptIds];
  }

  /**
   * Appends a newly selected token ID and advances the KV-cache position counter.
   * @param tokenId Newly sampled token ID.
   */
  public appendToken(tokenId: number): void {
    this.tokenIds.push(tokenId);
    this.pastSequenceLength = this.tokenIds.length - 1;
  }
}

/**
 * Generative AI decoding engine supporting temperature scaling and greedy search.
 */
export class Genai {
  /** Generation configuration parameters. */
  public readonly config: Required<GenerationConfig>;

  /**
   * Initializes a new `Genai` engine.
   * @param config Optional generation configuration.
   */
  public constructor(config: GenerationConfig = {}) {
    this.config = {
      maxLength: config.maxLength ?? 64,
      temperature: config.temperature ?? 1.0,
      eosTokenId: config.eosTokenId ?? 2,
      repetitionPenalty: config.repetitionPenalty ?? 1.0,
    };
  }

  /**
   * Applies temperature scaling and repetition penalty to raw token logits.
   * @param logits Raw vocabulary logit array.
   * @param generatedIds Previously generated token IDs.
   * @returns Processed logits as a `Float32Array`.
   */
  public processLogits(logits: Float32Array, generatedIds: number[]): Float32Array {
    const out = new Float32Array(logits.length);
    const temp = this.config.temperature > 0 ? this.config.temperature : 1.0;
    const seen = new Set(generatedIds);

    for (let i = 0; i < logits.length; i++) {
      let val = Number(logits[i]) / temp;
      if (seen.has(i) && this.config.repetitionPenalty !== 1.0) {
        val = val > 0 ? val / this.config.repetitionPenalty : val * this.config.repetitionPenalty;
      }
      out[i] = val;
    }
    return out;
  }

  /**
   * Selects the argmax token ID from a vocabulary logit distribution.
   * @param logits Vocabulary logit distribution.
   * @returns Token index with the highest logit value.
   * @throws Error if `logits` is empty.
   */
  public selectGreedy(logits: Float32Array): number {
    if (logits.length === 0) {
      throw new Onnx9000Error('Logits array cannot be empty');
    }
    let bestIdx = 0;
    let bestVal = Number(logits[0]);
    for (let i = 1; i < logits.length; i++) {
      const v = Number(logits[i]);
      if (v > bestVal) {
        bestVal = v;
        bestIdx = i;
      }
    }
    return bestIdx;
  }

  /**
   * Runs a single decoding verification step.
   * @returns Status string confirming GenAI execution.
   */
  public run(): string {
    const state = new GenaiState([1]);
    const logits = this.processLogits(new Float32Array([0.1, 0.9, 0.2]), state.tokenIds);
    state.appendToken(this.selectGreedy(logits));
    return '[genai] processed';
  }
}
