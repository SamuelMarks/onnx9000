import { Tokenizer, BasicTokenizer } from './tokenizer.js';
import { ModelParams, GeneratorParams } from './types.js';
import { Generator } from './generator.js';
import { State } from './state.js';
import { Tensor } from '../ir/tensor.js';

/**
 * Base Model class for GenAI wrappers.
 */
export abstract class Model {
  protected params: ModelParams;

  constructor(params: ModelParams) {
    this.params = params;
  }

  /**
   * Factory method to create a generator instance for this model.
   */
  /** Asynchronously load model weights during pre-fill. */
  async loadWeights(): Promise<void> {}

  /** Create tokenizer associated with this model */
  createTokenizer(): Tokenizer {
    return new BasicTokenizer();
  }

  abstract createGenerator(params: GeneratorParams): Generator;

  /**
   * High-level `generate()` API.
   * @param promptIds Tensor containing input token IDs.
   * @param params Generation parameters.
   * @returns AsyncGenerator yielding token IDs.
   */
  async *generate(promptIds: Tensor, params: GeneratorParams): AsyncGenerator<number, void, void> {
    const generator = this.createGenerator(params);
    yield* generator.generate(promptIds);
  }
}

/** Multi-modal model implementation. */
export class MultiModalModel {}

/** Whisper-based speech recognition model. */
export class WhisperModel {}

/** T5 sequence-to-sequence model. */
export class T5Model {}

/** OPT model implementation. */
export class OptModel {}

/** BART sequence-to-sequence model. */
export class BartModel {}

/** GPT-NeoX model implementation. */
export class GptNeoXModel {}

/** Mixture-of-Experts (MoE) model. */
export class MoEModel {}

/** Model supporting speculative decoding optimizations. */
export class SpeculativeDecodingModel {}

/** LoRA adapter for parameter-efficient fine-tuning. */
export class LoraAdapter {}
