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
   */
  async *generate(
    promptIds: Tensor,
    params: GeneratorParams,
  ): AsyncGenerator<number, void, unknown> {
    const generator = this.createGenerator(params);
    yield* generator.generate(promptIds);
  }
}

export class MultiModalModel {}

export class WhisperModel {}

export class T5Model {}

export class OptModel {}

export class BartModel {}

export class GptNeoXModel {}

export class MoEModel {}

export class SpeculativeDecodingModel {}

export class LoraAdapter {}
