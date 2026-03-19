import { Tensor } from '../ir/tensor.js';
import { State } from './state.js';
import { GeneratorParams } from './types.js';

/**
 * Base Generator class for stateful decoding.
 */
export abstract class Generator {
  protected state: State;
  protected params: GeneratorParams;

  constructor(state: State, params: GeneratorParams) {
    this.state = state;
    this.params = params;
  }

  /**
   * Compute logits for the current state.
   */
  abstract computeLogits(inputIds: Tensor): Promise<Tensor>;

  /**
   * Compute logits synchronously (if supported).
   */
  abstract computeLogitsSync(inputIds: Tensor): Tensor;

  /**
   * Process pre-fill phase.
   */
  abstract prefill(promptIds: Tensor): Promise<Tensor>;

  /**
   * Process a single decoding step.
   */
  abstract decodeStep(tokenId: number): Promise<Tensor>;

  /**
   * High-level generation API. Yields tokens as they are generated.
   */
  async *generate(promptIds: Tensor): AsyncGenerator<number, void, unknown> {
    let currentTokens = 0;
    const maxTokens =
      this.params.maxNewTokens ??
      this.params.maxLength - (promptIds.shape[promptIds.shape.length - 1] as number);

    let logits = await this.prefill(promptIds);
    let nextToken = this.sample(logits);

    yield nextToken;
    currentTokens++;

    while (currentTokens < maxTokens) {
      // Check early stopping
      if (this.params.abortSignal?.aborted) {
        break;
      }
      if (this.params.earlyStopping && this.isEos(nextToken)) {
        break;
      }

      logits = await this.decodeStep(nextToken);
      nextToken = this.sample(logits);

      yield nextToken;
      currentTokens++;
    }
  }

  /**
   * Sample the next token from logits.
   */
  protected sample(logits: Tensor): number {
    // Basic greedy search implementation for now.
    // Needs proper logit processing pipeline later.
    if (logits.data instanceof Float32Array || logits.data instanceof Float64Array) {
      const data = logits.data;
      const vocabSize = logits.shape[logits.shape.length - 1] as number;
      // Get the last logits
      const offset = data.length - vocabSize;
      let maxVal = -Infinity;
      let maxIdx = -1;
      for (let i = 0; i < vocabSize; i++) {
        if (data[offset + i]! > maxVal) {
          maxVal = data[offset + i]!;
          maxIdx = i;
        }
      }
      return maxIdx;
    }
    throw new Error('Unsupported logit data type for sampling.');
  }

  protected isEos(tokenId: number): boolean {
    // Implement proper EOS checking logic with ModelParams
    return false; // placeholder
  }
}
