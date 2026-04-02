import { Tensor } from '../index.js';

/**
 * Interface for processors that modify model logits during generation.
 */
export interface LogitProcessor {
  /**
   * Process logits based on current sequence and model output.
   * @param inputIds Previous token sequence.
   * @param logits Raw output logits from the model.
   * @returns Modified logits tensor.
   */
  process(inputIds: number[], logits: Tensor): Tensor;
}

/**
 * Scales logits by a temperature factor.
 */
export class TemperatureLogitProcessor implements LogitProcessor {
  /**
   * Create a new TemperatureLogitProcessor.
   * @param temperature Scaling factor (> 0).
   */
  constructor(private temperature: number) {
    if (temperature <= 0.0) {
      throw new Error('Temperature must be strictly positive.');
    }
  }

  /** Process logits with temperature scaling. */
  process(inputIds: number[], logits: Tensor): Tensor {
    if (this.temperature === 1.0 || !(logits.data instanceof Float32Array)) {
      return logits;
    }

    const newData = new Float32Array(logits.data.length);
    for (let i = 0; i < logits.data.length; i++) {
      const data = logits.data;
      newData[i] = data[i]! / this.temperature;
    }

    return new Tensor(
      logits.name,
      logits.shape,
      logits.dtype,
      logits.isInitializer,
      logits.requiresGrad,
      newData,
    );
  }
}

/**
 * Keeps only the top-K most probable tokens and sets others to -Infinity.
 */
export class TopKLogitProcessor implements LogitProcessor {
  /**
   * Create a new TopKLogitProcessor.
   * @param topK Number of top tokens to keep.
   */
  constructor(private topK: number) {
    if (topK <= 0) {
      throw new Error('topK must be strictly positive.');
    }
  }

  /** Process logits with Top-K filtering. */
  process(inputIds: number[], logits: Tensor): Tensor {
    if (!(logits.data instanceof Float32Array)) {
      return logits;
    }

    const vocabSize = logits.shape[logits.shape.length - 1] as number;
    const offset = logits.data.length - vocabSize;

    const vals: { val: number; idx: number }[] = [];
    for (let i = 0; i < vocabSize; i++) {
      const data = logits.data;
      vals.push({ val: data[offset + i]!, idx: i });
    }

    vals.sort((a, b) => b.val - a.val);

    if (vals.length > this.topK) {
      const threshold = vals[this.topK - 1]!.val;
      const newData = new Float32Array(logits.data);

      for (let i = 0; i < vocabSize; i++) {
        const data = logits.data;
        if (data[offset + i]! < threshold) {
          newData[offset + i] = -Infinity;
        }
      }
      return new Tensor(
        logits.name,
        logits.shape,
        logits.dtype,
        logits.isInitializer,
        logits.requiresGrad,
        newData,
      );
    }

    return logits;
  }
}

/**
 * Penalizes tokens that have already appeared in the sequence.
 */
export class RepetitionPenaltyLogitProcessor implements LogitProcessor {
  /**
   * Create a new RepetitionPenaltyLogitProcessor.
   * @param penalty Penalty factor (> 1.0 increases penalty).
   */
  constructor(private penalty: number) {
    if (penalty <= 0.0) {
      throw new Error('Penalty must be strictly positive.');
    }
  }

  /** Process logits with repetition penalty. */
  process(inputIds: number[], logits: Tensor): Tensor {
    if (this.penalty === 1.0 || !(logits.data instanceof Float32Array) || inputIds.length === 0) {
      return logits;
    }

    const vocabSize = logits.shape[logits.shape.length - 1] as number;
    const offset = logits.data.length - vocabSize;
    const newData = new Float32Array(logits.data);

    const uniqueIds = new Set(inputIds);
    for (const tokenId of uniqueIds) {
      if (tokenId < vocabSize) {
        const idx = offset + tokenId;
        let val = logits.data[idx];
        if (val! < 0) {
          val = val! * this.penalty;
        } else {
          val = val! / this.penalty;
        }
        newData[idx] = val;
      }
    }

    return new Tensor(
      logits.name,
      logits.shape,
      logits.dtype,
      logits.isInitializer,
      logits.requiresGrad,
      newData,
    );
  }
}

/**
 * Composite list of logit processors executed in sequence.
 */
export class LogitProcessorList implements LogitProcessor {
  /**
   * Create a new LogitProcessorList.
   * @param processors Initial list of processors.
   */
  constructor(private processors: LogitProcessor[] = []) {}

  /** Process logits through all registered processors. */
  process(inputIds: number[], logits: Tensor): Tensor {
    let currentLogits = logits;
    for (const processor of this.processors) {
      currentLogits = processor.process(inputIds, currentLogits);
    }
    return currentLogits;
  }
}

/**
 * Filters logits based on their relative probability compared to the maximum.
 */
export class MinPLogitProcessor implements LogitProcessor {
  /**
   * Create a new MinPLogitProcessor.
   * @param minP Minimum probability threshold relative to the top choice.
   */
  constructor(private minP: number) {
    if (minP <= 0.0 || minP > 1.0) {
      throw new Error('minP must be in (0, 1].');
    }
  }

  /** Process logits with Min-P filtering. */
  process(inputIds: number[], logits: Tensor): Tensor {
    if (this.minP >= 1.0 || !(logits.data instanceof Float32Array)) {
      return logits;
    }

    const vocabSize = logits.shape[logits.shape.length - 1] as number;
    const offset = logits.data.length - vocabSize;

    let maxVal = -Infinity;
    for (let i = 0; i < vocabSize; i++) {
      const data = logits.data;
      const val = data[offset + i]!;

      if (val > maxVal) {
        maxVal = val!;
      }
    }

    if (maxVal === -Infinity) {
      return logits;
    }

    const threshold = maxVal + Math.log(this.minP);
    const newData = new Float32Array(logits.data);

    for (let i = 0; i < vocabSize; i++) {
      const data = logits.data;
      if (data[offset + i]! < threshold) {
        newData[offset + i] = -Infinity;
      }
    }

    return new Tensor(
      logits.name,
      logits.shape,
      logits.dtype,
      logits.isInitializer,
      logits.requiresGrad,
      newData,
    );
  }
}

/**
 * Penalizes tokens based on their presence in the sequence.
 */
export class PresencePenaltyLogitProcessor implements LogitProcessor {
  /**
   * Create a new PresencePenaltyLogitProcessor.
   * @param penalty Fixed penalty added to seen tokens.
   */
  constructor(private penalty: number) {}

  /** Process logits with presence penalty. */
  process(inputIds: number[], logits: Tensor): Tensor {
    if (this.penalty === 0.0 || !(logits.data instanceof Float32Array) || inputIds.length === 0) {
      return logits;
    }

    const vocabSize = logits.shape[logits.shape.length - 1] as number;
    const offset = logits.data.length - vocabSize;
    const newData = new Float32Array(logits.data);

    const uniqueIds = new Set(inputIds);
    for (const tokenId of uniqueIds) {
      if (tokenId < vocabSize) {
        const idx = offset + tokenId;
        const data = logits.data;
        newData[idx] = data[idx]! - this.penalty;
      }
    }

    return new Tensor(
      logits.name,
      logits.shape,
      logits.dtype,
      logits.isInitializer,
      logits.requiresGrad,
      newData,
    );
  }
}

/**
 * Penalizes tokens proportionally to their frequency in the sequence.
 */
export class FrequencyPenaltyLogitProcessor implements LogitProcessor {
  /**
   * Create a new FrequencyPenaltyLogitProcessor.
   * @param penalty Penalty factor multiplied by token count.
   */
  constructor(private penalty: number) {}

  /** Process logits with frequency penalty. */
  process(inputIds: number[], logits: Tensor): Tensor {
    if (this.penalty === 0.0 || !(logits.data instanceof Float32Array) || inputIds.length === 0) {
      return logits;
    }

    const vocabSize = logits.shape[logits.shape.length - 1] as number;
    const offset = logits.data.length - vocabSize;
    const newData = new Float32Array(logits.data);

    const counts = new Map<number, number>();
    for (const id of inputIds) {
      counts.set(id, (counts.get(id) || 0) + 1);
    }

    for (const [tokenId, count] of counts.entries()) {
      if (tokenId < vocabSize) {
        const idx = offset + tokenId;
        const data = logits.data;
        newData[idx] = data[idx]! - this.penalty * count;
      }
    }

    return new Tensor(
      logits.name,
      logits.shape,
      logits.dtype,
      logits.isInitializer,
      logits.requiresGrad,
      newData,
    );
  }
}

/**
 * Forces a Begin Of Sequence (BOS) token to be selected at the first step.
 */
export class ForcedBOSLogitProcessor implements LogitProcessor {
  /**
   * Create a new ForcedBOSLogitProcessor.
   * @param bosTokenId The token ID to force.
   */
  constructor(private bosTokenId: number) {}

  /** Force BOS token at the start. */
  process(inputIds: number[], logits: Tensor): Tensor {
    if (inputIds.length === 0 && logits.data instanceof Float32Array) {
      const vocabSize = logits.shape[logits.shape.length - 1] as number;
      const offset = logits.data.length - vocabSize;
      const newData = new Float32Array(logits.data);

      for (let i = 0; i < vocabSize; i++) {
        if (i !== this.bosTokenId) {
          newData[offset + i] = -Infinity;
        }
      }
      return new Tensor(
        logits.name,
        logits.shape,
        logits.dtype,
        logits.isInitializer,
        logits.requiresGrad,
        newData,
      );
    }
    return logits;
  }
}

/**
 * Forces an End Of Sequence (EOS) token when maximum length is reached.
 */
export class ForcedEOSLogitProcessor implements LogitProcessor {
  /**
   * Create a new ForcedEOSLogitProcessor.
   * @param maxLength Maximum allowed sequence length.
   * @param eosTokenId The token ID to force.
   */
  constructor(
    private maxLength: number,
    private eosTokenId: number,
  ) {}

  /** Force EOS token at the limit. */
  process(inputIds: number[], logits: Tensor): Tensor {
    if (inputIds.length === this.maxLength - 1 && logits.data instanceof Float32Array) {
      const vocabSize = logits.shape[logits.shape.length - 1] as number;
      const offset = logits.data.length - vocabSize;
      const newData = new Float32Array(logits.data);

      for (let i = 0; i < vocabSize; i++) {
        if (i !== this.eosTokenId) {
          newData[offset + i] = -Infinity;
        }
      }
      return new Tensor(
        logits.name,
        logits.shape,
        logits.dtype,
        logits.isInitializer,
        logits.requiresGrad,
        newData,
      );
    }
    return logits;
  }
}

/**
 * Applies a manual bias to specific token IDs.
 */
export class LogitBiasProcessor implements LogitProcessor {
  /**
   * Create a new LogitBiasProcessor.
   * @param biasMap Map of token IDs to bias values.
   */
  constructor(private biasMap: Map<number, number>) {}

  /** Apply manual biases to logits. */
  process(inputIds: number[], logits: Tensor): Tensor {
    if (this.biasMap.size === 0 || !(logits.data instanceof Float32Array)) {
      return logits;
    }

    const vocabSize = logits.shape[logits.shape.length - 1] as number;
    const offset = logits.data.length - vocabSize;
    const newData = new Float32Array(logits.data);

    for (const [tokenId, bias] of this.biasMap.entries()) {
      if (tokenId < vocabSize) {
        const data = logits.data;
        newData[offset + tokenId]! += bias;
      }
    }

    return new Tensor(
      logits.name,
      logits.shape,
      logits.dtype,
      logits.isInitializer,
      logits.requiresGrad,
      newData,
    );
  }
}

/**
 * Prevents repeating the same N-Gram twice in a sequence.
 */
export class NoRepeatNGramLogitProcessor implements LogitProcessor {
  /**
   * Create a new NoRepeatNGramLogitProcessor.
   * @param ngramSize Size of the N-Gram to block from repeating.
   */
  constructor(private ngramSize: number) {
    if (ngramSize <= 0) {
      throw new Error('ngramSize must be strictly positive');
    }
  }

  /** Block repeat N-Grams. */
  process(inputIds: number[], logits: Tensor): Tensor {
    if (
      this.ngramSize === 0 ||
      inputIds.length < this.ngramSize - 1 ||
      !(logits.data instanceof Float32Array)
    ) {
      return logits;
    }

    const vocabSize = logits.shape[logits.shape.length - 1] as number;
    const offset = logits.data.length - vocabSize;

    const prefix = this.ngramSize > 1 ? inputIds.slice(-(this.ngramSize - 1)) : [];
    const bannedTokens = new Set<number>();

    for (let i = 0; i <= inputIds.length - this.ngramSize; i++) {
      let match = true;
      for (let j = 0; j < prefix.length; j++) {
        if (inputIds[i + j] !== prefix[j]) {
          match = false;
          break;
        }
      }
      if (match) {
        bannedTokens.add(inputIds[i + this.ngramSize - 1]!);
      }
    }

    if (bannedTokens.size === 0) {
      return logits;
    }

    const newData = new Float32Array(logits.data);
    for (const tokenId of bannedTokens) {
      if (tokenId < vocabSize) {
        newData[offset + tokenId] = -Infinity;
      }
    }

    return new Tensor(
      logits.name,
      logits.shape,
      logits.dtype,
      logits.isInitializer,
      logits.requiresGrad,
      newData,
    );
  }
}

/**
 * Blocks specific word sequences from being generated.
 */
export class NoBadWordsLogitProcessor implements LogitProcessor {
  /**
   * Create a new NoBadWordsLogitProcessor.
   * @param badWordsIds List of forbidden token sequences.
   */
  constructor(private badWordsIds: number[][]) {}

  /** Block forbidden token sequences. */
  process(inputIds: number[], logits: Tensor): Tensor {
    if (this.badWordsIds.length === 0 || !(logits.data instanceof Float32Array)) {
      return logits;
    }

    const vocabSize = logits.shape[logits.shape.length - 1] as number;
    const offset = logits.data.length - vocabSize;
    const bannedTokens = new Set<number>();

    for (const badWord of this.badWordsIds) {
      if (badWord.length === 1) {
        bannedTokens.add(badWord[0]!);
      } else if (badWord.length > 1) {
        const prefix = badWord.slice(0, -1);
        if (inputIds.length >= prefix.length) {
          let match = true;
          for (let i = 0; i < prefix.length; i++) {
            if (inputIds[inputIds.length - prefix.length + i] !== prefix[i]) {
              match = false;
              break;
            }
          }
          if (match) {
            bannedTokens.add(badWord[badWord.length - 1]!);
          }
        }
      }
    }

    if (bannedTokens.size === 0) {
      return logits;
    }

    const newData = new Float32Array(logits.data);
    for (const tokenId of bannedTokens) {
      if (tokenId < vocabSize) {
        newData[offset + tokenId] = -Infinity;
      }
    }

    return new Tensor(
      logits.name,
      logits.shape,
      logits.dtype,
      logits.isInitializer,
      logits.requiresGrad,
      newData,
    );
  }
}

/**
 * Limits generation to a specific set of allowed tokens.
 */
export class AllowedWordsLogitProcessor implements LogitProcessor {
  private allowedTokens: Set<number>;

  /**
   * Create a new AllowedWordsLogitProcessor.
   * @param allowedTokenIds White-list of token IDs.
   */
  constructor(allowedTokenIds: number[]) {
    this.allowedTokens = new Set(allowedTokenIds);
  }

  /** Limit generation to allowed tokens. */
  process(inputIds: number[], logits: Tensor): Tensor {
    if (this.allowedTokens.size === 0 || !(logits.data instanceof Float32Array)) {
      return logits;
    }

    const vocabSize = logits.shape[logits.shape.length - 1] as number;
    const offset = logits.data.length - vocabSize;
    const newData = new Float32Array(logits.data);

    for (let i = 0; i < vocabSize; i++) {
      if (!this.allowedTokens.has(i)) {
        newData[offset + i] = -Infinity;
      }
    }

    return new Tensor(
      logits.name,
      logits.shape,
      logits.dtype,
      logits.isInitializer,
      logits.requiresGrad,
      newData,
    );
  }
}

/**
 * Typical sampling processor based on local information gain.
 */
export class TypicalLogitProcessor implements LogitProcessor {
  private mass: number;
  /** @param mass Targeted probability mass. */
  constructor(mass: number = 0.9) {
    this.mass = mass;
  }
  /** Placeholder for typical sampling implementation. */
  process(inputIds: number[], scores: Tensor): Tensor {
    return scores;
  }
}

/**
 * Diverse beam search processor that penalizes sibling beams.
 */
export class DiverseBeamSearchLogitProcessor implements LogitProcessor {
  /**
   * Create a new DiverseBeamSearchLogitProcessor.
   * @param numBeamGroups Number of beam groups.
   * @param numBeams Number of beams.
   * @param diversityPenalty Penalty for inter-group similarity.
   */
  constructor(
    private numBeamGroups: number,
    private numBeams: number,
    private diversityPenalty: number,
  ) {}
  /** Placeholder for diverse beam search implementation. */
  process(inputIds: number[], scores: Tensor): Tensor {
    return scores;
  }
}

/**
 * Contrastive search processor that penalizes tokens based on context similarity.
 */
export class ContrastiveSearchLogitProcessor implements LogitProcessor {
  /** @param penaltyAlpha Contrastive penalty alpha factor. */
  constructor(private penaltyAlpha: number) {}
  /** Placeholder for contrastive search implementation. */
  process(inputIds: number[], scores: Tensor): Tensor {
    return scores;
  }
}

/** Grammar-guided generation processor placeholder. */
export class GrammarGuidedLogitProcessor {}

/** JSON Schema-constrained generation processor placeholder. */
export class JSONSchemaLogitProcessor {}

/** Regex-constrained generation processor placeholder. */
export class RegexLogitProcessor {}

/** Digital watermarking generation processor placeholder. */
export class WatermarkLogitProcessor {}

/** Advanced generation stopping criteria placeholder. */
export class ComplexStoppingCriteria {}
