import { Tensor } from '../index.js';

export interface LogitProcessor {
  process(inputIds: number[], logits: Tensor): Tensor;
}

export class TemperatureLogitProcessor implements LogitProcessor {
  constructor(private temperature: number) {
    if (temperature <= 0.0) {
      throw new Error('Temperature must be strictly positive.');
    }
  }

  process(inputIds: number[], logits: Tensor): Tensor {
    if (this.temperature === 1.0 || !(logits.data instanceof Float32Array)) {
      return logits;
    }

    const newData = new Float32Array(logits.data.length);
    for (let i = 0; i < logits.data.length; i++) {
      const data = logits.data as Float32Array;
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

export class TopKLogitProcessor implements LogitProcessor {
  constructor(private topK: number) {
    if (topK <= 0) {
      throw new Error('topK must be strictly positive.');
    }
  }

  process(inputIds: number[], logits: Tensor): Tensor {
    if (!(logits.data instanceof Float32Array)) {
      return logits;
    }

    const vocabSize = logits.shape[logits.shape.length - 1] as number;
    const offset = logits.data.length - vocabSize;

    const vals: { val: number; idx: number }[] = [];
    for (let i = 0; i < vocabSize; i++) {
      const data = logits.data as Float32Array;
      vals.push({ val: data[offset + i]!, idx: i });
    }

    vals.sort((a, b) => b.val - a.val);

    if (vals.length > this.topK) {
      const threshold = vals[this.topK - 1]!.val;
      const newData = new Float32Array(logits.data);

      for (let i = 0; i < vocabSize; i++) {
        const data = logits.data as Float32Array;
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

export class RepetitionPenaltyLogitProcessor implements LogitProcessor {
  constructor(private penalty: number) {
    if (penalty <= 0.0) {
      throw new Error('Penalty must be strictly positive.');
    }
  }

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

export class LogitProcessorList implements LogitProcessor {
  constructor(private processors: LogitProcessor[] = []) {}

  process(inputIds: number[], logits: Tensor): Tensor {
    let currentLogits = logits;
    for (const processor of this.processors) {
      currentLogits = processor.process(inputIds, currentLogits);
    }
    return currentLogits;
  }
}

export class MinPLogitProcessor implements LogitProcessor {
  constructor(private minP: number) {
    if (minP <= 0.0 || minP > 1.0) {
      throw new Error('minP must be in (0, 1].');
    }
  }

  process(inputIds: number[], logits: Tensor): Tensor {
    if (this.minP >= 1.0 || !(logits.data instanceof Float32Array)) {
      return logits;
    }

    const vocabSize = logits.shape[logits.shape.length - 1] as number;
    const offset = logits.data.length - vocabSize;

    let maxVal = -Infinity;
    for (let i = 0; i < vocabSize; i++) {
      const data = logits.data as Float32Array;
      const val = data[offset + i]!;

      if (val! > maxVal) {
        maxVal = val!;
      }
    }

    if (maxVal === -Infinity) {
      return logits;
    }

    const threshold = maxVal + Math.log(this.minP);
    const newData = new Float32Array(logits.data);

    for (let i = 0; i < vocabSize; i++) {
      const data = logits.data as Float32Array;
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

export class PresencePenaltyLogitProcessor implements LogitProcessor {
  constructor(private penalty: number) {}

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
        const data = logits.data as Float32Array;
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

export class FrequencyPenaltyLogitProcessor implements LogitProcessor {
  constructor(private penalty: number) {}

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
        const data = logits.data as Float32Array;
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

export class ForcedBOSLogitProcessor implements LogitProcessor {
  constructor(private bosTokenId: number) {}

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

export class ForcedEOSLogitProcessor implements LogitProcessor {
  constructor(
    private maxLength: number,
    private eosTokenId: number,
  ) {}

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

export class LogitBiasProcessor implements LogitProcessor {
  constructor(private biasMap: Map<number, number>) {}

  process(inputIds: number[], logits: Tensor): Tensor {
    if (this.biasMap.size === 0 || !(logits.data instanceof Float32Array)) {
      return logits;
    }

    const vocabSize = logits.shape[logits.shape.length - 1] as number;
    const offset = logits.data.length - vocabSize;
    const newData = new Float32Array(logits.data);

    for (const [tokenId, bias] of this.biasMap.entries()) {
      if (tokenId < vocabSize) {
        const data = logits.data as Float32Array;
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

export class NoRepeatNGramLogitProcessor implements LogitProcessor {
  constructor(private ngramSize: number) {
    if (ngramSize <= 0) {
      throw new Error('ngramSize must be strictly positive');
    }
  }

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

export class NoBadWordsLogitProcessor implements LogitProcessor {
  constructor(private badWordsIds: number[][]) {}

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

export class AllowedWordsLogitProcessor implements LogitProcessor {
  private allowedTokens: Set<number>;

  constructor(allowedTokenIds: number[]) {
    this.allowedTokens = new Set(allowedTokenIds);
  }

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

export class TypicalLogitProcessor implements LogitProcessor {
  private mass: number;
  constructor(mass: number = 0.9) {
    this.mass = mass;
  }
  process(inputIds: number[], scores: Tensor): Tensor {
    return scores;
  }
}

export class DiverseBeamSearchLogitProcessor implements LogitProcessor {
  constructor(
    private numBeamGroups: number,
    private numBeams: number,
    private diversityPenalty: number,
  ) {}
  process(inputIds: number[], scores: Tensor): Tensor {
    return scores;
  }
}

export class ContrastiveSearchLogitProcessor implements LogitProcessor {
  constructor(private penaltyAlpha: number) {}
  process(inputIds: number[], scores: Tensor): Tensor {
    return scores;
  }
}

export class GrammarGuidedLogitProcessor {}

export class JSONSchemaLogitProcessor {}

export class RegexLogitProcessor {}

export class WatermarkLogitProcessor {}

export class ComplexStoppingCriteria {}
