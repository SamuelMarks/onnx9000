import { Tensor } from '../index.js';
import { LogitProcessor } from './logit_processors.js';

export class TopPLogitProcessor implements LogitProcessor {
  constructor(private topP: number) {
    if (topP <= 0 || topP > 1.0) {
      throw new Error('topP must be in (0, 1].');
    }
  }

  process(inputIds: number[], logits: Tensor): Tensor {
    if (this.topP >= 1.0 || !(logits.data instanceof Float32Array)) {
      return logits;
    }

    const vocabSize = logits.shape[logits.shape.length - 1] as number;
    const offset = logits.data.length - vocabSize;

    // Apply softmax first to get probabilities
    const maxLogit = Math.max(...Array.from(logits.data.subarray(offset)));
    const probs = new Float32Array(vocabSize);
    let sumProbs = 0;

    for (let i = 0; i < vocabSize; i++) {
      const data = logits.data as Float32Array;
      probs[i] = Math.exp(data[offset + i]! - maxLogit);
      sumProbs += probs[i]!;
    }

    for (let i = 0; i < vocabSize; i++) {
      probs[i] = probs[i]! / sumProbs;
    }

    const vals: { val: number; idx: number }[] = [];
    for (let i = 0; i < vocabSize; i++) {
      vals.push({ val: probs[i]!, idx: i });
    }

    vals.sort((a, b) => b.val - a.val);

    let cumulativeProb = 0;
    let thresholdIdx = vocabSize - 1;

    for (let i = 0; i < vocabSize; i++) {
      cumulativeProb += vals[i]!.val;
      if (cumulativeProb > this.topP) {
        thresholdIdx = i;
        break;
      }
    }

    // We keep at least one token
    const indicesToRemove = vals.slice(thresholdIdx + 1).map((x) => x.idx);
    if (indicesToRemove.length > 0) {
      const newData = new Float32Array(logits.data);
      for (const idx of indicesToRemove) {
        newData[offset + idx] = -Infinity;
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
