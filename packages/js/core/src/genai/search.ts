import { Tensor } from '../index.js';

export interface SearchAlgorithm {
  selectNextToken(logits: Tensor, inputIds: number[]): number;
}

export class GreedySearch implements SearchAlgorithm {
  selectNextToken(logits: Tensor, inputIds: number[]): number {
    if (!(logits.data instanceof Float32Array)) {
      return 0;
    }

    const vocabSize = logits.shape[logits.shape.length - 1] as number;
    const offset = logits.data.length - vocabSize;

    let maxVal = -Infinity;
    let maxIdx = 0;

    for (let i = 0; i < vocabSize; i++) {
      const data = logits.data;
      const val = data[offset + i]!;

      if (isNaN(val) || (!isFinite(val) && val > 0)) {
        return 0;
      }
      if (val > maxVal) {
        maxVal = val!;
        maxIdx = i;
      }
    }

    return maxIdx;
  }
}

export class MultinomialSampling implements SearchAlgorithm {
  selectNextToken(logits: Tensor, inputIds: number[]): number {
    if (!(logits.data instanceof Float32Array)) {
      return 0;
    }

    const vocabSize = logits.shape[logits.shape.length - 1] as number;
    const offset = logits.data.length - vocabSize;

    let maxVal = -Infinity;
    for (let i = 0; i < vocabSize; i++) {
      const data = logits.data;
      if (data[offset + i]! > maxVal) {
        maxVal = data[offset + i]!;
      }
    }

    let sumExp = 0;
    const probs = new Float32Array(vocabSize);
    for (let i = 0; i < vocabSize; i++) {
      const data = logits.data;
      probs[i] = Math.exp(data[offset + i]! - maxVal);
      sumExp += probs[i]!;
    }

    for (let i = 0; i < vocabSize; i++) {
      probs[i] = probs[i]! / sumExp;
    }

    // Simplistic pseudo-randomness for mock implementation without explicit seed handling
    const r = Math.random();
    let cumulative = 0.0;
    for (let i = 0; i < vocabSize; i++) {
      cumulative += probs[i]!;
      if (r <= cumulative) {
        return i;
      }
    }
    return vocabSize - 1;
  }
}

export class BeamSearchState {
  numBeams: number;
  numReturnSequences: number;
  activeBeams: { score: number; tokens: number[] }[];
  finishedBeams: { score: number; tokens: number[] }[];

  constructor(numBeams: number, numReturnSequences: number) {
    this.numBeams = numBeams;
    this.numReturnSequences = numReturnSequences;
    this.activeBeams = [];
    this.finishedBeams = [];
  }

  addFinished(score: number, tokens: number[]): void {
    this.finishedBeams.push({ score, tokens });
  }

  getBestFinished(): { score: number; tokens: number[] }[] {
    this.finishedBeams.sort((a, b) => b.score - a.score);
    return this.finishedBeams.slice(0, this.numReturnSequences);
  }
}

export class BeamSearchAlgorithm implements SearchAlgorithm {
  state: BeamSearchState;

  constructor(state: BeamSearchState) {
    this.state = state;
  }

  selectNextToken(logits: Tensor, inputIds: number[]): number {
    return 0; // Not used directly in traditional beam search loops
  }

  processLogits(nextTokenLogits: Tensor, beamIdx: number): { val: number; idx: number }[] {
    if (!(nextTokenLogits.data instanceof Float32Array)) {
      return [];
    }

    const vocabSize = nextTokenLogits.shape[nextTokenLogits.shape.length - 1] as number;
    const data = nextTokenLogits.data;
    const offset = data.length - vocabSize;

    const vals: { val: number; idx: number }[] = [];
    for (let i = 0; i < vocabSize; i++) {
      vals.push({ val: data[offset + i]!, idx: i });
    }

    vals.sort((a, b) => b.val - a.val);
    return vals.slice(0, this.state.numBeams);
  }

  pruneAndSortBeams(candidates: { score: number; tokens: number[] }[]): void {
    candidates.sort((a, b) => b.score - a.score);
    this.state.activeBeams = candidates.slice(0, this.state.numBeams);
  }
}
