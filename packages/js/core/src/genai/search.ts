import { Tensor } from '../index.js';

/**
 * Interface for token search algorithms used in GenAI.
 */
export interface SearchAlgorithm {
  /**
   * Select the next token based on model logits.
   * @param logits Output logits from the model (usually shape [1, vocab_size]).
   * @param inputIds Previous token sequence.
   */
  selectNextToken(logits: Tensor, inputIds: number[]): number;
}

/**
 * Greedy search algorithm that always selects the token with highest probability.
 */
export class GreedySearch implements SearchAlgorithm {
  /**
   * Select the highest probability token.
   */
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

/**
 * Multinomial sampling algorithm that samples tokens based on their probability distribution.
 */
export class MultinomialSampling implements SearchAlgorithm {
  /**
   * Sample the next token.
   */
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

/**
 * State container for beam search execution.
 */
export class BeamSearchState {
  /** Number of beams. */
  numBeams: number;
  /** Number of best sequences to return. */
  numReturnSequences: number;
  /** Current active candidate beams. */
  activeBeams: { score: number; tokens: number[] }[];
  /** Completed sequences. */
  finishedBeams: { score: number; tokens: number[] }[];

  /**
   * Create a new BeamSearchState.
   * @param numBeams Number of beams.
   * @param numReturnSequences Number of results.
   */
  constructor(numBeams: number, numReturnSequences: number) {
    this.numBeams = numBeams;
    this.numReturnSequences = numReturnSequences;
    this.activeBeams = [];
    this.finishedBeams = [];
  }

  /**
   * Add a finished sequence to the results.
   * @param score Cumulative score.
   * @param tokens Sequence tokens.
   */
  addFinished(score: number, tokens: number[]): void {
    this.finishedBeams.push({ score, tokens });
  }

  /**
   * Get the top finished sequences.
   */
  getBestFinished(): { score: number; tokens: number[] }[] {
    this.finishedBeams.sort((a, b) => b.score - a.score);
    return this.finishedBeams.slice(0, this.numReturnSequences);
  }
}

/**
 * Beam search algorithm implementation.
 */
export class BeamSearchAlgorithm implements SearchAlgorithm {
  /** Internal state of the beam search. */
  state: BeamSearchState;

  /**
   * Create a new BeamSearchAlgorithm.
   * @param state Beam search state.
   */
  constructor(state: BeamSearchState) {
    this.state = state;
  }

  /**
   * Dummy implementation for interface compliance.
   */
  selectNextToken(logits: Tensor, inputIds: number[]): number {
    return 0; // Not used directly in traditional beam search loops
  }

  /**
   * Process logits to find top-K candidates for a beam.
   * @param nextTokenLogits Logits for next token.
   * @param beamIdx Beam index.
   */
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

  /**
   * Sort and prune candidate beams to maintain fixed beam width.
   * @param candidates All potential new beams.
   */
  pruneAndSortBeams(candidates: { score: number; tokens: number[] }[]): void {
    candidates.sort((a, b) => b.score - a.score);
    this.state.activeBeams = candidates.slice(0, this.state.numBeams);
  }
}
