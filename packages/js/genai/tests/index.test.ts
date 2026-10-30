import { describe, expect, it } from 'vitest';
import { Genai, GenaiState } from '../src/index.js';

describe('Genai', () => {
  it('should run default decoding step', () => {
    const genai = new Genai();
    expect(genai.run()).toBe('[genai] processed');
  });

  it('should process logits with repetition penalty and temperature', () => {
    const genai = new Genai({ temperature: 2.0, repetitionPenalty: 2.0, maxLength: 10 });
    const raw = new Float32Array([4.0, -4.0, 2.0]);
    const scaled = genai.processLogits(raw, [0, 1]);
    expect(scaled[0]).toBeCloseTo(1.0);
    expect(scaled[1]).toBeCloseTo(-4.0);
    expect(scaled[2]).toBeCloseTo(1.0);

    const zeroTemp = new Genai({ temperature: 0 });
    expect(zeroTemp.processLogits(new Float32Array([1.0]), [])[0]).toBeCloseTo(1.0);

    expect(genai.selectGreedy(new Float32Array([0.1, 0.5, 0.9, 0.2]))).toBe(2);
    expect(() => genai.selectGreedy(new Float32Array([]))).toThrow('Logits array cannot be empty');

    const state = new GenaiState();
    state.appendToken(7);
    expect(state.tokenIds).toEqual([7]);
    expect(state.pastSequenceLength).toBe(0);
  });
});
