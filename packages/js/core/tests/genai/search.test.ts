import { describe, expect, it } from 'vitest';
import { Tensor } from '../../src/index.js';
import { GreedySearch, MultinomialSampling } from '../../src/genai/search.js';

function createLogits(vals: number[]): Tensor {
  const data = new Float32Array(vals);
  return new Tensor('logits', [1, vals.length], 1, false, false, data);
}

describe('Search Algorithms', () => {
  it('GreedySearch', () => {
    const search = new GreedySearch();
    const logits = createLogits([1.0, 5.0, 3.0]);
    const idx = search.selectNextToken(logits, []);
    expect(idx).toBe(1);
  });

  it('MultinomialSampling', () => {
    const search = new MultinomialSampling();
    const logits = createLogits([1.0, 5.0, 3.0]);
    const idx = search.selectNextToken(logits, []);
    expect([0, 1, 2]).toContain(idx);
  });
});
