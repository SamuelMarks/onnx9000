import { test, expect } from 'vitest';
import { GreedySearch, BeamSearchAlgorithm, MultinomialSampling } from '../../src/genai/search';
import { Tensor } from '../../src/ir/tensor';

test('GreedySearch selectNextToken coverage', () => {
  const s = new GreedySearch();
  const t = new Tensor('t', 'float32', [2]);
  t.data = new Float32Array([0.1, 0.9]);
  expect(s.selectNextToken(t, [])).toBe(1);
});

test('BeamSearchAlgorithm invalid tensor data', () => {
  const b = new BeamSearchAlgorithm(2);
  // 92-93, 96-98: beam search invalid data types / fallbacks
  // The loop of beam search with non-float32
  expect(b.selectNextToken(new Tensor('t', 'int32', [1]), [])).toBe(0);
});

test('MultinomialSampling invalid tensor data', () => {
  const m = new MultinomialSampling();
  expect(m.selectNextToken(new Tensor('t', 'int32', [1]), [])).toBe(0);
});
