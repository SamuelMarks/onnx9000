import { describe, expect, it } from 'vitest';
import { Tensor } from '../../src/index.js';
import { TopKLogitProcessor, ForcedBOSLogitProcessor } from '../../src/genai/logit_processors.js';

function createLogits(vals: number[]): Tensor {
  const data = new Float32Array(vals);
  return new Tensor('logits', [1, vals.length], 1, false, false, data);
}

describe('TopKLogitProcessor Coverage', () => {
  it('should return original logits if not Float32Array', () => {
    const p = new TopKLogitProcessor(10);
    const tensor = new Tensor('logits', [1, 2], 1, false, false, new Int32Array([1, 2]));
    const res = p.process([], tensor);
    expect(res).toBe(tensor);
  });

  it('should return original logits if topK >= vocab size', () => {
    const p = new TopKLogitProcessor(10);
    const tensor = createLogits([1.0, 2.0, 3.0]);
    const res = p.process([], tensor);
    expect(res).toBe(tensor);
  });
});

describe('ForcedBOSLogitProcessor Coverage', () => {
  it('should return original logits if inputIds is not empty', () => {
    const p = new ForcedBOSLogitProcessor(1);
    const tensor = createLogits([1.0, 2.0]);
    const res = p.process([1], tensor);
    expect(res).toBe(tensor);
  });

  it('should return original logits if not Float32Array', () => {
    const p = new ForcedBOSLogitProcessor(1);
    const tensor = new Tensor('logits', [1, 2], 1, false, false, new Int32Array([1, 2]));
    const res = p.process([], tensor);
    expect(res).toBe(tensor);
  });
});

import { BeamSearchState } from '../../src/genai/search.js';
describe('BeamSearchState Coverage', () => {
  it('should add finished and return best', () => {
    const s = new BeamSearchState(1, 2);
    s.addFinished(0.5, [1, 2]);
    s.addFinished(0.9, [3, 4]);
    s.addFinished(0.1, [5]);
    const best = s.getBestFinished();
    expect(best.length).toBe(2);
    expect(best[0].score).toBe(0.9);
    expect(best[1].score).toBe(0.5);
  });
});

import { MultiHeadAttentionCache } from '../../src/genai/state.js';
describe('MultiHeadAttentionCache Coverage', () => {
  it('should return null for non-existent layer', () => {
    const cache = new MultiHeadAttentionCache(2, 64);
    expect(cache.get(99)).toBeNull();
  });
});

import { HuggingFaceTokenizerLoader } from '../../src/genai/tokenizer.js';
describe('HuggingFaceTokenizerLoader Coverage', () => {
  it('should load BPE with empty model/vocab', () => {
    const json = JSON.stringify({
      model: { type: 'BPE' }, // missing vocab, merges, unk_token
    });
    const t = HuggingFaceTokenizerLoader.loadFromJson(json);
    expect(t).toBeDefined();
  });
});
