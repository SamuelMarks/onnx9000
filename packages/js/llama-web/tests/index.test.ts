import { describe, expect, it } from 'vitest';
import { LlamaWeb } from '../src/index.js';

describe('LlamaWeb', () => {
  it('should run session and compute KV-cache bytes', () => {
    const l = new LlamaWeb();
    expect(l.run('llama-3-8b')).toBe('[LLaMA-Web] processing llama-3-8b');
    expect(() => l.run('')).toThrow('Invalid model string');

    const custom = new LlamaWeb({ numLayers: 2, numKvHeads: 4, headDim: 64, maxSeqLen: 128 });
    expect(custom.computeKvCacheBytes(10, 2)).toBe(2 * 2 * 4 * 64 * 10 * 2);
    expect(custom.computeKvCacheBytes(200)).toBe(2 * 2 * 4 * 64 * 128 * 2);
  });
});
