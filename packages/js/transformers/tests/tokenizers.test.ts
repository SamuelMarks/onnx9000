import { describe, it, expect } from 'vitest';
import { AutoTokenizer, BPEEncoder } from '../src/tokenizers/index.js';

describe('AutoTokenizer', () => {
  it('should encode and decode', async () => {
    const tokenizer = await AutoTokenizer.fromPretrained('dummy');

    expect(tokenizer.encode('')).toEqual([]);
    expect(tokenizer.decode([])).toEqual('');

    const text = 'hello world';
    const encoded = tokenizer.encode(text);
    expect(encoded).toEqual(['hello'.charCodeAt(0), 'world'.charCodeAt(0)]);

    const decoded = tokenizer.decode(encoded);
    expect(decoded).toEqual('h w');
  });
});

describe('BPEEncoder', () => {
  it('should encode using vocab', () => {
    const encoder = new BPEEncoder({ a: 1, b: 2 }, [['a', 'b']]);
    expect(encoder.encode('aba')).toEqual([1, 2, 1]);
    expect(encoder.encode('abc')).toEqual([1, 2, 0]);
  });
});
