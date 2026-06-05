import { describe, it, expect } from 'vitest';
import {
  BasicTokenizer,
  BPETokenizer,
  WordPieceTokenizer,
  HuggingFaceTokenizerLoader,
} from '../src/genai/tokenizer.js';

describe('tokenizers', () => {
  it('should basic tokenize', () => {
    const t = new BasicTokenizer();
    const ids = t.encode('abc');
    expect(ids.length).toBe(3);
    expect(t.decode(ids)).toBe('abc');
  });

  it('should bpe tokenize', () => {
    const t = new BPETokenizer(
      [['a', 'b']],
      new Map([
        ['a', 1],
        ['b', 2],
        ['ab', 3],
      ]),
    );
    const ids = t.encode('a b ab');
    expect(ids).toBeDefined();
  });

  it('should load hf tokenizer', () => {
    const t = HuggingFaceTokenizerLoader.loadFromJson('{"model": {"type": "BPE"}}');
    expect(t).toBeDefined();
  });
});
