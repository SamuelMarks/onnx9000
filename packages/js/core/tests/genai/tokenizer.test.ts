import { describe, expect, it } from 'vitest';
import {
  BasicTokenizer,
  BPETokenizer,
  WordPieceTokenizer,
  UnigramTokenizer,
  HuggingFaceTokenizerLoader,
  UnicodeNormalizer,
  PreTokenizer,
} from '../../src/genai/tokenizer.js';

describe('Tokenizers', () => {
  it('BasicTokenizer', () => {
    const tokenizer = new BasicTokenizer();
    const ids = tokenizer.encode('hello');
    expect(ids).toEqual([104, 101, 108, 108, 111]);
    const text = tokenizer.decode(ids);
    expect(text).toBe('hello');
  });

  it('BPETokenizer', () => {
    const merges: [string, string][] = [
      ['h', 'e'],
      ['he', 'l'],
      ['hel', 'l'],
      ['hell', 'o'],
    ];
    const vocab = new Map([
      ['h', 0],
      ['e', 1],
      ['l', 2],
      ['o', 3],
      ['he', 4],
      ['hel', 5],
      ['hell', 6],
      ['hello', 7],
      ['<unk>', 8],
    ]);

    const tokenizer = new BPETokenizer(merges, vocab);
    const ids = tokenizer.encode('hello');
    expect(ids).toEqual([7]);

    const text = tokenizer.decode([7]);
    expect(text).toBe('hello');
  });

  it('WordPieceTokenizer', () => {
    const vocab = new Map([
      ['[UNK]', 0],
      ['un', 1],
      ['##aff', 2],
      ['##able', 3],
    ]);
    const tokenizer = new WordPieceTokenizer(vocab);
    const ids = tokenizer.encode('unaffable');
    expect(ids.length).toBeGreaterThan(0);
  });

  it('UnigramTokenizer', () => {
    const vocab = new Map([
      ['<unk>', -100.0],
      ['un', -1.0],
      ['aff', -2.0],
      ['able', -3.0],
      ['unaffable', -4.0],
    ]);
    const tokenizer = new UnigramTokenizer(vocab);
    const ids = tokenizer.encode('unaffable');
    expect(ids.length).toBe(1);

    const text = tokenizer.decode(ids);
    expect(text).toBe('unaffable');
  });

  it('HuggingFaceTokenizerLoader', () => {
    const jsonContent = `{
            "model": {
                "type": "BPE",
                "vocab": {"h": 0, "e": 1, "he": 2},
                "merges": ["h e"]
            }
        }`;

    const tokenizer = HuggingFaceTokenizerLoader.loadFromJson(jsonContent);
    expect(tokenizer instanceof BPETokenizer).toBeTruthy();
  });

  it('PreTokenizer & Normalizer', () => {
    const text = 'Hello, world!  ';
    expect(PreTokenizer.whitespaceSplit(text)).toEqual(['Hello,', ' ', 'world!', '  ']);
    expect(PreTokenizer.punctuationSplit(text)).toEqual(['Hello', ',', ' world', '!', '  ']);

    const textNFD = 'e\u0301'; // e + acute accent
    expect(textNFD.length).toBe(2);

    const textNFC = UnicodeNormalizer.normalize(textNFD, 'NFC');
    expect(textNFC.length).toBe(1);
  });
});
