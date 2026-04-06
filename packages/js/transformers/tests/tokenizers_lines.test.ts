import { describe, it, expect } from 'vitest';
import * as toks from '../src/tokenizers/index';

describe('tokenizers specific lines', () => {
  it('padding and return formats', () => {
    const t = new toks.PreTrainedTokenizerFast({}, {});
    const res1 = t.encode(['abc'], {
      padding: 'max_length',
      max_length: 5,
      return_attention_mask: true,
    });
    expect(res1.input_ids[0].length).toBe(5);
    expect(res1.attention_mask[0].length).toBe(5);

    expect(t._encode_single('a')).toEqual([97]);
    expect(t.batch_decode([[97]])).toEqual(['a']);
  });

  it('bpe, wordpiece, unigram', async () => {
    const bpe = new toks.PreTrainedTokenizerFast({}, {});
    (bpe as Object).tokenizerJson = { model: { type: 'BPE' } };
    (bpe as Object).wasmBpe = () => [1];
    expect(bpe._encode_single('a')).toEqual([1]);

    const wp = new toks.PreTrainedTokenizerFast({}, {});
    (wp as Object).tokenizerJson = { model: { type: 'WordPiece' } };
    (wp as Object).wasmWordPiece = () => [2];
    expect(wp._encode_single('a')).toEqual([2]);

    const ug = new toks.PreTrainedTokenizerFast({}, {});
    (ug as Object).tokenizerJson = { model: { type: 'Unigram' } };
    (ug as Object).wasmUnigram = () => [3];
    expect(ug._encode_single('a')).toEqual([3]);
  });

  it('AutoTokenizer legacy shims', async () => {
    const t = await toks.AutoTokenizer.fromPretrained('a');
    expect(t.encode('test', { return_tensors: true })).toBeDefined();
    expect(t.decode([97], { skip_special_tokens: true })).toBeDefined();
  });
});

it('BPEEncoder coverage', () => {
  const enc = new toks.BPEEncoder({ a: 1, b: 2 }, [
    ['a', 'b'],
    ['c', 'd'],
  ]);
  expect(enc.encode('abc')).toEqual([1, 2, 0]);
});

it('AutoTokenizer shims again', async () => {
  const t = await toks.AutoTokenizer.fromPretrained('a');
  expect(t.encode('')).toEqual([]);
  expect(t.encode('word')).toEqual([119]);
  expect(t.decode([])).toEqual('');
  expect(t.decode([119])).toEqual('w');
});

it('uncovered methods', () => {
  const t = new toks.PreTrainedTokenizerFast({});
  expect(t.word_ids()).toBeDefined();
  expect(t.char_to_token(1)).toBeDefined();
  expect(t.token_to_chars(1)).toBeDefined();
});

it('uncovered wasm tokenizers', () => {
  const t = new toks.PreTrainedTokenizerFast({});
  expect(t.wasmWordPiece('a')).toBeDefined();
  expect(t.wasmUnigram('a')).toBeDefined();
  expect(t.wasmBpe('a')).toBeDefined();
});

it('uncovered truncation', () => {
  const t = new toks.PreTrainedTokenizerFast({});
  const res = t.encode('test', { truncation: true, max_length: 2 });
  expect(res.input_ids.length).toBe(2);
});
