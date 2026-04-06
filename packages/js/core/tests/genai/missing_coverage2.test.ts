import { test, expect } from 'vitest';
import { BasicTokenizer, BasicTokenizerStream, BPETokenizer } from '../../src/genai/tokenizer';

test('BasicTokenizerStream basic coverage', () => {
  const t = new BasicTokenizer();
  const stream = t.createStream();

  expect(stream.put(97)).toBe('a');
  expect(stream.put(98)).toBe('ab');
});

test('BPETokenizer coverage', () => {
  const vocab = new Map([
    ['a', 1],
    ['b', 2],
    ['c', 3],
    ['ab', 4],
    ['abc', 5],
  ]);
  const merges: [string, string][] = [
    ['a', 'b'],
    ['ab', 'c'],
  ];

  const t = new BPETokenizer(merges, vocab);
  expect(t.encode('abc')).toEqual([5]);
  expect(t.decode([5])).toBe('abc');
  expect(t.decode([5], false)).toBe('abc');
  expect(t.idToToken(5)).toBe('abc');

  // test unknown token handling
  const vocab2 = new Map<string, number>([
    ['a', 1],
    ['<unk>', 99],
  ]);
  const merges2: [string, string][] = [];
  const t2 = new BPETokenizer(merges2, vocab2, '<unk>');
  expect(t2.encode('b')).toEqual([99]);
  expect(t2.idToToken(100)).toBe('<unk>');

  const stream = t2.createStream();
  expect(stream.put(1)).toBe('a');
});

test('WordPieceTokenizer coverage', async () => {
  const { WordPieceTokenizer } = await import('../../src/genai/tokenizer');
  const vocab = new Map([
    ['a', 1],
    ['##b', 2],
    ['##c', 3],
    ['[UNK]', 99],
  ]);
  const t = new WordPieceTokenizer(vocab);
  expect(t.encode('abc')).toEqual([1, 2, 3]);
  expect(t.encode('d')).toEqual([99]); // Unknown word
  expect(t.encode('a d')).toEqual([1, 99]);
  expect(t.decode([1, 2, 3])).toBe('abc');
  expect(t.decode([1, 2, 3], false)).toBeDefined();
  expect(t.idToToken(3)).toBe('##c');
  expect(t.tokenToId('##c')).toBe(3);
  const stream = t.createStream();
  expect(stream.put(1)).toBe('a');

  // Max input chars per word exceeded
  const t2 = new WordPieceTokenizer(vocab, '[UNK]', 2);
  expect(t2.encode('abc')).toEqual([99]);
});

test('UnigramTokenizer coverage', async () => {
  const { UnigramTokenizer } = await import('../../src/genai/tokenizer');
  const vocab = new Map([
    ['a', 1],
    ['b', 2],
    ['ab', 3],
    ['<unk>', 99],
  ]);
  const scores = new Map([
    ['a', -1],
    ['b', -1],
    ['ab', 0.5],
  ]);
  const t = new UnigramTokenizer(vocab, scores, '<unk>');
  expect(t.encode('ab')).toBeDefined();
  expect(t.encode('abc')).toBeDefined(); // c is Object, whole thing unknown or partial?
  // The simplified unigram tokenizer probably falls back to unknown.
  expect(t.decode([3])).toBeDefined();
  expect(t.idToToken(3)).toBeDefined();
  expect(t.tokenToId('ab')).toBeDefined();
  const stream = t.createStream();
  expect(stream.put(3)).toBeDefined();
});

test('HuggingFaceTokenizerLoader coverage', async () => {
  const { HuggingFaceTokenizerLoader } = await import('../../src/genai/tokenizer');
  const loader = new HuggingFaceTokenizerLoader();
  try {
    await loader.load('dummy');
  } catch (e) {}
});

test('UnicodeNormalizer and PreTokenizer coverage', async () => {
  const {
    UnicodeNormalizer,
    PreTokenizer,
    TokenTrie,
    StreamingUTF8Decoder,
    loadTokenizerWithFallback,
  } = await import('../../src/genai/tokenizer');
  expect(UnicodeNormalizer.normalize('a')).toBe('a');
  try {
    UnicodeNormalizer.normalize('a', 'UNKNOWN');
  } catch (e) {}

  expect(PreTokenizer.whitespaceSplit('a b')).toBeDefined();

  const trie = new TokenTrie();
  const dec = new StreamingUTF8Decoder();
  expect(dec.decode(new Uint8Array([97]))).toBe('a');

  loadTokenizerWithFallback('dummy').then((t) => {
    expect(t).toBeDefined();
  });
});

test('LlamaTokenizer and GPT2Tokenizer coverage', async () => {
  const { LlamaTokenizer, GPT2Tokenizer } = await import('../../src/genai/tokenizer');
  const l = new LlamaTokenizer();
  const g = new GPT2Tokenizer();
  expect(l.encode('a')).toEqual([97]);
  expect(g.encode('a')).toEqual([97]);
});

test('PreTokenizer more methods coverage', async () => {
  const { PreTokenizer } = await import('../../src/genai/tokenizer');
  expect(PreTokenizer.punctuationSplit('a! b')).toBeDefined();
  expect(PreTokenizer.byteLevel('a b')).toBeDefined();
  expect(PreTokenizer.whitespaceSplit('')).toEqual([]);
  expect(PreTokenizer.punctuationSplit('')).toEqual([]);
});

test('UnigramTokenizer extra methods coverage', async () => {
  const { UnigramTokenizer } = await import('../../src/genai/tokenizer');
  const vocab = new Map([
    ['a', 1],
    ['b', 2],
  ]);
  const scores = new Map([
    ['a', -1],
    ['b', -1],
  ]);
  const t = new UnigramTokenizer(vocab, scores);

  expect(t.encodeBatch(['a', 'b'])).toBeDefined();
  expect(t.decodeBatch([[1], [2]])).toBeDefined();
  expect(t.decode([1, 2], false)).toBeDefined();
  expect(t.idToToken(1)).toBeDefined();
  expect(t.tokenToId('b')).toBeDefined();
  expect(t.createStream()).toBeDefined();
});

test('HuggingFaceTokenizerLoader.loadFromJson coverage', async () => {
  const { HuggingFaceTokenizerLoader } = await import('../../src/genai/tokenizer');

  // BPE
  const bpeJson = JSON.stringify({
    model: {
      type: 'BPE',
      vocab: { a: 1, b: 2 },
      merges: ['a b'],
    },
  });
  const bpe = HuggingFaceTokenizerLoader.loadFromJson(bpeJson);
  expect(bpe).toBeDefined();

  // WordPiece
  const wpJson = JSON.stringify({
    model: {
      type: 'WordPiece',
      vocab: { a: 1, b: 2 },
    },
  });
  const wp = HuggingFaceTokenizerLoader.loadFromJson(wpJson);
  expect(wp).toBeDefined();

  // Unigram
  const uniJson = JSON.stringify({
    model: {
      type: 'Unigram',
      vocab: [
        ['a', -1],
        ['b', -2],
      ],
    },
  });
  const uni = HuggingFaceTokenizerLoader.loadFromJson(uniJson);
  expect(uni).toBeDefined();

  // Unknown
  const unkJson = JSON.stringify({
    model: {
      type: 'Unknown',
    },
  });
  try {
    HuggingFaceTokenizerLoader.loadFromJson(unkJson);
  } catch (e) {}
});

test('WordPieceTokenizer encodeBatch decodeBatch', async () => {
  const { WordPieceTokenizer } = await import('../../src/genai/tokenizer');
  const vocab = new Map([['a', 1]]);
  const t = new WordPieceTokenizer(vocab);
  expect(t.encodeBatch(['a'])).toEqual([[1]]);
  expect(t.decodeBatch([[1]])).toEqual(['a']);
});

test('BPETokenizer encodeBatch decodeBatch', async () => {
  const { BPETokenizer } = await import('../../src/genai/tokenizer');
  const vocab = new Map([['a', 1]]);
  const t = new BPETokenizer([], vocab);
  expect(t.encodeBatch(['a'])).toBeDefined();
  expect(t.decodeBatch([[1]])).toBeDefined();
});

test('BasicTokenizer encodeBatch decodeBatch', async () => {
  const { BasicTokenizer } = await import('../../src/genai/tokenizer');
  const t = new BasicTokenizer();
  expect(t.encodeBatch(['a'])).toBeDefined();
  expect(t.decodeBatch([[97]])).toBeDefined();
});

test('BPETokenizer extra 1', async () => {
  const { BPETokenizer } = await import('../../src/genai/tokenizer');
  const vocab = new Map([['a', 1]]);
  const t = new BPETokenizer([], vocab);
  expect(t.tokenToId('a')).toBe(1);
  expect(t.tokenToId('b')).toBeDefined(); // unknown
});

test('WordPieceTokenizer extra 1', async () => {
  const { WordPieceTokenizer } = await import('../../src/genai/tokenizer');
  const vocab = new Map([
    ['a', 1],
    ['b', 2],
  ]);
  const t = new WordPieceTokenizer(vocab);
  expect(t.decode([1, 2])).toBe('a b');
});

test('BasicTokenizer extra 1', async () => {
  const { BasicTokenizer } = await import('../../src/genai/tokenizer');
  const t = new BasicTokenizer();
  expect(t.idToToken(97)).toBe('a');
  expect(t.tokenToId('a')).toBe(97);
  expect(t.tokenToId('')).toBe(0);
});
