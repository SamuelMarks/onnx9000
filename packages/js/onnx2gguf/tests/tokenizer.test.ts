import { expect, test } from 'vitest';
import { extractTokenizerMetadata } from '../src/tokenizer';

test('extractTokenizerMetadata', () => {
  const meta = extractTokenizerMetadata();
  expect(meta['tokenizer.ggml.model']).toBe('llama');
  expect(meta['tokenizer.ggml.tokens'].length).toBe(2);

  const meta2 = extractTokenizerMetadata(null, 5);
  expect(meta2['tokenizer.ggml.tokens'].length).toBe(5);

  const jsonStr = JSON.stringify({
    model: { type: 'BPE', vocab: { a: 0, b: 1 }, merges: ['a b'] },
    added_tokens: [{ id: 0 }],
    chat_template: 'hello',
  });
  const meta3 = extractTokenizerMetadata(jsonStr);
  expect(meta3['tokenizer.ggml.model']).toBe('gpt2');
  expect(meta3['tokenizer.ggml.tokens']).toEqual(['a', 'b']);
  expect(meta3['tokenizer.ggml.merges']).toEqual(['a b']);
  expect(meta3['tokenizer.chat_template']).toBe('hello');

  const meta4 = extractTokenizerMetadata(jsonStr, 5);
  expect(meta4['tokenizer.ggml.tokens'].length).toBe(5);

  const meta5 = extractTokenizerMetadata(jsonStr, 1);
  expect(meta5['tokenizer.ggml.tokens'].length).toBe(1);

  const jsonStrUnigram = JSON.stringify({ model: { type: 'Unigram', vocab: { a: 0, b: 1 } } });
  const meta6 = extractTokenizerMetadata(jsonStrUnigram);
  expect(meta6['tokenizer.ggml.model']).toBe('llama');

  const meta7 = extractTokenizerMetadata('{invalid');
  expect(meta7['tokenizer.ggml.model']).toBe('llama');

  const jsonStrOther = JSON.stringify({ model: { type: 'WordPiece', vocab: { a: 0, b: 1 } } });
  const meta8 = extractTokenizerMetadata(jsonStrOther);
  expect(meta8['tokenizer.ggml.model']).toBe('llama');
});
