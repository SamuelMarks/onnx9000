import { Graph, Tensor } from '@onnx9000/core';
import { describe, expect, it } from 'vitest';
import { extractLlamaMetadata } from '../src/llama.js';

describe('llama', () => {
  it('should extract metadata', () => {
    const g = new Graph('llama');
    g.tensors['embed_tokens.weight'] = new Tensor('embed_tokens.weight', [100, 100], 'float32');
    const meta = extractLlamaMetadata(g);
    expect(meta['llama.vocab_size']).toBe(100);
  });
});
