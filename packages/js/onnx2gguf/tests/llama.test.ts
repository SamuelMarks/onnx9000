import { expect, test } from 'vitest';
import { Graph, Tensor, Node } from '@onnx9000/core';
import { extractLlamaMetadata } from '../src/llama';

test('extractLlamaMetadata', () => {
  const g = new Graph('llama');
  g.addTensor(new Tensor('model.embed_tokens.weight', [32000, 4096], 'float32'));
  g.addTensor(new Tensor('model.layers.0.self_attn.q_proj.weight', [4096, 4096], 'float32'));
  g.addTensor(new Tensor('model.layers.0.self_attn.k_proj.weight', [1024, 4096], 'float32'));
  g.addTensor(new Tensor('model.layers.0.mlp.up_proj.weight', [11008, 4096], 'float32'));

  const n1 = new Node('Silu', [], [], 'silu');
  g.addNode(n1);
  const n2 = new Node('RMSNormalization', [], [], 'rms');
  n2.attributes = { epsilon: 1e-6 };
  g.addNode(n2);

  const meta = extractLlamaMetadata(g);
  expect(meta['llama.embedding_length']).toBe(4096);
  expect(meta['llama.attention.head_count']).toBe(32);
  expect(meta['llama.attention.head_count_kv']).toBe(8);
  expect(meta['llama.feed_forward_length']).toBe(11008);
  expect(meta['llama.vocab_size']).toBe(32000);
  expect(meta['llama.block_count']).toBe(1);
  expect(meta['custom.is_swiglu']).toBe(true);
  expect(meta['llama.attention.layer_norm_rms_epsilon']).toBe(1e-6);
});
