import { expect, test } from 'vitest';
import { Graph, Tensor } from '@onnx9000/core';
import { extractMetadata, inferArchitecture } from '../src/arch';

test('arch inference', () => {
  const arches = [
    'mistral',
    'mixtral',
    'phi2',
    'qwen2',
    'gemma',
    'starcoder',
    'falcon',
    'bloom',
    'stablelm',
    'command-r',
    'bert',
  ];
  for (const name of arches) {
    const g = new Graph(name);
    expect(inferArchitecture(g)).toBe(name);
  }

  const g = new Graph('unknown');
  expect(inferArchitecture(g)).toBe('unknown');

  expect(() => extractMetadata(g, 'invalid_arch')).toThrow(
    'Unsupported strict architecture mapping: invalid_arch',
  );
  expect(extractMetadata(g)).toEqual({});

  const gLlama = new Graph('random');
  gLlama.addTensor(new Tensor('llama.weight', [1], 1));
  expect(inferArchitecture(gLlama)).toBe('llama');

  const gMistral = new Graph('mistral');
  const metaMistral = extractMetadata(gMistral);
  expect(metaMistral['mistral.attention.sliding_window']).toBe(4096);

  const gMixtral = new Graph('mixtral');
  const metaMixtral = extractMetadata(gMixtral);
  expect(metaMixtral['mixtral.expert_count']).toBe(8);

  const gGemma = new Graph('gemma');
  const metaGemma = extractMetadata(gGemma);
  expect(metaGemma['gemma.attention.layer_norm_rms_epsilon']).toBe(1e-6);
});
