import { expect, test } from 'vitest';
import { Graph, Tensor } from '@onnx9000/core';
import { compileGGUF } from '../src/compiler';

test('compileGGUF basic', () => {
  const g = new Graph('test');
  g.producerName = 'test_producer';
  g.modelVersion = 1;

  const data1 = new Float32Array([1, 2, 3, 4]);
  const data2 = new Uint16Array([1, 2]); // F16 mocked

  const t1 = new Tensor('model.layers.0.weight', [2, 2], 'float32');
  t1.data = new Uint8Array(data1.buffer);
  g.addTensor(t1);
  g.initializers.push('model.layers.0.weight');

  const t2 = new Tensor('model.layers.0.bias', [2], 'float16');
  t2.data = new Uint8Array(data2.buffer);
  g.addTensor(t2);
  g.initializers.push('model.layers.0.bias');

  const buf = compileGGUF(g);
  expect(buf.byteLength).toBeGreaterThan(0);
});

test('compileGGUF overrides', () => {
  const g = new Graph('llama');
  g.addTensor(new Tensor('x', [2], 'float16'));
  g.initializers.push('x');
  const buf = compileGGUF(g, {
    'custom.bool': true,
    'custom.float': 1.0,
    'custom.int': 1,
    'custom.str': 'str',
    'general.alignment': 64,
  });
  expect(buf.byteLength).toBeGreaterThan(0);
});
