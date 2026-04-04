import { expect, test } from 'vitest';
import { PyTorchFXParser, JAXprParser, XLAHLOParser } from '../src/parsers.js';
import { Graph, Tensor } from '@onnx9000/core';

test('PyTorchFXParser', () => {
  const parser = new PyTorchFXParser();
  const g = parser.parse(null);
  expect(g).toBeInstanceOf(Graph);
  expect(g.name).toBe('PyTorch_Exported');

  const add = (parser as any).atenToIr['aten.add.Tensor'];
  const matmul = (parser as any).atenToIr['aten.mm.default'];
  const t = new Tensor('in', [1], 'float32', false, false, new Float32Array([1]));
  expect(add(t).name).toBe('Add_out');
  expect(matmul(t).name).toBe('MatMul_out');
});

test('JAXprParser', () => {
  const parser = new JAXprParser();
  const g = parser.parse(null);
  expect(g).toBeInstanceOf(Graph);
  expect(g.name).toBe('JAX_Exported');
});

test('XLAHLOParser', () => {
  const parser = new XLAHLOParser();
  const g = parser.parse(null);
  expect(g).toBeInstanceOf(Graph);
  expect(g.name).toBe('XLA_Exported');
});
