import { expect, test } from 'vitest';
import { PyTorchFXParser, JAXprParser, XLAHLOParser } from '../src/parsers.js';
import { Graph, Tensor } from '@onnx9000/core';

test('PyTorchFXParser', () => {
  const parser = new PyTorchFXParser();
  const g = parser.parse(null);
  expect(g).toBeInstanceOf(Graph);
  expect(g.name).toBe('PyTorch_Exported');

  const fxDict = {
    nodes: [
      {
        op: 'call_function',
        target: 'aten.add.default',
        name: 'add_1',
        args: ['a', 'b'],
        kwargs: {},
      },
    ],
  };
  const g2 = parser.parse(fxDict);
  expect(g2.nodes.length).toBe(1);
  expect(g2.nodes[0].opType).toBe('add');
  expect(g2.nodes[0].inputs).toEqual(['a', 'b']);

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

  const jaxpr = {
    invars: [{ name: 'a', shape: [10], type: 'f32' }],
    constvars: [{ name: 'c', shape: [10], type: 'i32' }],
    eqns: [
      {
        primitive: 'add',
        invars: [{ name: 'a' }, { name: 'c' }],
        outvars: [{ name: 'b', shape: [10], type: 'f32' }],
        params: { foo: 'bar' },
      },
    ],
    outvars: [{ name: 'b' }],
  };

  const g2 = parser.parse(jaxpr);
  expect(g2.inputs.length).toBe(1);
  expect(g2.initializers.length).toBe(1);
  expect(g2.outputs.length).toBe(1);
  expect(g2.nodes.length).toBe(1);
  expect(g2.nodes[0].name).toBe('add_b');
  expect(g2.tensors['a'].dtype).toBe('float32');
  expect(g2.tensors['c'].dtype).toBe('int32');
  expect(g2.tensors['b'].dtype).toBe('float32');

  const jaxpr2 = {
    invars: [{ name: 'a', shape: [10], type: 'unknown' }],
    eqns: [
      {
        primitive: 'identity',
        outvars: [],
      },
    ],
  };
  const g3 = parser.parse(jaxpr2);
  expect(g3.nodes[0].name).toBe('identity');
  expect(g3.tensors['a'].dtype).toBe('float32');
});

test('XLAHLOParser', () => {
  const parser = new XLAHLOParser();
  const g = parser.parse(null);
  expect(g).toBeInstanceOf(Graph);
  expect(g.name).toBe('XLA_Exported');
});
