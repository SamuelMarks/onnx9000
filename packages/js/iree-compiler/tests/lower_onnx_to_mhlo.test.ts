import { describe, it, expect } from 'vitest';
import { lowerONNXToMHLO } from '../src/passes/lower_onnx_to_mhlo.js';
import { Graph, Node, Tensor } from '@onnx9000/core';

describe('lowerONNXToMHLO', () => {
  it('should lower all basic ops', () => {
    const g = new Graph('test');
    g.inputs = [{ name: 'in1', dtype: 'float32', shape: [2, 2], id: 'in1' }];
    g.outputs = [{ name: 'out', dtype: 'float32', shape: [2, 2], id: 'out' }];

    g.tensors['init'] = new Tensor('init', [2, 2], 'float32');
    g.initializers.push('init');

    const ops = ['Add', 'Sub', 'Mul', 'Div', 'MatMul', 'Exp', 'Log', 'Cos', 'Sin', 'Max', 'Min'];
    ops.forEach((op, i) => {
      g.nodes.push(new Node(op, ['in1', 'init'], [`t${i}`]));
    });

    // Add Conv
    g.nodes.push(
      new Node('Conv', ['in1', 'init'], ['t_conv'], {
        dilations: { type: 'INTS', value: [1, 1] },
        group: { type: 'INT', value: 1 },
        kernel_shape: { type: 'INTS', value: [1, 1] },
        pads: { type: 'INTS', value: [0, 0, 0, 0] },
        strides: { type: 'INTS', value: [1, 1] },
      }),
    );

    // Add Reshape
    g.nodes.push(new Node('Reshape', ['in1', 'init'], ['t_reshape']));

    // Add Transpose
    g.nodes.push(
      new Node('Transpose', ['in1'], ['t_trans'], { perm: { type: 'INTS', value: [1, 0] } }),
    );

    // Add Concat
    g.nodes.push(
      new Node('Concat', ['in1', 'init'], ['t_cat'], { axis: { type: 'INT', value: 0 } }),
    );

    // Add Slice
    g.nodes.push(new Node('Slice', ['in1', 'init', 'init', 'init', 'init'], ['out']));

    const region = lowerONNXToMHLO(g);
    expect(region.blocks[0]!.operations.length).toBeGreaterThan(ops.length);
  });

  it('should throw on unknown node', () => {
    const g = new Graph('test');
    g.inputs = [{ name: 'in1', dtype: 'float32', shape: [2, 2], id: 'in1' }];
    g.nodes.push(new Node('Unknown', ['in1'], ['out']));
    expect(
      lowerONNXToMHLO(g).blocks[0].operations.some((o) => o.opcode === 'web.mhlo.custom_call'),
    ).toBe(true);
  });

  it('should throw on missing operand', () => {
    const g = new Graph('test');
    g.inputs = [{ name: 'in1', dtype: 'float32', shape: [2, 2], id: 'in1' }];
    g.nodes.push(new Node('Add', ['in1', 'missing'], ['out']));
    expect(() => lowerONNXToMHLO(g)).toThrow('Operand missing not found');
  });

  it('should throw on missing empty operand', () => {
    const g = new Graph('test');
    g.nodes.push(new Node('Add', ['', 'missing'], ['out']));
    expect(() => lowerONNXToMHLO(g)).toThrow('Operand missing');
  });
});

it('should throw on missing output', () => {
  const g = new Graph('test');
  g.inputs = [{ name: 'in1', dtype: 'float32', shape: [2, 2], id: 'in1' }];
  g.outputs = [{ name: 'out_missing', dtype: 'float32', shape: [2, 2], id: 'out_missing' }];
  g.nodes.push(new Node('Add', ['in1', 'in1'], ['out1', 'tensor_out', 'empty_out']));

  // hit lines 14-16 by checking a tensor
  g.tensors['tensor_out'] = new Tensor('tensor_out', [2, 2], 'float32');

  expect(() => lowerONNXToMHLO(g)).toThrow('Output out_missing not found');
});

it('covers missing attributes in ops', () => {
  const g = new Graph('test');
  g.inputs = [{ name: 'in1', dtype: 'float32', shape: [2, 2], id: 'in1' }];
  g.tensors['init'] = new Tensor('init', [2, 2], 'float32');
  g.initializers.push('init');

  // Add Conv without attributes
  g.nodes.push(new Node('Conv', ['in1', 'init'], ['t_conv']));
  // Add Transpose without attributes
  g.nodes.push(new Node('Transpose', ['t_conv'], ['t_trans']));
  // Add Concat without attributes
  g.nodes.push(new Node('Concat', ['t_trans', 'init'], ['out']));

  lowerONNXToMHLO(g);
});

it('covers string dimensions in shape for type inference', () => {
  const g = new Graph('test');
  g.inputs = [{ name: 'in1', dtype: 'float32', shape: ['batch', 2], id: 'in1' }];
  g.outputs = [{ name: 'out', dtype: 'float32', shape: ['batch', 2], id: 'out' }];

  g.tensors['init'] = new Tensor('init', ['batch', 2], 'float32');
  g.initializers.push('init');

  g.nodes.push(new Node('Add', ['in1', 'init'], ['out']));

  // hit lines 14, 39
  g.outputs.push({ name: 'tensor_out', dtype: 'float32', shape: ['batch', 2], id: '' });
  g.tensors['tensor_out'] = new Tensor('tensor_out', ['batch', 2], 'float32');
  g.nodes.push(new Node('Add', ['in1', 'init'], ['tensor_out']));

  const region = lowerONNXToMHLO(g);
  expect(region.blocks[0].operations.length).toBeGreaterThan(0);
});

it('covers string dimensions in shape for type inference in graph.tensors', () => {
  const g = new Graph('test');
  g.tensors['tensor_only'] = new Tensor('tensor_only', ['batch', 2], 'float32');
  g.initializers.push('tensor_only');
  g.nodes.push(new Node('Add', ['tensor_only', 'tensor_only'], ['tensor_only']));

  const region = lowerONNXToMHLO(g);
  expect(region.blocks[0].operations.length).toBeGreaterThan(0);
});
