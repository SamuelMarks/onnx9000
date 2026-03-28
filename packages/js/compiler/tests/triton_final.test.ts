import { describe, it, expect } from 'vitest';
import { Graph, Node, ValueInfo, Attribute } from '@onnx9000/core';
import { generateTriton } from '../src/triton/ast.js';

describe('Triton Generator Final', () => {
  it('should generate Triton code for all op types and hit coverage', () => {
    const g = new Graph('test_graph');
    g.inputs.push(new ValueInfo('x', [1024, 1024], 'float32'));
    g.inputs.push(new ValueInfo('y', [1024], 'float32'));
    g.inputs.push(new ValueInfo('mask', [1], 'bool'));
    g.inputs.push(new ValueInfo('scalar', [], 'float32'));
    g.inputs.push(new ValueInfo('dynamic', ['N'], 'float32'));
    g.outputs.push(new ValueInfo('z', [1024, 1024], 'float32'));

    const ops = [
      'Add',
      'Sub',
      'Mul',
      'Div',
      'Pow',
      'Exp',
      'Log',
      'Sqrt',
      'Sin',
      'Cos',
      'Cast',
      'Sign',
      'Round',
      'IsNaN',
      'IsInf',
      'Floor',
      'Ceil',
      'Reciprocal',
      'Rsqrt',
      'MatMul',
      'Abs',
      'Max',
      'Min',
      'Where',
      'Relu',
      'Clip',
      'Tanh',
      'LeakyRelu',
      'PRelu',
      'Sigmoid',
      'Softplus',
      'Gelu',
      'ReduceSum',
      'ReduceMax',
      'ReduceMin',
      'ArgMax',
      'ArgMin',
      'Softmax',
      'LogSoftmax',
      'LayerNormalization',
      'Identity',
      'Expand',
      'Transpose',
      'Constant',
      'BitShift',
      'BitwiseAnd',
      'BitwiseOr',
      'BitwiseNot',
      'Pad',
      'Shape',
      'GatherElements',
      'QuantizeLinear',
      'Placeholder',
    ];

    for (const op of ops) {
      const node = new Node(op, ['x', 'y', 'mask'], [op + '_out']);
      if (op === 'Cast') node.attributes['to'] = new Attribute('to', 'STRING', 'float16');
      if (op === 'Constant') node.attributes['value'] = new Attribute('value', 'FLOAT', 1.0);
      if (op === 'BitShift')
        node.attributes['direction'] = new Attribute('direction', 'STRING', 'LEFT');
      g.nodes.push(node);
    }

    // Add special nodes for coverage
    g.nodes.push(new Node('SequenceAt', [], []));
    g.nodes.push(new Node('TopK', [], [], {}, 'topk_node'));

    const nodeString = new Node('StringNormalizer', [], ['s_out']);
    g.nodes.push(nodeString);

    const nodeCustom = new Node('Add', ['x', 'y'], ['custom_out'], {}, 'custom_node', 'com.custom');
    g.nodes.push(nodeCustom);

    const nodeUnk = new Node('UnknownOp', [], []);
    g.nodes.push(nodeUnk);

    const code = generateTriton(g, { headers: ['# Custom Header'] });
    expect(code).toContain('def test_graph');
    expect(code).toContain('# Custom Header');
    expect(code).toContain('BLOCK_M');
    expect(code).toContain('tl.dot');
    expect(code).toContain('tl.math.tanh');
    expect(code).toContain('WARNING: Custom domain com.custom');
    expect(code).toContain('WARNING: String tensors are unsupported');
    expect(code).toContain('WARNING: Unsupported op UnknownOp');
  });

  it('should throw on large BLOCK_M', () => {
    const g = new Graph('g');
    expect(() => generateTriton(g, { blockM: 3000 })).toThrow('BLOCK_M too large');
  });
});
