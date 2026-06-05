import { describe, it, expect } from 'vitest';
import { Graph, Node, ValueInfo } from '@onnx9000/core';
import { emitWGSL } from '../src/wgsl/index.js';

describe('WGSL Emitter Final', () => {
  it('should emit WGSL for all op types', () => {
    const g = new Graph('g');
    g.inputs.push(new ValueInfo('in1', [64], 'float32'));
    g.inputs.push(new ValueInfo('in2', [64], 'float32'));
    g.outputs.push('out' as any);

    const ops = ['Add', 'Sub', 'Mul', 'Div', 'Relu', 'Exp', 'Log', 'Sqrt', 'ReduceSum', 'MatMul'];
    for (const op of ops) {
      g.nodes.push(new Node(op, ['in1', 'in2'], ['out']));
    }

    const code = emitWGSL(g);
    expect(code).toContain('main');
    for (const op of ops) {
      if (op === 'Add') expect(code).toContain('+');
      if (op === 'Sub') expect(code).toContain('-');
      if (op === 'Mul') expect(code).toContain('*');
      if (op === 'Div') expect(code).toContain('/');
      if (op === 'Relu') expect(code).toContain('max');
      if (op === 'Exp') expect(code).toContain('exp');
      if (op === 'Log') expect(code).toContain('log');
      if (op === 'Sqrt') expect(code).toContain('sqrt');
    }
  });

  it('should throw on empty graph', () => {
    const g = new Graph('empty');
    expect(() => emitWGSL(g)).toThrow('Graph is empty');
  });
});
