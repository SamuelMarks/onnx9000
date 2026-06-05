import { describe, it, expect } from 'vitest';
import { Graph, Node, ValueInfo, Attribute } from '@onnx9000/core';
import { generateTriton } from '../src/triton/ast.js';

describe('Triton Generator Final 2', () => {
  it('should hit more op branches', () => {
    const g = new Graph('g');
    g.inputs.push(new ValueInfo('x', [1], 'float32'));
    g.outputs.push(new ValueInfo('y', [1], 'float32'));

    // BitShift RIGHT
    const n1 = new Node('BitShift', ['x', 'x'], ['y1']);
    n1.attributes['direction'] = new Attribute('direction', 'STRING', 'RIGHT');
    g.nodes.push(n1);

    // Clip with min/max
    const n2 = new Node('Clip', ['x', 'min', 'max'], ['y2']);
    g.nodes.push(n2);

    // Softplus
    g.nodes.push(new Node('Softplus', ['x'], ['y3']));

    // GatherElements
    g.nodes.push(new Node('GatherElements', ['x', 'x'], ['y4']));

    // QuantizeLinear
    g.nodes.push(new Node('QuantizeLinear', ['x', 'scale', 'zp'], ['y5']));

    // Placeholder
    g.nodes.push(new Node('Placeholder', [], ['y6']));

    // Identity, Expand, Transpose
    g.nodes.push(new Node('Identity', ['x'], ['y7']));
    g.nodes.push(new Node('Expand', ['x'], ['y8']));
    g.nodes.push(new Node('Transpose', ['x'], ['y9']));

    const code = generateTriton(g);
    expect(code).toContain('>>');
    expect(code).toContain('tl.log(1.0 + tl.exp');
    expect(code).toContain('tl.trans');
  });

  it('should handle string and sequence outputs with warnings', () => {
    const g = new Graph('g');
    g.inputs.push(new ValueInfo('x', [1], 'float32'));
    g.outputs.push(new ValueInfo('s', [1], 'string'));
    g.outputs.push(new ValueInfo('seq', [1], 'sequence' as any));

    const code = generateTriton(g);
    expect(code).toContain('WARNING: String outputs are unsupported');
    expect(code).toContain('WARNING: Sequence outputs are unsupported');
  });
});
