import { Graph, Node, ValueInfo } from '@onnx9000/core';
import { describe, expect, it } from 'vitest';
import { emitWGSL } from '../src/wgsl/index.js';

describe('wgsl index', () => {
  it('should generate wgsl', () => {
    const graph = new Graph('test');
    graph.inputs.push(new ValueInfo('in1', [1], 'float32'), new ValueInfo('in2', [1], 'float32'));
    graph.outputs.push('out' as never as ValueInfo);
    graph.nodes.push(new Node('Add', ['in1', 'in2'], ['out']));

    const code = emitWGSL(graph);
    expect(code).toContain('@compute');
    expect(code).toContain('in1[i] + in2[i]');
  });

  it('should throw on empty graph', () => {
    expect(() => emitWGSL(new Graph('empty'))).toThrow('Graph is empty');
  });
});
