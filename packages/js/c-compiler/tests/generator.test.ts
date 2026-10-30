import { Graph, Node, ValueInfo } from '@onnx9000/core';
import { describe, expect, it } from 'vitest';
import { CGenerator } from '../src/generator.js';

describe('CGenerator', () => {
  it('should generate source', () => {
    const graph = new Graph('test');
    graph.inputs.push(new ValueInfo('in', [10], 'float32'));
    graph.outputs.push(new ValueInfo('out', [10], 'float32'));
    graph.nodes.push(new Node('Relu', ['in'], ['out']));

    const gen = new CGenerator(graph, 'model_', false);
    const header = gen.generateHeader();
    expect(header).toContain('void model_run');

    const src = gen.generateSource();
    expect(src).toContain('Relu -> out');
    expect(src).toContain('> 0 ?');
  });
});
