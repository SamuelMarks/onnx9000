import { describe, it, expect } from 'vitest';
import { CGenerator } from '../src/generator.js';
import { Graph, Node } from '@onnx9000/core';

describe('CGenerator edge cases', () => {
  it('handles empty outputs and missing inputs', () => {
    const graph = new Graph('Edge');
    const noOutNode = new Node('Add', ['in1', 'in2'], [], {}, 'noOut');
    const noInNode = new Node('Add', [], ['out1'], {}, 'noIn');
    graph.nodes.push(noOutNode, noInNode);

    const generator = new CGenerator(graph, '', false);
    const source = generator.generateSource();

    expect(source).not.toContain('// Add -> noOut');
    expect(source).toContain('// Add -> out1');
    expect(source).toContain('0[i]'); // uses '0' for missing inputs
  });
});
