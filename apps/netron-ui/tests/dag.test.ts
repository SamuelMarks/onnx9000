import { describe, it, expect } from 'vitest';
import { Graph, Node, ValueInfo, Tensor } from '@onnx9000/core';
import { computeLayout } from '../src/layout/dag';

describe('Layout computation', () => {
  it('should compute TB and LR layouts', () => {
    const g = new Graph('test');
    g.inputs.push(new ValueInfo('A', [1], 'float32'));
    g.outputs.push(new ValueInfo('B', [1], 'float32'));
    g.initializers.push('C');
    g.addTensor(new Tensor('C', [1], 'float32'));
    g.addNode(new Node('Add', ['A', 'C'], ['B'], {}, 'Node1'));

    // Just ensure it runs without error and returns something structured
    const layout1 = computeLayout(g, 'TB');
    expect(layout1.nodes.length).toBeGreaterThan(0);
    expect(layout1.edges.length).toBeGreaterThan(0);

    const layout2 = computeLayout(g, 'LR');
    expect(layout2.nodes.length).toBeGreaterThan(0);
    expect(layout2.edges.length).toBeGreaterThan(0);

    // Test missing inputs/outputs/initializers
    const g2 = new Graph('empty');
    g2.addNode(new Node('Sub', ['X'], ['Y'], {}, 'Node2'));
    const layout3 = computeLayout(g2, 'TB');
    expect(layout3.nodes.length).toBeGreaterThan(0);
  });
});
