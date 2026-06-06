import { Graph } from '@onnx9000/core';
import { describe, expect, it } from 'vitest';
import { computeLayout } from '../../src/layout/dag.ts';

describe('computeLayout', () => {
  it('should compute TB layout for empty graph', () => {
    const graph = new Graph('test');
    const layout = computeLayout(graph, 'TB');
    expect(layout.nodes).toEqual([]);
    expect(layout.edges).toEqual([]);
    expect(layout.groups).toEqual([]);
    expect(layout.width).toBe(0);
    expect(layout.height).toBe(170); // based on maxLevel logic for empty graph
  });

  it('should compute LR layout for simple graph', () => {
    const graph = new Graph('test');
    graph.inputs = [{ name: 'input1', dtype: 'float32', shape: [1, 3, 224, 224] }];
    graph.outputs = [{ name: 'output1', dtype: 'float32', shape: [1, 1000] }];
    graph.addNode({
      id: 'node1',
      opType: 'Relu',
      name: 'relu1',
      inputs: ['input1'],
      outputs: ['output1'],
      attributes: {},
      domain: '',
    });
    const layout = computeLayout(graph, 'LR');
    expect(layout.nodes.length).toBe(3); // input1, node1, output1
    expect(layout.edges.length).toBe(2);
  });
});
