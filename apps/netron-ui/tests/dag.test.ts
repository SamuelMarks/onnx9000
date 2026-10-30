import { Graph, Node } from '@onnx9000/core';
import { describe, expect, it } from 'vitest';
import { computeLayout } from '../src/layout/dag.js';

describe('netron-ui dag layout', () => {
  it('should compute basic layout', () => {
    const g = new Graph('test');
    g.inputs.push({ name: 'in', shape: [1], dtype: 'float32', id: 'in' });
    g.outputs.push({ name: 'out', shape: [1], dtype: 'float32', id: 'out' });
    g.nodes.push(new Node('Identity', ['in'], ['out'], {}, 'id'));

    const layout = computeLayout(g, 'TB');
    expect(layout.nodes.length).toBeGreaterThan(0);
    expect(layout.edges.length).toBeGreaterThan(0);
  });
});
