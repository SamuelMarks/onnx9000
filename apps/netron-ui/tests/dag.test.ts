import { describe, it, expect } from 'vitest';
import { computeLayout } from '../src/layout/dag.js';
import { Graph } from '@onnx9000/core';

describe('netron-ui dag layout', () => {
  it('should compute basic layout', () => {
    const g = new Graph('test');
    g.inputs.push({ name: 'in', shape: [1], dtype: 'float32' });
    g.outputs.push({ name: 'out', shape: [1], dtype: 'float32' });
    g.nodes.push({
      id: 'n1',
      opType: 'Identity',
      name: 'id',
      inputs: ['in'],
      outputs: ['out'],
      attributes: {},
    } as any);

    const layout = computeLayout(g, 'TB');
    expect(layout.nodes.length).toBeGreaterThan(0);
    expect(layout.edges.length).toBeGreaterThan(0);
  });
});
