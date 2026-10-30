import { Graph, Node } from '@onnx9000/core';
import { describe, expect, it } from 'vitest';
import { DagreLayoutEngine } from '../src/render/layout.js';

describe('DagreLayoutEngine', () => {
  it('should compute layout', () => {
    const g = new Graph('test');
    const n = new Node('Add', [], [], {});
    n.id = '1';
    g.nodes.push(n);

    const engine = new DagreLayoutEngine();
    const l = engine.compute(g);
    expect(l.nodes.has('1')).toBe(true);

    const lgrid = engine.computeGrid(g);
    expect(lgrid.nodes.has('1')).toBe(true);
  });
});
