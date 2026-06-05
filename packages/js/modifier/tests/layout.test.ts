import { describe, it, expect } from 'vitest';
import { DagreLayoutEngine } from '../src/render/layout.js';
import { Graph } from '@onnx9000/core';

describe('DagreLayoutEngine', () => {
  it('should compute layout', () => {
    const g = new Graph('test');
    g.nodes.push({ id: '1', opType: 'Add', inputs: [], outputs: [] } as any);

    const engine = new DagreLayoutEngine();
    const l = engine.compute(g);
    expect(l.nodes.has('1')).toBe(true);

    const lgrid = engine.computeGrid(g);
    expect(lgrid.nodes.has('1')).toBe(true);
  });
});
