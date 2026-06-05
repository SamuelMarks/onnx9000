import { describe, it, expect } from 'vitest';
import { LayoutOptimizer } from '../src/compiler/layout.js';
import { Graph } from '@onnx9000/core';

describe('LayoutOptimizer', () => {
  it('should optimize', () => {
    const g = new Graph('test');
    g.nodes.push({ opType: 'Identity', inputs: ['a'], outputs: ['b'] } as any);

    const opt = new LayoutOptimizer(g);
    opt.optimize();
    expect(g.nodes.length).toBe(0);
  });
});
