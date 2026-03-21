import { describe, it, expect } from 'vitest';
import { LayoutOptimizer, OnnxNodeLike } from '../src/keras/layout-optimizer.js';

describe('LayoutOptimizer', () => {
  it('records and gets layout', () => {
    const optimizer = new LayoutOptimizer();
    expect(optimizer.getLayout('t1')).toBe('UNKNOWN');
    optimizer.recordEdge('t1', 'n1', 'n2', 'NCHW');
    expect(optimizer.getLayout('t1')).toBe('NCHW');
    expect(optimizer.needsTranspose('t1', 'NHWC')).toBe(true);
    expect(optimizer.needsTranspose('t1', 'NCHW')).toBe(false);
    expect(optimizer.needsTranspose('t1', 'UNKNOWN')).toBe(false);
    expect(optimizer.needsTranspose('t2', 'NCHW')).toBe(false); // t2 is UNKNOWN
  });

  it('optimizes adjacent identity transposes', () => {
    const optimizer = new LayoutOptimizer();
    const nodes: OnnxNodeLike[] = [
      {
        opType: 'Transpose',
        attributes: [{ name: 'perm', ints: [0, 2, 3, 1] }],
      },
      {
        opType: 'Transpose',
        attributes: [{ name: 'perm', ints: [0, 3, 1, 2] }],
      },
      {
        opType: 'Relu',
        attributes: [],
      },
    ];
    const optimized = optimizer.optimize(nodes);
    expect(optimized).toHaveLength(1);
    expect(optimized[0].opType).toBe('Relu');
  });

  it('does not optimize non-identity transposes', () => {
    const optimizer = new LayoutOptimizer();
    const nodes: OnnxNodeLike[] = [
      {
        opType: 'Transpose',
        attributes: [{ name: 'perm', ints: [0, 2, 3, 1] }],
      },
      {
        opType: 'Transpose',
        attributes: [{ name: 'perm', ints: [0, 2, 3, 1] }],
      },
    ];
    const optimized = optimizer.optimize(nodes);
    expect(optimized).toHaveLength(2);
  });

  it('does not optimize if attributes are missing', () => {
    const optimizer = new LayoutOptimizer();
    const nodes: OnnxNodeLike[] = [
      {
        opType: 'Transpose',
        attributes: [],
      },
      {
        opType: 'Transpose',
        attributes: [],
      },
    ];
    const optimized = optimizer.optimize(nodes);
    expect(optimized).toHaveLength(2);
  });
});
