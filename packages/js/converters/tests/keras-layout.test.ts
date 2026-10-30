/**
 * @fileoverview Unit tests for Keras memory layout tracking and redundant Transpose elimination.
 */
import { describe, expect, it } from 'vitest';
import { LayoutOptimizer, type OnnxNodeLike } from '../src/keras/layout-optimizer.js';

describe('Keras Phase 10 - Memory Layout & Explicit Propagation', () => {
  it('should track tensor layouts and detect when explicit transposition is needed', () => {
    const optimizer = new LayoutOptimizer();
    expect(optimizer.getLayout('untracked')).toBe('UNKNOWN');
    expect(optimizer.needsTranspose('untracked', 'NCHW')).toBe(false);

    optimizer.recordEdge('conv_in', 'InputLayer', 'Conv2D_1', 'NHWC');
    expect(optimizer.getLayout('conv_in')).toBe('NHWC');
    expect(optimizer.needsTranspose('conv_in', 'NCHW')).toBe(true);
    expect(optimizer.needsTranspose('conv_in', 'NHWC')).toBe(false);
    expect(optimizer.needsTranspose('conv_in', 'UNKNOWN')).toBe(false);
    expect(optimizer.edges).toHaveLength(1);
  });

  it('should eliminate cancelling NHWC -> NCHW -> NHWC Transpose pairs while preserving non-cancelling nodes', () => {
    const optimizer = new LayoutOptimizer();
    const nodes: OnnxNodeLike[] = [
      {
        opType: 'Transpose',
        inputs: ['x_nhwc'],
        outputs: ['x_nchw'],
        attributes: [{ name: 'perm', ints: [0, 3, 1, 2] }],
      },
      {
        opType: 'Transpose',
        inputs: ['x_nchw'],
        outputs: ['x_back_nhwc'],
        attributes: [{ name: 'perm', ints: [0, 2, 3, 1] }],
      },
      {
        opType: 'Relu',
        inputs: ['x_back_nhwc'],
        outputs: ['y'],
        attributes: [],
      },
    ];

    const optimized = optimizer.optimize(nodes);
    expect(optimized).toHaveLength(1);
    expect(optimized[0]?.opType).toBe('Relu');
  });
});
