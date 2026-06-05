import { describe, it, expect } from 'vitest';
import { ONNXNormalizer } from '../src/mmdnn/verification/normalizer.js';

describe('ONNXNormalizer', () => {
  it('should normalize graph', () => {
    const norm = new ONNXNormalizer();
    const g: any = {
      inputs: [{ name: 'in' }],
      outputs: [{ name: 'out' }],
      initializers: [],
      valueInfo: [],
      tensors: {},
      nodes: [{ opType: 'CaffeScale', name: 'n1', inputs: ['in'], outputs: ['out'] }],
    };

    norm.normalize(g);
    expect(g.nodes[0].opType).toBe('Mul');
    expect(norm.verifyParity()).toBe(true);
  });
});
