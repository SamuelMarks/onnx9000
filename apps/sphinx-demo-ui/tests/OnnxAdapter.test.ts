// @ts-nocheck
import { describe, it, expect } from 'vitest';
import { OnnxAdapter } from '../src/core/OnnxAdapter.js';

describe('OnnxAdapter', () => {
  it('should convert to cytoscape', () => {
    const elements = OnnxAdapter.toCytoscape({
      inputs: [{ name: 'in', type: 'f32' }],
      outputs: [{ name: 'out', type: 'f32' }],
      nodes: [{ id: 'n1', name: 'n1', opType: 'Identity', inputs: ['in'], outputs: ['out'] }]
    });
    expect(elements.length).toBeGreaterThan(0);
  });
});
