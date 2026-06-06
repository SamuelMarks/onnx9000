// @ts-nocheck
import { describe, expect, it } from 'vitest';
import { OnnxAstFormatter } from '../src/core/OnnxAstFormatter.js';

describe('OnnxAstFormatter', () => {
  it('should format ast', () => {
    const text = OnnxAstFormatter.format({
      inputs: [{ name: 'in', type: 'f32' }],
      outputs: [{ name: 'out', type: 'f32' }],
      nodes: [
        {
          id: 'n1',
          name: 'n1',
          opType: 'Identity',
          inputs: ['in'],
          outputs: ['out'],
          attributes: { test: 123 },
        },
      ],
    });
    expect(text).toContain('op_type: "Identity"');
    expect(text).toContain('test');
    expect(text).toContain('123');
  });
});
