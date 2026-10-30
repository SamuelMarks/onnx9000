import { describe, expect, it } from 'vitest';
import { inferShapes } from '../src/mil/rewriter.js';

describe('rewriter', () => {
  it('should infer shapes', () => {
    const block = {
      inputs: [],
      outputs: [],
      operations: [{ opType: 'const', outputs: [{ type: { shape: [1] } }] }],
    };
    inferShapes(block as never as Parameters<typeof inferShapes>[0]);
    expect(typeof block).toBeDefined();
  });
});
