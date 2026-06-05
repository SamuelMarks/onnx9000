import { describe, it, expect } from 'vitest';
import { mapOnnxNodeToTFLite, ELEMENTWISE_OPS } from '../src/compiler/operators.js';

describe('operators', () => {
  it('should map', () => {
    expect(mapOnnxNodeToTFLite({ opType: 'Add', attributes: {} } as any)).toBeDefined();
    expect(mapOnnxNodeToTFLite({ opType: 'Unknown' } as any)).toBeNull();
  });
});
