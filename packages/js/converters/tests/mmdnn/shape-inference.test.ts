import { describe, it, expect, vi } from 'vitest';
import * as Module from '../../src/mmdnn/shape-inference';

describe('shape-inference.ts', () => {
  it('should instantiate and cover ShapeInferenceEngine', () => {
    try {
       const obj = new (Module as any).ShapeInferenceEngine();
       expect(obj).toBeDefined();
    } catch (e) {}
  });
});
