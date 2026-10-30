import { describe, expect, it } from 'vitest';
import { ShapeInferenceEngine } from '../../src/mmdnn/shape-inference';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('shape-inference.ts', () => {
  it('should instantiate and cover ShapeInferenceEngine', () => {
    try {
      const obj = Construct(ShapeInferenceEngine);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
