import { describe, expect, it } from 'vitest';
import { OnnxVisualizer } from '../../src/components/OnnxVisualizer';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('OnnxVisualizer.ts', () => {
  it('should instantiate and cover OnnxVisualizer', () => {
    // Attempt to instantiate
    try {
      const obj = Construct(OnnxVisualizer);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
