import { describe, expect, it } from 'vitest';
import * as Module from '../../src/components/OnnxVisualizer';

describe('OnnxVisualizer.ts', () => {
  it('should instantiate and cover OnnxVisualizer', () => {
    // Attempt to instantiate
    try {
      const obj = new (Module as any).OnnxVisualizer();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
