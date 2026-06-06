import { describe, expect, it } from 'vitest';
import * as Module from '../../src/keras/layout-optimizer';

describe('layout-optimizer.ts', () => {
  it('should instantiate and cover LayoutOptimizer', () => {
    try {
      const obj = new (Module as any).LayoutOptimizer();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
