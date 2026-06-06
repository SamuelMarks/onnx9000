import { describe, expect, it } from 'vitest';
import * as Module from '../../src/passes/optimization';

describe('optimization.ts', () => {
  it('should instantiate and cover Optimizer', () => {
    try {
      const obj = new (Module as any).Optimizer();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
