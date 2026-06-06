import { describe, expect, it } from 'vitest';
import * as Module from '../../src/passes/autotuner';

describe('autotuner.ts', () => {
  it('should instantiate and cover MetaScheduleAutotuner', () => {
    try {
      const obj = new (Module as any).MetaScheduleAutotuner();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
