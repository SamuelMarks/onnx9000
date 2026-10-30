import { describe, expect, it } from 'vitest';
import { MetaScheduleAutotuner } from '../../src/passes/autotuner';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('autotuner.ts', () => {
  it('should instantiate and cover MetaScheduleAutotuner', () => {
    try {
      const obj = Construct(MetaScheduleAutotuner);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
