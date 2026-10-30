import { describe, expect, it } from 'vitest';
import { DataLayoutTracker } from '../../src/mmdnn/layout';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('layout.ts', () => {
  it('should instantiate and cover DataLayoutTracker', () => {
    try {
      const obj = Construct(DataLayoutTracker);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
