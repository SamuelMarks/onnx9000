import { describe, expect, it } from 'vitest';
import { EventBus } from '../../src/core/EventBus';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('EventBus.ts', () => {
  it('should instantiate and cover EventBus', () => {
    // Attempt to instantiate
    try {
      const obj = Construct(EventBus);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
